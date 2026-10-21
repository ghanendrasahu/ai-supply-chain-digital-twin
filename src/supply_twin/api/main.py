"""FastAPI application exposing the digital twin as a service."""

from __future__ import annotations

import logging
import secrets
from contextlib import asynccontextmanager
from threading import BoundedSemaphore
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request

from supply_twin.api.deps import (
    get_result,
    get_settings_dep,
    get_twin,
    store_result,
)
from supply_twin.api.rate_limit import FixedWindowRateLimiter
from supply_twin.config import Settings
from supply_twin.logging_setup import configure_logging
from supply_twin.reporting.report import build_report, render_markdown
from supply_twin.schemas import (
    DemandPoint,
    DemandSeries,
    Intervention,
    KPISnapshot,
    SimulationResult,
)
from supply_twin.twin import SupplyChainTwin

logger = logging.getLogger("supply_twin.api")

# Health/docs stay probe-accessible; everything else may require a key.
_PUBLIC_PATHS = frozenset({"/health", "/docs", "/redoc", "/openapi.json"})


def require_api_key(request: Request, x_api_key: Annotated[str | None, Header()] = None) -> None:
    """Gate every route unless ``api_keys`` is empty (auth off) or public."""
    keys = get_settings_dep().api_keys
    if not keys or request.url.path in _PUBLIC_PATHS:
        return
    if not x_api_key or not any(
        secrets.compare_digest(x_api_key.encode(), key.encode()) for key in keys
    ):
        logger.warning("rejected client: invalid or missing API key path=%s", request.url.path)
        raise HTTPException(status_code=401, detail="Missing or invalid API key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings_dep()
    configure_logging(json_logs=settings.log_json, level=settings.log_level)
    logger.info(
        "starting supply-twin v%s (env=%s log_json=%s)",
        app.version,
        settings.env,
        settings.log_json,
    )
    logger.info(
        "bounds: horizon_days=%d monte_carlo_runs=%d default_service_level=%s",
        settings.horizon_days,
        settings.monte_carlo_runs,
        settings.default_service_level,
    )
    yield
    logger.info("shutdown")


app = FastAPI(
    title="AI Supply Chain Digital Twin",
    description="Demand forecasting, inventory optimisation and a what-if "
    "disruption simulator over the Supplier→Factory→Warehouse→Distribution→"
    "Retailer→Customer chain.",
    version="0.1.0",
    lifespan=lifespan,
    dependencies=[Depends(require_api_key)],
)

# CPU protection: per-client request quota (worker-local by design).
app.add_middleware(FixedWindowRateLimiter)


@app.get("/health", tags=["system"])
def health(settings: Annotated[Settings, Depends(get_settings_dep)]) -> dict[str, str]:
    return {"status": "ok", "env": settings.env, "service": "supply-twin"}


@app.get("/network", tags=["data"])
def network(twin: Annotated[SupplyChainTwin, Depends(get_twin)]) -> dict:
    nodes = [n.model_dump() for n in twin.network.nodes.values()]
    lanes = [lane.model_dump() for lane in twin.network.lanes]
    return {"nodes": nodes, "lanes": lanes}


@app.get("/skus", tags=["data"])
def skus(twin: Annotated[SupplyChainTwin, Depends(get_twin)]) -> list[dict]:
    return [s.model_dump() for s in twin.network.skus.values()]


@app.get("/demand/forecast", tags=["intelligence"])
def demand_forecast(
    twin: Annotated[SupplyChainTwin, Depends(get_twin)],
    retailer: str = "RET-1",
    sku: str = "SKU-001",
    horizon: int = Query(default=30, ge=1, le=365),
) -> DemandSeries:
    if (retailer, sku) not in twin.demand_split:
        raise HTTPException(status_code=404, detail=f"No demand series for {retailer}/{sku}")
    result = twin.forecast(retailer, sku, horizon)
    return DemandSeries(
        node_id=result.node_id,
        sku=result.sku,
        points=[
            DemandPoint(day=d, quantity=float(q))
            for d, q in zip(result.dates, result.forecast, strict=False)
        ],
    )


@app.get("/inventory/optimize", tags=["intelligence"])
def inventory_optimize(
    twin: Annotated[SupplyChainTwin, Depends(get_twin)],
    sku: str = "SKU-001",
    node: str = "WH-1",
    service_level: float | None = Query(default=None, gt=0.5, lt=1.0),
) -> dict:
    if sku not in twin.network.skus:
        raise HTTPException(status_code=404, detail=f"Unknown SKU {sku}")
    if node not in twin.network.nodes:
        raise HTTPException(status_code=404, detail=f"Unknown node {node}")
    policy = twin.optimize(sku, node, service_level)
    return policy.__dict__


@app.get("/kpis/current", tags=["intelligence"])
def current_kpis(twin: Annotated[SupplyChainTwin, Depends(get_twin)]) -> dict[str, float]:
    return twin.kpis()


# CPU-bound simulations run in the thread pool; bound their concurrency so a
# client cannot saturate the worker with parallel 20k-run Monte Carlos.
_whatif_semaphore = BoundedSemaphore(get_settings_dep().what_if_max_concurrent)


@app.post("/simulate/what-if", tags=["simulation"])
def what_if(
    interventions: list[Intervention],
    twin: Annotated[SupplyChainTwin, Depends(get_twin)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
    horizon: int | None = Query(default=None, ge=1, le=730),
    n_runs: int | None = Query(default=None, ge=1, le=20000),
    label: str | None = None,
) -> SimulationResult:
    if not _whatif_semaphore.acquire(blocking=False):
        logger.warning(
            "what-if rejected: concurrency cap %d reached", settings.what_if_max_concurrent
        )
        raise HTTPException(status_code=503, detail="Simulator busy; retry shortly")
    try:
        result = twin.what_if(
            interventions=interventions,
            horizon=horizon,
            n_runs=n_runs,
            label=label,
        )
        store_result(result)
        logger.info(
            "what-if complete: run_id=%s horizon=%d n_runs=%d interventions=%d",
            result.run_id,
            horizon or settings.horizon_days,
            n_runs or settings.monte_carlo_runs,
            len(interventions),
        )

        settings.ensure_dirs()
        build_report(result, settings, settings.report_dir, kpis=twin.kpis())
        logger.info("report generated for run_id=%s", result.run_id)
    finally:
        _whatif_semaphore.release()
    return result


@app.get("/simulate/baseline", tags=["simulation"])
def baseline(
    twin: Annotated[SupplyChainTwin, Depends(get_twin)],
    horizon: int | None = Query(default=None, ge=1, le=730),
) -> list[KPISnapshot]:
    return twin.run(interventions=[], horizon=horizon, seed=twin.settings.random_seed)


@app.get("/scenarios/templates", tags=["simulation"])
def scenario_templates() -> list[dict]:
    return [
        {
            "name": "supplier_failure_14d",
            "description": "Supplier A offline for 14 days (canonical ₹2.7Cr scenario)",
            "interventions": [
                {
                    "kind": "supplier_failure",
                    "supplier_id": "SUP-A",
                    "duration_days": 14,
                    "severity": 1.0,
                }
            ],
        },
        {
            "name": "demand_surge_20pct",
            "description": "20% demand surge for 30 days",
            "interventions": [{"kind": "demand_shock", "magnitude": 0.20, "duration_days": 30}],
        },
        {
            "name": "lane_delay",
            "description": "Port congestion: DC→Retailer lanes +5 days",
            "interventions": [
                {
                    "kind": "transport_delay",
                    "lane_origin": "DC-1",
                    "lane_destination": "RET-1",
                    "added_days": 5.0,
                }
            ],
        },
    ]


@app.get("/report/{run_id}", tags=["reporting"])
def report(
    run_id: str,
    settings: Annotated[Settings, Depends(get_settings_dep)],
    format: str = Query(default="markdown", pattern="^(markdown|json)$"),
) -> str | dict:
    result = get_result(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Unknown run {run_id}")
    if format == "json":
        return result.model_dump()
    return render_markdown(result, settings)
