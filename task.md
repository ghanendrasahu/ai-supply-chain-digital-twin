# AI Supply Chain Digital Twin — Full Build Plan (task.md)

> A production-grade, simulation-first digital twin of an end-to-end supply chain
> with demand forecasting, inventory optimization and a **what-if disruption
> simulator** that turns a single event ("Supplier A fails for 14 days") into a
> quantified financial consequence chain.

---

## 0. Product Vision

Decision intelligence for supply chain leaders. The system models the physical
chain:

```
Supplier ──▶ Factory ──▶ Warehouse ──▶ Distribution ──▶ Retailer ──▶ Customer
```

and simulates how shocks propagate through it:

```
Supplier failure
   ▼
Production   −18%
   ▼
Inventory    −31%
   ▼
Stockout P   +42%
   ▼
Revenue      −₹2.7Cr
```

The headline deliverable is a reproducible, tested, containerised service that a
data scientist / ML engineer can extend.

---

## 1. Goals & Non-Goals

### Goals

- [x] Model all 6 echelons of the supply chain as a connected network.
- [x] Demand forecasting per SKU/region (seasonality + trend + noise).
- [x] Inventory optimization (safety stock, reorder point, EOQ).
- [x] Stochastic supplier-failure and delivery-delay models.
- [x] Discrete-event, multi-step what-if simulator.
- [x] Consequence-chain engine that maps an intervention to $/₹ impact.
- [x] Monte-Carlo uncertainty bands (1 000+ runs).
- [x] FastAPI service + CLI + generated report.
- [x] Full test suite; reproducible install; Docker.

### Non-Goals (v1)

- Live ERP/WMS connectors (interfaces stubbed for v2).
- Real geospatial routing engine (lane model is statistical).
- Deep-learning forecasting beyond classical + gradient boosting fallback.

---

## 2. Tech Stack

| Layer            | Choice                                    | Why                                    |
| ---------------- | ----------------------------------------- | -------------------------------------- |
| Language         | Python 3.11                               | User's primary stack                   |
| Data             | pandas, numpy, scipy                      | Core DS toolkit                        |
| Forecasting      | statsmodels (seasonal) + gradient boost   | Robust, no heavy deps, offline-safe    |
| Simulation       | SimPy + custom Monte-Carlo runner         | Standard DES, deterministic RNG seeds  |
| Graph            | networkx                                  | Supply-chain topology & propagation    |
| Validation       | pydantic v2                               | Typed contracts across the system      |
| API              | FastAPI + uvicorn                         | Async, auto OpenAPI docs               |
| Config           | pydantic-settings                         | 12-factor config                       |
| Reporting        | matplotlib + pandas → HTML/Markdown       | Offline, no browser needed             |
| Testing          | pytest + pytest-cov                       | Industry standard                      |
| Packaging        | pyproject.toml (setuptools) + requirements| pip-installable, no Poetry needed      |
| Containers       | Docker + docker-compose                   | Reproducible deploy                    |
| CI               | GitHub Actions                            | Lint + test on push                    |
| VCS              | Git + GitHub (gh CLI)                     | Commit & push per user identity        |

> **Note:** Poetry is not installed on this machine, so packaging uses
> `pyproject.toml` (PEP 621) with setuptools + a pinned `requirements.txt` for
> fast, deterministic installs.

---

## 3. Repository Layout

```
xyz/
├── task.md                     # this plan
├── README.md                   # usage & architecture
├── pyproject.toml              # PEP 621 package metadata
├── requirements.txt            # pinned runtime deps
├── requirements-dev.txt        # test/lint deps
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── .env.example
├── .gitignore
├── .github/workflows/ci.yml
├── src/supply_twin/
│   ├── __init__.py
│   ├── config.py               # settings
│   ├── schemas.py              # pydantic domain models
│   ├── network.py              # supply-chain topology (networkx)
│   ├── data/
│   │   ├── __init__.py
│   │   └── generator.py        # synthetic history generator
│   ├── forecasting/
│   │   ├── __init__.py
│   │   └── demand.py           # demand forecasting
│   ├── inventory/
│   │   ├── __init__.py
│   │   └── optimizer.py        # safety stock / EOQ / ROP
│   ├── risk/
│   │   ├── __init__.py
│   │   └── models.py           # failure + delay stochastic models
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── engine.py           # DES what-if engine
│   │   └── consequences.py     # consequence-chain engine
│   ├── reporting/
│   │   ├── __init__.py
│   │   └── report.py           # KPI + report + charts
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py             # FastAPI app
│   │   └── deps.py             # shared deps / twin factory
│   └── cli.py                  # `supply-twin` CLI
├── scripts/
│   └── run_demo.py             # end-to-end demo (the ₹2.7Cr scenario)
├── tests/
│   ├── conftest.py
│   ├── test_generator.py
│   ├── test_forecasting.py
│   ├── test_inventory.py
│   ├── test_risk.py
│   ├── test_simulation.py
│   ├── test_consequences.py
│   └── test_api.py
└── data/                       # generated artifacts (gitignored)
```

---

## 4. Domain Model (schemas.py)

- `NodeType` enum: SUPPLIER, FACTORY, WAREHOUSE, DISTRIBUTION, RETAILER, CUSTOMER
- `SupplyNode`: id, name, type, region, capacity, reliability, lead_time_days
- `SKU`: code, name, category, unit_cost, price, elasticity
- `InventoryState`: node_id, sku, on_hand, in_transit, reorder_point, safety_stock
- `Lane`: origin, destination, transit_days, transit_std, cost_per_unit
- `DemandSeries`: node_id, sku, dates, actual, forecast
- `Intervention` (union): supplier_failure, demand_shock, transport_delay,
  price_change, factory_outage
- `ConsequenceStep`: label, metric, delta_pct, absolute, narrative
- `SimulationResult`: kpis, timeline, consequence_chain, monte_carlo bands

---

## 5. Core Algorithms

### 5.1 Demand forecasting (`forecasting/demand.py`)
- Decompose into trend + weekly + annual seasonality + noise.
- Baseline: classical seasonal-naive / Holt-Winters when history is short.
- `GradientBoostForecaster` (sklearn HistGradientBoostingRegressor) using
  lag/rolling/calendar features when enough history.
- Output: point forecast + prediction interval (`residual std × z`).
- Metrics: MAPE, RMSE, bias.

### 5.2 Inventory optimization (`inventory/optimizer.py`)
- Safety stock: `SS = z * σ_d * sqrt(LT)`.
- Reorder point: `ROP = μ_d * LT + SS`.
- EOQ: `sqrt(2 * D * S / H)`.
- Service level → z via `scipy.stats.norm.ppf`.
- Fill-rate & expected-shortage estimation with normal loss function.

### 5.3 Risk models (`risk/models.py`)
- Supplier failure: Bernoulli/Markov availability + Weibull time-to-recover.
- Delivery delay: lognormal lane transit; mode-switch penalty.
- Deterministic seed plumbing for reproducibility.

### 5.4 Simulation engine (`simulation/engine.py`)
- Discrete-event over daily ticks for a configurable horizon (default 90 d).
- State: per-node per-SKU inventory, work-in-progress, pending shipments.
- Each tick: apply interventions → produce → ship (lead-time lanes) →
  receive → sell (min(demand, on_hand)) → record KPIs.
- Capacity constraints, backorder vs lost-sales policy.

### 5.5 Consequence chain (`simulation/consequences.py`)
- Runs baseline vs intervention (paired, common random numbers).
- Computes deltas: production %, inventory %, stockout probability,
  revenue ₹ impact.
- Monte-Carlo across N seeds → mean + p10/p90 bands.
- Emits the exact narrative chain:

  ```
  Supplier failure → Production −18% → Inventory −31%
                   → Stockout +42% → Revenue −₹2.7Cr
  ```

### 5.6 Calibration
- Calibration constants live in `config.py` and are overridable by env:
  - bullwhip/decay multiplier, affected-customer share, horizon.
- The demo scenario is tuned so that the canonical 14-day Supplier-A failure
  reproduces the headline numbers within tolerance (validated in tests).

---

## 6. API Surface (FastAPI)

| Method | Path                         | Purpose                               |
| ------ | ---------------------------- | ------------------------------------- |
| GET    | `/health`                    | liveness                              |
| GET    | `/network`                   | nodes + lanes topology                |
| GET    | `/skus`                      | SKU catalogue                         |
| GET    | `/demand/forecast`           | forecast for a SKU/node               |
| POST   | `/inventory/optimize`        | SS/ROP/EOQ for a SKU/node             |
| POST   | `/simulate/what-if`          | run intervention(s) + consequence     |
| GET    | `/scenarios/templates`       | canned scenarios                      |
| GET    | `/report/{run_id}`           | generated report (HTML/Markdown/JSON) |

---

## 7. CLI (`supply-twin`)

```
supply-twin demo                     # end-to-end ₹2.7Cr scenario
supply-twin simulate --event supplier_failure --supplier "Supplier A" --days 14
supply-twin forecast --sku SKU-001 --node RETAIL-01 --horizon 30
supply-twin optimize --sku SKU-001 --node WAREHOUSE-01
supply-twin serve --host 0.0.0.0 --port 8000
```

---

## 8. Testing & Quality Gates

- Unit tests for every module (generator, forecasting, inventory, risk,
  simulation, consequences, API).
- Property/invariant tests: non-negative inventory, conservation of flow.
- Reproducibility: same seed ⇒ identical results.
- Canonical-scenario test: 14-day Supplier-A failure reproduces headline
  magnitudes within tolerance.
- Coverage target ≥ 85 %.
- Lint: ruff. Format: ruff format. Types: optional mypy.
- CI workflow runs `ruff` + `pytest` on Python 3.11.

---

## 9. Build / Run / Deploy

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
pip install -r requirements-dev.txt
pip install -e .
pytest -q
python scripts/run_demo.py       # prints ₹ consequence chain + writes report
uvicorn supply_twin.api.main:app --reload
docker compose up --build
```

---

## 10. Deliverables Checklist

- [x] `task.md` build plan
- [x] Installable package `supply_twin`
- [x] Domain schemas + config
- [x] Synthetic data generator
- [x] Demand forecasting
- [x] Inventory optimization
- [x] Risk (failure + delay) models
- [x] DES what-if engine + Monte-Carlo
- [x] Consequence-chain engine (₹2.7Cr narrative)
- [x] FastAPI service + CLI + demo script
- [x] Reporting (KPIs, charts, markdown/HTML)
- [x] Tests (≥85 % coverage) + CI
- [x] Docker + compose + Makefile
- [x] README
- [x] GitHub repo + commits/push

---

## 11. Milestones (execution order in this session)

1. **M0 – Plan** — write `task.md`, scaffold, git init.
2. **M1 – Domain** — schemas, config, network, generator.
3. **M2 – Intelligence** — forecasting, inventory, risk.
4. **M3 – Simulation** — engine + consequence chain + Monte-Carlo.
5. **M4 – Interfaces** — API, CLI, reporting, demo.
6. **M5 – Quality** — tests, CI, lint, coverage.
7. **M6 – Ship** — README, Docker, commit, push.

---

## 12. Definition of Done

- One command reproduces the headline consequence chain.
- `pytest` green, coverage ≥ 85 %.
- API serves OpenAPI docs; demo script writes a report artifact.
- Repo pushed to GitHub under identity `ghanendrasahu`.
