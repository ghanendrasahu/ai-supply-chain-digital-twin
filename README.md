# 🥉 AI Supply Chain Digital Twin

A production-grade, simulation-first digital twin of an end-to-end supply chain with **demand forecasting**, **inventory optimization**, and a **what-if disruption simulator** that translates operational shocks into quantified financial consequence chains.

```text
Supplier ──▶ Factory ──▶ Warehouse ──▶ Distribution ──▶ Retailer ──▶ Customer
```

---

## 🎯 What-If Decision Intelligence

Ask real-world questions like:

> **"What happens if Supplier A fails for 14 days?"**

The system executes a discrete-event, multi-echelon simulation across a Monte-Carlo ensemble and produces the exact financial consequence chain:

```text
Supplier failure
       ↓
Production -18%
       ↓
Inventory -31%
       ↓
Stockout probability +42%
       ↓
Revenue impact -₹2.7Cr
```

---

## 🌟 Key Features

1. **Full 6-Echelon Supply Chain Graph**
   - Modeled via `networkx` directed graph with lead times, standard deviations, transit costs, and capacity constraints.

2. **Demand Forecasting Module**
   - Classical seasonal-naive baseline & HistGradientBoostingRegressor recursive multi-step forecasting with lag, rolling-window, and calendar features.
   - Prediction intervals and automatic in-sample scoring (MAPE, RMSE).

3. **Continuous-Review (Q, R) Inventory Optimization**
   - Analytic calculations:
     - Safety Stock: $SS = z \cdot \sigma_d \cdot \sqrt{LT}$
     - Reorder Point: $ROP = \mu_d \cdot LT + SS$
     - Economic Order Quantity: $EOQ = \sqrt{\frac{2 \cdot D \cdot S}{H}}$
     - Cycle fill-rate via the standard Normal Loss function $L(z)$.

4. **Stochastic Risk Models**
   - Markov-availability supplier failure process with Weibull recovery distributions.
   - Lognormal transit time distributions with congestion factors.

5. **Discrete-Event What-If Simulator**
   - Paired runs with Common Random Numbers (CRN) for variance reduction.
   - Monte-Carlo uncertainty bands (p10, p50, p90, mean).
   - Multi-event support: supplier failure, factory outage, demand shock, transport delays, price changes.

6. **Prescriptive AI Mitigation Advisor**
   - Automated counterfactual evaluation of response playbooks (secondary supplier expediting, safety buffer shifts, demand shaping).
   - Ranked mitigation recommendations sorted by Net Recovered Benefit, ROI multiple, and implementation feasibility.

7. **Production Interfaces & Web UI**
   - **Streamlit Web Dashboard**: Interactive simulation console, dynamic consequence chain cards, and mitigation playbook generator.
   - **FastAPI** REST backend with OpenAPI/Swagger docs at `/docs`.
   - **ERP Connectors**: Ingestion schemas and validators for CSV and JSON enterprise payloads.
   - **Typer & Rich** CLI (`supply-twin demo`, `supply-twin simulate`, `supply-twin ui`, etc.).
   - Headless automated report generation with Markdown and Matplotlib charts.
   - Docker & Docker Compose setup.

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/ghanendrasahu/ai-supply-chain-digital-twin.git
cd ai-supply-chain-digital-twin

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # Linux / macOS

# Install package and dependencies
pip install -r requirements-dev.txt
pip install -e .
```

### 2. Launch Interactive Streamlit UI

```bash
supply-twin ui
# or: streamlit run src/supply_twin/ui/app.py
```

### 3. Run the Headline What-If Demo

```bash
python scripts/run_demo.py
```

Or using the CLI:

```bash
supply-twin demo
```

### 3. Start the API Service

```bash
uvicorn supply_twin.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Visit the interactive OpenAPI docs at [http://localhost:8000/docs](http://localhost:8000/docs).

---

## 🛠️ CLI Usage

```bash
# Run the canonical 14-day supplier failure demo
supply-twin demo --days 14 --runs 1000

# Run custom what-if disruptions
supply-twin simulate --event demand_shock --magnitude 0.25 --days 30
supply-twin simulate --event transport_delay --days 5

# Demand forecasting
supply-twin forecast --retailer RET-1 --sku SKU-001 --horizon 30

# Inventory policy optimization
supply-twin optimize --sku SKU-001 --node WH-1 --service-level 0.95

# View steady-state KPIs
supply-twin kpis
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Service health status |
| `GET` | `/network` | Graph topology (nodes & lanes) |
| `GET` | `/skus` | SKU catalogue with margin & price elasticity |
| `GET` | `/demand/forecast` | Point & interval forecast for a retailer/SKU |
| `GET` | `/inventory/optimize` | (Q, R) recommended policy |
| `GET` | `/kpis/current` | Steady-state system KPIs |
| `POST` | `/simulate/what-if` | Run multi-event disruption simulation |
| `GET` | `/scenarios/templates` | Pre-configured scenario presets |
| `GET` | `/report/{run_id}` | Retrieve generated report (Markdown/JSON) |

---

## 🧪 Testing & Code Quality

```bash
# Run pytest with code coverage
pytest

# Code formatting and linting
ruff check src tests
ruff format src tests
```

---

## 🐳 Docker Deployment

```bash
docker compose up --build
```

---

## 📄 License

MIT © [ghanendrasahu](https://github.com/ghanendrasahu)
