# CLAUDE.md - Analytics Engine

## Project Overview

A production-grade, domain-agnostic analytics platform with a **FastAPI backend** and **React + TypeScript frontend**. It transforms raw business data into statistical insights through automated EDA, interactive exploration, and plain-English narrative generation. This is an internal Rystad Energy tool.

## Architecture

```
analytics-engine/
├── api/                        # FastAPI backend
│   ├── main.py                 # App entry point, CORS, router registration
│   ├── session_store.py        # In-memory session store (UUID → SessionData)
│   ├── middleware.py            # Session cookie middleware
│   ├── dependencies.py         # get_session(), require_data() DI
│   ├── models/
│   │   ├── requests.py         # Pydantic request schemas
│   │   └── responses.py        # Pydantic response schemas
│   └── routers/
│       ├── data_source.py      # /api/data/* (upload, connect, load, query, sample)
│       ├── profiling.py        # /api/profile/* (overview, columns)
│       ├── statistics.py       # /api/stats/* (correlation, trend, regression, ttest, anova)
│       └── visualizations.py   # /api/viz/* (heatmap, distribution, scatter, boxplot)
├── frontend/                   # React + TypeScript + Vite
│   └── src/
│       ├── App.tsx             # QueryClientProvider root
│       ├── lib/api.ts          # Axios API client
│       ├── types/              # TypeScript interfaces
│       └── components/         # Dashboard, Sidebar, tabs, PlotlyChart
├── core/
│   ├── profiling.py            # Automated EDA, type detection, quality checks
│   ├── statistics.py           # Correlation, regression, t-test, ANOVA, trend detection
│   ├── insights.py             # Technical → plain-English narrative translation
│   └── visualizations.py       # Plotly interactive charts (9 chart types)
├── db/
│   ├── connection.py           # DatabaseClient wrapper (lazy init)
│   ├── connect_to_sql.py       # SQL Server connection via pyodbc/SQLAlchemy
│   ├── introspection.py        # Schema discovery (tables, columns, row counts)
│   ├── loader.py               # Table → DataFrame loading
│   └── query.py                # Read-only SQL validation (security layer)
├── config/settings.py          # Singleton config from environment variables
├── run_backend.py              # Uvicorn launcher
├── models/                     # Placeholder for future ML models
└── tests/
    ├── conftest.py             # Pytest fixtures
    ├── test_profiling.py       # 15 tests across 5 test classes
    └── test_statistics.py      # 16 tests across 5 test classes
```

### Key Design Patterns
- **Separation of concerns**: API (api/) | Frontend (frontend/) | Analytics (core/) | Data Access (db/)
- **Session-based state**: UUID cookie → in-memory SessionData (config, db_client, DataFrame, profile)
- **Singleton config**: `Config` class in config/settings.py
- **Lazy initialization**: DatabaseClient creates connection on first use
- **Dataclasses for results**: CorrelationResult, RegressionResult, TTestResult, ANOVAResult
- **Static insight generator**: InsightGenerator uses @staticmethod methods

## Commands

### Run the Backend
```bash
python run_backend.py
```
API launches at http://localhost:8000 (Swagger docs at /docs)

### Run the Frontend
```bash
cd frontend && npm run dev
```
UI launches at http://localhost:5173 (proxies /api to backend)

### Install Dependencies
```bash
# Backend
poetry install

# Frontend
cd frontend && npm install
```

### Run Tests
```bash
# All tests
poetry run pytest tests/ -v

# With coverage
poetry run pytest tests/ --cov=core --cov=db --cov-report=html

# Single test file
poetry run pytest tests/test_profiling.py -v
poetry run pytest tests/test_statistics.py -v
```

## Dependencies

### Backend
- **Python**: >=3.11
- **Package manager**: Poetry 2.0+
- **API**: FastAPI, uvicorn, python-multipart
- **Data**: pandas, numpy
- **Statistics**: scipy, statsmodels, scikit-learn
- **Visualization**: plotly
- **Database**: sqlalchemy, pyodbc (SQL Server), psycopg2-binary (PostgreSQL), pymysql (MySQL)
- **Config**: python-dotenv
- **Testing**: pytest, pytest-cov

### Frontend
- **Runtime**: Node.js 20+
- **Framework**: React 19, TypeScript, Vite
- **Styling**: Tailwind CSS v4
- **Data fetching**: @tanstack/react-query, axios
- **Charts**: react-plotly.js
- **Icons**: lucide-react

## Configuration

Environment variables loaded from `.env` (see `.env.example`):

| Variable | Purpose | Default |
|----------|---------|---------|
| `DB_HOST` | Database server | ekofisk |
| `DB_PORT` | Database port | 1433 |
| `DB_DRIVER` | SQLAlchemy driver | mssql+pyodbc |
| `SQL_TRUSTED_CONNECTION` | Windows auth | Yes |
| `CORRELATION_THRESHOLD` | Min correlation for insights | 0.3 |
| `P_VALUE_THRESHOLD` | Statistical significance level | 0.05 |
| `OUTLIER_IQR_MULTIPLIER` | IQR multiplier for outliers | 1.5 |
| `MAX_ROWS_DISPLAY` | UI row display limit | 10000 |
| `SAMPLE_SIZE` | Sampling for large datasets | 1000 |
| `API_HOST` | Backend bind address | 0.0.0.0 |
| `API_PORT` | Backend port | 8000 |

## Development Guidelines

### Adding a New Analysis
1. Add the statistical method to `core/statistics.py` (return a dataclass)
2. Add narrative generation to `core/insights.py` (static method on InsightGenerator)
3. Add visualization to `core/visualizations.py` (return `go.Figure`)
4. Add a FastAPI endpoint in `api/routers/` with Pydantic request/response models
5. Add the API call to `frontend/src/lib/api.ts`
6. Build the UI component in `frontend/src/components/`
7. Add tests in `tests/`

### Adding a New Chart Type
Add a function to `core/visualizations.py` following the pattern:
```python
def create_<chart_type>(...) -> go.Figure:
```
Then add a corresponding endpoint in `api/routers/visualizations.py` using `_fig_response(fig)`.

### Database Security
- All user SQL goes through `db/query.py` → `validate_read_only_query()`
- Only SELECT and WITH (CTE) queries are allowed
- FORBIDDEN_KEYWORDS blocks DELETE, DROP, UPDATE, INSERT, ALTER, etc.
- Never bypass this validation layer

### Code Style
- PEP 8 compliant
- Type hints on all function signatures
- Docstrings on all public functions
- Use dataclasses for structured results
- Keep functions focused and modular

### Testing
- Fixtures in `tests/conftest.py` provide reproducible DataFrames (seeded with np.random.seed)
- Test classes group related tests (e.g., TestCorrelation, TestLinearRegression)
- Test both happy paths and edge cases (empty DataFrames, nulls, insufficient data)

## Data Flow

```
Data Source (CSV/Excel/SQL)
    → POST /api/data/* → pandas DataFrame → SessionData
    → GET /api/profile/* → core/profiling.py (DatasetProfile)
    → GET/POST /api/stats/* → core/statistics.py (correlation, regression, trends)
    → GET /api/stats/insights → core/insights.py (plain-English narratives)
    → GET /api/viz/* → core/visualizations.py (Plotly figures as JSON)
    → React frontend renders: Overview, Relationships, Trends, Insights tabs
```

## Color Scheme
- **Navy blue** (#1a1f36) — 85%: backgrounds, cards, sidebar
- **Orange** (#f97316) — 15%: accent, active tabs, primary buttons, headings
- Navy bg → white text; Orange bg → black text

## Current Status

Phase 2 complete (FastAPI + React migration). Core analytics unchanged. Phase 3 planned: schema intelligence, anomaly detection, NL queries, key driver analysis.
