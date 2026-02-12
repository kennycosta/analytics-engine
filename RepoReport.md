# Repo Report: Analytics Engine

_Last updated: February 11, 2026_

## Project Overview
- **Purpose:** A production-grade, domain-agnostic analytics platform that automates exploratory data analysis, statistical modeling, narrative insight generation, and interactive visualization on top of existing data infrastructure.
- **Positioning:** Acts as an intelligence layer rather than a static dashboard, targeting rapid, code-free insight delivery for business stakeholders.
- **Primary Interfaces:** Streamlit web UI (`app/main.py`) backed by modular analytics engines (`core/`) and a SQL connectivity layer (`db/`).

## Tech Stack & Dependencies
- **Language & Runtime:** Python ≥ 3.11, managed via Poetry (see `pyproject.toml`).
- **UI:** Streamlit 1.53 (wide layout, custom session-state helpers, cached profiling).
- **Data & Stats:** pandas, numpy, scipy, statsmodels, scikit-learn, plotly, openpyxl.
- **Database Access:** SQLAlchemy + pyodbc/pymysql/psycopg2 adapters through a company-provided connector wrapper.
- **Configuration:** `.env`-driven settings loaded into `config.settings.Config` with strongly-typed dataclasses.
- **Testing:** pytest + pytest-cov (configured under `[dependency-groups.dev]`).

## Repository Layout
```
app/             Streamlit UI entry point and layout logic
config/          Environment-aware configuration objects
core/            Analytics engines: profiling, statistics, insights, viz
    profiling.py Automated dataset/column profiling & quality checks
    statistics.py Correlations, regression, hypothesis tests, ANOVA, trend detection
    insights.py  Narrative generation from statistical results
    visualizations.py Plotly chart builders and helpers
db/              SQL Server connectivity, introspection, loaders, guarded querying
models/          Placeholder for future ML artifacts
tests/           pytest suites for profiling/statistics + fixtures in conftest
```

## Application Architecture
1. **UI Layer (`app/main.py`):**
   - Configures Streamlit page metadata, session state, upload/SQL/sample data ingestion, filtering widgets, and tabbed navigation (Overview, Relationships, Trends, Geospatial, Insights).
   - Wraps expensive operations (profiling, correlations, distribution figures) with caching keyed by dataset/filter signatures.
   - Integrates visualization factories, statistical routines, and insight generators to render real-time analytics plus plain-language callouts.
2. **Core Analytics (`core/`):**
   - `profiling.py`: builds `DatasetProfile`/`ColumnProfile` dataclasses, detects column semantics, computes numeric/categorical/datetime stats, flags issues (null density, uniqueness, outliers) and calculates completeness + correlation matrices.
   - `statistics.py`: houses dataclasses for every test result plus implementations for Pearson/Spearman/Kendall correlations, linear regression, multiple flavors of t-tests, non-parametric rank tests, ANOVA (one-way, two-way, Tukey HSD), chi-square, Levene’s variance checks, trend detection, and z-score outlier detection.
   - `insights.py`: translates profiles and statistical outputs into business-readable bulletins, effect-size descriptors, summary narratives, and context-aware warnings/emojis.
   - `visualizations.py`: Plotly figures for distributions (hist/box overlays, violin), scatter/trend/regression diagnostics, grouped box plots, bar charts, time series, correlation heatmaps, etc., emphasizing reusable, publication-quality defaults.
3. **Database Layer (`db/`):**
   - `connect_to_sql.py`: company-supplied `MySqlConnection` (SQL Server via pyodbc/SQLAlchemy) with helpers for reading/writing tables, truncation, and templated query execution.
   - `connection.py`: thin `DatabaseClient` that instantiates the connector from `DatabaseConfig`, lazily opens connections, and exposes test/close utilities.
   - `introspection.py`: metadata queries against `INFORMATION_SCHEMA` for tables, columns, and row counts to power dynamic table pickers.
   - `loader.py`: safe table loads with optional column subsets and pagination-like limits.
   - `query.py`: defensive wrapper that validates user SQL (only `SELECT`/`WITH`, blocks destructive keywords) before delegating to the connector.
4. **Configuration (`config/settings.py`):**
   - `DatabaseConfig` and `AppConfig` dataclasses with env-backed constructors and helpers (e.g., `connection_string`).
   - Singleton `Config` surfaces directories, DB/app configs, and is cached in Streamlit session state for reuse.

## Streamlit UX Highlights (`app/main.py`)
- Adds project root to `sys.path` for module imports when run via Streamlit.
- Session-state keys manage dataset persistence, filter versions, navigation, caches, and visual filter contexts.
- File upload supports CSV/XLSX; sample datasets and SQL table pulls share the same downstream profile/analysis pipeline.
- Visual filter contexts include log-scaling toggles, outlier overlays (IQR or z-score), and sampling guards for large frames.
- Correlation caching keyed on dataset + filters prevents recomputation during navigation.
- Tabs orchestrate:
  - **Overview:** profiling summaries, column cards, quality warnings, dataset-level insights.
  - **Relationships:** correlation heatmaps, scatter matrices, pair plots, statistical test launchers.
  - **Trends:** time-series analysis, trend detection via `detect_trend`, regression diagnostics.
  - **Geospatial:** conditional CRS/transformation support via `pyproj` (optional dependency, gracefully degrades).
  - **Insights:** narrative bullet lists from `InsightGenerator`, status badges, recommended follow-ups.
- Database sidebar mode lets users test connections, browse tables, preview schemas, and load limited data samples before running analyses.

## Data & Insight Workflow
1. **Ingestion:** User uploads/selects data → `persist_dataset` caches frame and triggers profiling.
2. **Profiling:** `profile_dataset` builds dataset/column stats, completeness, type distributions, duplicates, correlations.
3. **Exploration:** UI widgets filter data, configure chart parameters, and call visualization builders with sampling controls.
4. **Statistical Testing:** Users choose numeric/categorical combos → corresponding helpers compute regression, t-tests (independent, Welch, paired), ANOVA variants, nonparametric tests, chi-square, Levene, etc.
5. **Insight Narratives:** Statistical outputs feed `InsightGenerator` to create context-aware interpretations (significance, effect size adjectives, ranked lists).
6. **Storytelling:** Summary narrative stitches dataset overview, data quality, and key relationships; UI surfaces badges/emojis for quick scanning.

## Configuration & Environment
- `.env` values (see README/Quickstart) control DB host/port/name/user/pass/driver plus app thresholds (`MAX_ROWS_DISPLAY`, `CORRELATION_THRESHOLD`, `P_VALUE_THRESHOLD`).
- `Config.load()` ensures directories (`data/`, `logs/`) exist conceptually and centralizes access to `DatabaseConfig` + `AppConfig`.
- Streamlit caches maintain TTL and dataset versioning, while `AppConfig` toggles profiling and caching defaults.

## Database Connectivity Notes
- Default driver expects SQL Server via ODBC Driver 17; other DBs (PostgreSQL/MySQL/SQLite) supported through SQLAlchemy DSNs when configured.
- `run_safe_query()` enforces read-only behavior; destructive keywords raise `ValueError` before reaching the database.
- Bulk upload/write helpers exist but aren’t invoked by the Streamlit app (left for back-office workflows).

## Testing Strategy (`tests/`)
- `test_profiling.py`: validates column-type detection, profiling stats, null handling, outlier detection, dataset completeness, and data-quality issue detection.
- `test_statistics.py`: covers correlation flavors, regression diagnostics, t-tests, ANOVA, trend detection, and z-score outlier detection, including failure cases for insufficient data.
- Fixtures (`tests/conftest.py`, not shown) likely supply sample DataFrames for deterministic assertions.
- Run commands:
  - `poetry run pytest tests/ -v`
  - `poetry run pytest tests/ --cov=core --cov=db --cov-report=html`

## Usage & Operations
- **Install:** `poetry install`
- **Run App:** `poetry run streamlit run app/main.py` (or `python -m poetry run streamlit run app/main.py` per QUICKSTART) → launches at http://localhost:8501.
- **Sample Flow:** Load sample data (Sales/Customer) via sidebar → navigate tabs → inspect insights and visualizations.
- **Custom Analysis Hooks:**
  - Add domain-specific stats in `core/statistics.py` and surface them in `app/main.py`.
  - Extend narratives via new generators in `core/insights.py`.
  - Introduce bespoke charts within `core/visualizations.py` and register them in the UI layer.

## Roadmap (from README)
- **Phase 1 (Current):** Profiling engine, statistical methods, Streamlit UI, SQL connectivity.
- **Phase 2 (Next):** Schema intelligence, anomaly detection, natural-language querying, key-driver analysis.
- **Phase 3 (Future):** ML forecasting, automated insight recommendations, intelligent joins, collaborative annotations.

## Enterprise Considerations
- Connection pooling & reuse through SQLAlchemy engine.
- Emphasis on graceful error handling (UI warnings, validation functions).
- Data-quality safeguards (null/outlier detection, mixed-type alerts, completeness scoring).
- Scalability tactics: row sampling for visuals, cached computations, log-scale filtering, memory usage tracking.

## Quick Reference Commands
- **Start UI:** `poetry run streamlit run app/main.py`
- **Run Tests:** `poetry run pytest tests/ -v`
- **Coverage:** `poetry run pytest tests/ --cov=core --cov=db`
- **Clear Streamlit Cache:** `poetry run streamlit cache clear`
- **Install Poetry (if missing on Windows PowerShell):** `(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -`

## Key Files
- Streamlit UI: `app/main.py`
- Configs: `config/settings.py`
- Profiling: `core/profiling.py`
- Statistics: `core/statistics.py`
- Insight Narratives: `core/insights.py`
- Visualizations: `core/visualizations.py`
- DB Connector Wrapper: `db/connect_to_sql.py`
- Safe Query Execution: `db/query.py`
- Tests: `tests/test_profiling.py`, `tests/test_statistics.py`

---
**Philosophy:** “Clarity over cleverness, stability over shortcuts, extensibility over quick hacks.” The codebase favors modularity, strong typing, and business-readable outputs to feel like “a senior data scientist continuously analyzing every dataset.”
