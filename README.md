# Statistical Engine

A data analysis platform built with Python and Streamlit that helps you explore datasets, run statistical tests, and generate readable summaries — all from a browser-based interface. It connects to SQL Server databases or accepts file uploads (CSV/Excel), profiles the data automatically, and presents findings through interactive Plotly charts.

## What It Does

The engine handles the repetitive parts of exploratory data analysis so you can focus on interpretation. Load a dataset and you immediately get:

- A profile of every column — data types, completeness, distributions, outliers (IQR method), and quality flags like constant columns or high-cardinality text.
- A correlation matrix across all numeric columns, with Pearson, Spearman, or Kendall options.
- Linear regression, independent-samples t-tests, and one-way ANOVA, each returning structured result objects with p-values, effect sizes, and fit metrics.
- Trend detection on sequential numeric data using ordinary least-squares regression with significance testing on the slope.
- Plain-English narrative summaries generated from the statistical output — the `InsightGenerator` class translates things like correlation strength, model R², and group mean differences into sentences a non-technical reader can follow.

None of this relies on machine learning or AI. The insights are produced by rule-based logic that maps statistical results to human-readable text using thresholds (e.g. |r| < 0.3 = "weak", Cohen's d < 0.2 = "negligible").

## Project Structure

```
statistical-engine/
├── app/
│   └── main.py              # Streamlit UI — sidebar, tabs, session state
├── core/
│   ├── profiling.py          # Column and dataset profiling, outlier detection, quality checks
│   ├── statistics.py         # Correlation, regression, t-test, ANOVA, trend detection
│   ├── insights.py           # Translates statistical results into narrative text
│   └── visualizations.py     # Plotly charts — histograms, heatmaps, scatter, time series, 3D
├── db/
│   ├── connection.py         # DatabaseClient wrapper around the SQL connection module
│   ├── connect_to_sql.py     # pyodbc + SQLAlchemy connector for SQL Server (Trusted Connection)
│   ├── query.py              # Read-only query validation and safe execution
│   ├── loader.py             # Table loading with optional column selection and row limits
│   └── introspection.py      # Schema introspection — list tables, describe columns, row counts
├── config/
│   └── settings.py           # Dataclass-based config loaded from environment variables
├── models/                   # Placeholder for future forecasting/clustering work
├── tests/
│   ├── conftest.py           # Shared pytest fixtures (numeric, mixed, null, categorical DataFrames)
│   ├── test_statistics.py    # Tests for correlation, regression, t-test, ANOVA, trend detection
│   └── test_profiling.py     # Tests for type detection, column profiling, outlier detection, quality checks
├── pyproject.toml            # Poetry project definition and dependencies
├── .env.example              # Template for environment variables
└── QUICKSTART.md             # Condensed setup and usage guide
```

## Getting Started

You need Python 3.11 or later and Poetry 2.0+ for dependency management.

```bash
# Clone and enter the project
git clone <repository-url>
cd statistical-engine

# Install everything
poetry install

# Launch the Streamlit app
poetry run streamlit run app/main.py
```

The app opens at `http://localhost:8501`. From the sidebar you can upload a CSV or Excel file, connect to a SQL Server database, or load one of the built-in sample datasets (Sales, Customer, or Random) to try things out immediately.

## Connecting to a Database

The database layer is built around SQL Server using pyodbc with Windows Trusted Connection. Copy `.env.example` to `.env` and fill in your server details:

```env
DB_HOST=your-server
DB_PORT=1433
DB_NAME=your-database
DB_DRIVER=mssql+pyodbc
```

Once connected through the sidebar, you can browse available tables or run custom read-only SELECT queries. The query layer validates every statement before execution — it rejects anything containing DELETE, DROP, UPDATE, INSERT, or other write operations.

## Configuration

All settings live in `config/settings.py` and are loaded from environment variables with sensible defaults. The main ones worth adjusting:

- `CORRELATION_THRESHOLD` (default 0.3) — minimum |r| to flag a correlation as noteworthy.
- `P_VALUE_THRESHOLD` (default 0.05) — significance cutoff used across all statistical tests.
- `OUTLIER_IQR_MULTIPLIER` (default 1.5) — how aggressively outliers are flagged.
- `MAX_ROWS_DISPLAY` (default 10,000) — caps how many rows the UI renders at once.

## Running Tests

```bash
# Full suite
poetry run pytest tests/ -v

# With coverage
poetry run pytest tests/ --cov=core --cov=db --cov-report=html
```

Tests cover correlation analysis, linear regression, t-tests, ANOVA, trend detection, column type inference, outlier detection, dataset profiling, and data quality checks.

## Extending the Engine

The codebase is modular by design. Each layer has a clear responsibility:

- To add a new statistical method, write a function in `core/statistics.py` that returns a dataclass result, then add a corresponding `generate_*_insights` method to `InsightGenerator` in `core/insights.py`.
- To add a new chart type, create a function in `core/visualizations.py` that returns a `plotly.graph_objects.Figure`.
- To surface a new analysis in the UI, wire it into the appropriate tab in `app/main.py`.

The `models/` directory is reserved for future work like time series forecasting, anomaly detection, and clustering — none of which is implemented yet.

## Dependencies

Python, Streamlit, Pandas, NumPy, SciPy, scikit-learn, Plotly, SQLAlchemy, pyodbc, and Poetry for dependency management. The full list with version constraints is in `pyproject.toml`.

## License

Internal use only. For feature requests or bugs, contact the Data Engineering team.
