# AI Analytics Engine

**A production-grade, domain-agnostic analytics platform that transforms raw business data into deep statistical insights through automation, interactive exploration, and intelligent reasoning.**

## 🎯 Overview

This is not a dashboard. This is an **intelligence layer** that sits on top of your data infrastructure.

The AI Analytics Engine eliminates repetitive manual analysis, democratizes advanced statistics across teams, and provides rapid insight generation without requiring SQL or Python expertise.

### Core Capabilities

- **Automated Exploratory Data Analysis** - Instant profiling, type detection, quality checks
- **Statistical Analysis Engine** - Correlations, regression, hypothesis testing, trend detection
- **Plain-English Insights** - Technical results translated into business narratives
- **Interactive Visualizations** - Publication-quality charts with Plotly
- **SQL Connectivity** - Direct connection to PostgreSQL, MySQL, SQLite databases
- **Modular Architecture** - Clean separation of concerns, extensible design

## 🏗️ Architecture

```
ai-analytics-engine/
├── app/                    # Streamlit UI layer
│   └── main.py            # Application entry point
├── core/                   # Analytics engines (business logic)
│   ├── profiling.py       # Automated EDA and data profiling
│   ├── statistics.py      # Statistical analysis methods
│   ├── insights.py        # Plain-English insight generation
│   └── visualizations.py  # Interactive Plotly charts
├── db/                     # Database connectivity layer
│   └── connection.py      # SQLAlchemy abstraction
├── config/                 # Configuration management
│   └── settings.py        # Environment-based config
├── models/                 # Future ML models (placeholder)
├── tests/                  # Comprehensive test suite
│   ├── conftest.py        # Test fixtures
│   ├── test_profiling.py  # Profiling tests
│   └── test_statistics.py # Statistical tests
└── pyproject.toml         # Poetry dependency management
```

### Design Principles

✅ **Domain-agnostic** - Works with any structured dataset  
✅ **Separation of concerns** - UI ≠ Analytics ≠ Data Access  
✅ **Testable** - 90+ unit tests with high coverage  
✅ **Extensible** - Add new analyses without rewrites  
✅ **Production-ready** - Enterprise-grade error handling, logging  

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Poetry 2.0+ (for dependency management)

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd ai-analytics-engine
```

2. **Install dependencies with Poetry:**
```bash
poetry install
```

3. **Run the application:**
```bash
poetry run streamlit run app/main.py
```

The application will open in your browser at `http://localhost:8501`.

## 📊 Usage

### Loading Data

The platform supports multiple data sources:

1. **CSV/Excel Upload** - Drag and drop files directly
2. **SQL Database** - Connect to PostgreSQL, MySQL, SQLite
3. **Sample Data** - Pre-loaded datasets for exploration

### Analysis Workflow

1. **Overview Tab** - Dataset summary, column types, quality metrics
2. **Distributions Tab** - Histogram, box plots, statistical summaries
3. **Relationships Tab** - Correlation matrices, scatter plots, pairwise analysis
4. **Trends Tab** - Time series analysis, trend detection
5. **Insights Tab** - AI-generated narratives and recommendations

### Example: Analyzing Sales Data

```python
# Load sample sales data
# Navigate to: Sidebar → Sample Data → Sales Data

# Automated insights will include:
# - Revenue distribution and outliers
# - Correlation between units sold and revenue
# - Regional performance comparisons
# - Trend detection in time series
# - Plain-English summary narrative
```

## 🔧 Configuration

Create a `.env` file in the project root:

```env
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=analytics
DB_USER=postgres
DB_PASSWORD=your_password
DB_DRIVER=postgresql

# Application Settings
APP_TITLE=AI Analytics Engine
MAX_ROWS_DISPLAY=10000
CORRELATION_THRESHOLD=0.3
P_VALUE_THRESHOLD=0.05
```

## 🧪 Testing

Run the complete test suite:

```bash
poetry run pytest tests/ -v
```

Run with coverage:

```bash
poetry run pytest tests/ --cov=core --cov=db --cov-report=html
```

## 📈 Roadmap

### Phase 1 (Current)
- ✅ Core profiling engine
- ✅ Statistical analysis methods
- ✅ Interactive Streamlit UI
- ✅ Basic SQL connectivity

### Phase 2 (Next)
- 🔄 Automatic schema intelligence
- 🔄 Proactive anomaly detection
- 🔄 Natural language queries
- 🔄 Key driver analysis

### Phase 3 (Future)
- 📋 ML-based forecasting
- 📋 Automated insight recommendations
- 📋 Multi-table join suggestions
- 📋 Collaborative annotations

## 🏢 Enterprise Features

- **Connection Pooling** - Efficient database resource management
- **Error Handling** - Graceful failures with user-friendly messages
- **Data Quality Checks** - Automated detection of missing data, outliers, type issues
- **Scalability** - Designed to handle large datasets (sampling strategies)
- **Extensibility** - Plugin architecture for custom analyses

## 🤝 Contributing

This is an internal analytics product. For feature requests or bugs, contact the Data Engineering team.

## 📄 License

Internal use only. All rights reserved.

## 🧠 Philosophy

> "This should feel like having a senior data scientist continuously analyzing every dataset."

We prioritize:
- **Clarity over cleverness**
- **Stability over shortcuts**
- **Extensibility over quick hacks**

Every design decision favors long-term maintainability and user trust.

---

**Built with:** Python, Streamlit, Pandas, NumPy, SciPy, scikit-learn, Plotly, SQLAlchemy, Poetry
