"""
Statistics endpoints: correlation, regression, t-test, ANOVA, trends.
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException
import numpy as np

from api.dependencies import require_data
from api.session_store import SessionData
from api.models.requests import (
    TrendRequest, CorrelationRequest, RegressionRequest,
    TTestRequest, AnovaRequest,
)
from api.models.responses import (
    CorrelationMatrixResponse, TrendResponse, InsightsResponse,
    CorrelationResponse, RegressionResponse, TTestResponse, AnovaResponse,
)
from core.statistics import (
    correlation_matrix, calculate_correlation, detect_trend,
    linear_regression_analysis, independent_t_test, anova_test,
)
from core.insights import InsightGenerator

router = APIRouter(prefix="/api/stats", tags=["statistics"])


@router.get("/correlation-matrix", response_model=CorrelationMatrixResponse)
def get_correlation_matrix(session: SessionData = Depends(require_data)):
    """Correlation matrix for all numeric columns."""
    df = session.current_data
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 numeric columns")
    corr = correlation_matrix(df, num_cols)
    return CorrelationMatrixResponse(
        columns=num_cols,
        matrix=corr.values.tolist(),
    )


@router.get("/insights", response_model=InsightsResponse)
def get_insights(session: SessionData = Depends(require_data)):
    """Generate full narrative insights."""
    df = session.current_data
    profile = session.dataset_profile
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    corrs = []
    for i, c1 in enumerate(num_cols):
        for c2 in num_cols[i + 1:]:
            try:
                corrs.append(calculate_correlation(df, c1, c2))
            except Exception:
                pass

    narrative = InsightGenerator.generate_summary_narrative(profile, corrs)
    correlation_dicts = [
        {
            "variable1": c.variable1,
            "variable2": c.variable2,
            "correlation": float(c.correlation),
            "p_value": float(c.p_value),
            "significant": bool(c.significant),
            "strength": c.strength,
        }
        for c in corrs
    ]
    return InsightsResponse(narrative=narrative, correlations=correlation_dicts)


@router.post("/trend", response_model=TrendResponse)
def analyze_trend(body: TrendRequest, session: SessionData = Depends(require_data)):
    """Detect linear trend in a numeric column."""
    df = session.current_data
    if body.column not in df.columns:
        raise HTTPException(status_code=404, detail=f"Column '{body.column}' not found")
    result = detect_trend(df[body.column])
    return TrendResponse(
        column=body.column,
        trend=result["trend"],
        slope=float(result["slope"]) if result.get("slope") is not None else None,
        r_squared=float(result["r_squared"]) if result.get("r_squared") is not None else None,
        p_value=float(result["p_value"]) if result.get("p_value") is not None else None,
        significant=bool(result["significant"]) if result.get("significant") is not None else None,
    )


@router.post("/correlation", response_model=CorrelationResponse)
def analyze_correlation(body: CorrelationRequest, session: SessionData = Depends(require_data)):
    """Calculate correlation between two columns."""
    df = session.current_data
    try:
        result = calculate_correlation(df, body.col1, body.col2, body.method)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return CorrelationResponse(
        variable1=result.variable1,
        variable2=result.variable2,
        correlation=float(result.correlation),
        p_value=float(result.p_value),
        method=result.method,
        significant=bool(result.significant),
        strength=result.strength,
    )


@router.post("/regression", response_model=RegressionResponse)
def analyze_regression(body: RegressionRequest, session: SessionData = Depends(require_data)):
    """Perform linear regression analysis."""
    df = session.current_data
    try:
        result = linear_regression_analysis(df, body.dependent_var, body.independent_vars)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    insights = InsightGenerator.generate_regression_insights(result)
    return RegressionResponse(
        dependent_var=result.dependent_var,
        independent_vars=result.independent_vars,
        coefficients={k: float(v) for k, v in result.coefficients.items()},
        intercept=float(result.intercept),
        r_squared=float(result.r_squared),
        adj_r_squared=float(result.adj_r_squared),
        mse=float(result.mse),
        rmse=float(result.rmse),
        p_values={k: float(v) for k, v in result.p_values.items()},
        insights=insights,
    )


@router.post("/ttest", response_model=TTestResponse)
def analyze_ttest(body: TTestRequest, session: SessionData = Depends(require_data)):
    """Perform independent samples t-test."""
    df = session.current_data
    try:
        result = independent_t_test(df, body.numeric_col, body.group_col, body.group1_value, body.group2_value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    insights = InsightGenerator.generate_ttest_insights(result)
    return TTestResponse(
        group1_name=result.group1_name,
        group2_name=result.group2_name,
        group1_mean=float(result.group1_mean),
        group2_mean=float(result.group2_mean),
        t_statistic=float(result.t_statistic),
        p_value=float(result.p_value),
        significant=bool(result.significant),
        effect_size=float(result.effect_size),
        insights=insights,
    )


@router.post("/anova", response_model=AnovaResponse)
def analyze_anova(body: AnovaRequest, session: SessionData = Depends(require_data)):
    """Perform one-way ANOVA test."""
    df = session.current_data
    try:
        result = anova_test(df, body.numeric_col, body.group_col)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    insights = InsightGenerator.generate_anova_insights(result)
    return AnovaResponse(
        groups=result.groups,
        f_statistic=float(result.f_statistic),
        p_value=float(result.p_value),
        significant=bool(result.significant),
        group_means={k: float(v) for k, v in result.group_means.items()},
        insights=insights,
    )
