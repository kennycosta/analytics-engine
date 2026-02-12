"""Utility helpers for applying cascading filters to pandas DataFrames."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd


FilterDict = Dict[str, Any]


def apply_filters(df: pd.DataFrame, filters: Sequence[FilterDict]) -> pd.DataFrame:
    """Apply a sequence of filter definitions to the dataframe."""
    if df is None or len(filters) == 0:
        return df

    filtered = df.copy()
    for raw_filter in filters:
        column = raw_filter.get("column")
        operator = (raw_filter.get("operator") or "").lower()
        ftype = (raw_filter.get("type") or "numeric").lower()

        if column not in filtered.columns:
            raise ValueError(f"Column '{column}' not found for filtering")

        series = filtered[column]
        mask = None

        if ftype in {"numeric", "number"}:
            mask = _apply_numeric_filter(series, operator, raw_filter)
        elif ftype in {"datetime", "date"}:
            mask = _apply_datetime_filter(series, operator, raw_filter)
        elif ftype in {"categorical", "text", "string", "boolean"}:
            mask = _apply_categorical_filter(series, operator, raw_filter)
        else:
            raise ValueError(f"Unsupported filter type '{ftype}' for column '{column}'")

        if mask is None:
            raise ValueError(f"Operator '{operator}' is not supported for type '{ftype}'")

        filtered = filtered.loc[mask]

    return filtered


def _apply_numeric_filter(series: pd.Series, operator: str, raw_filter: FilterDict) -> pd.Series:
    numeric_series = pd.to_numeric(series, errors="coerce")

    if operator == "between":
        low, high = _range_bounds(raw_filter)
        mask = pd.Series(True, index=numeric_series.index)
        if low is not None:
            mask &= numeric_series >= float(low)
        if high is not None:
            mask &= numeric_series <= float(high)
        return mask

    value = raw_filter.get("value")
    if value is None:
        raise ValueError("Numeric filters require a value or range")

    value = float(value)
    if operator == ">":
        return numeric_series > value
    if operator == ">=":
        return numeric_series >= value
    if operator == "<":
        return numeric_series < value
    if operator == "<=":
        return numeric_series <= value
    if operator in {"==", "equals"}:
        return numeric_series == value
    if operator in {"!=", "not_equals"}:
        return numeric_series != value

    raise ValueError(f"Unsupported numeric operator '{operator}'")


def _apply_datetime_filter(series: pd.Series, operator: str, raw_filter: FilterDict) -> pd.Series:
    dt_series = pd.to_datetime(series, errors="coerce")

    if operator == "between":
        low, high = _range_bounds(raw_filter)
        mask = pd.Series(True, index=dt_series.index)
        if low is not None:
            mask &= dt_series >= pd.to_datetime(low)
        if high is not None:
            mask &= dt_series <= pd.to_datetime(high)
        return mask

    value = raw_filter.get("value")
    if value is None:
        raise ValueError("Datetime filters require a value or range")

    value = pd.to_datetime(value)
    if operator in {"before", "<"}:
        return dt_series < value
    if operator in {"after", ">"}:
        return dt_series > value
    if operator in {"on", "==", "equals"}:
        return dt_series == value

    raise ValueError(f"Unsupported datetime operator '{operator}'")


def _apply_categorical_filter(series: pd.Series, operator: str, raw_filter: FilterDict) -> pd.Series:
    str_series = series.astype(str)
    values: List[Any] | None = raw_filter.get("values")
    value = raw_filter.get("value")

    if operator in {"in", "not_in"}:
        if not values:
            raise ValueError(f"Operator '{operator}' requires a list of values")
        mask = str_series.isin([str(v) for v in values])
        return mask if operator == "in" else ~mask

    if value is None:
        raise ValueError(f"Operator '{operator}' requires a value")

    value_str = str(value)
    if operator in {"==", "equals"}:
        return str_series == value_str
    if operator in {"!=", "not_equals"}:
        return str_series != value_str
    if operator == "contains":
        return str_series.str.contains(value_str, case=False, na=False)
    if operator == "starts_with":
        return str_series.str.startswith(value_str, na=False)
    if operator == "ends_with":
        return str_series.str.endswith(value_str, na=False)

    raise ValueError(f"Unsupported categorical operator '{operator}'")


def _range_bounds(raw_filter: FilterDict) -> Tuple[Any, Any]:
    range_values = raw_filter.get("range")
    if isinstance(range_values, (list, tuple)) and len(range_values) == 2:
        return range_values[0], range_values[1]
    return None, None
