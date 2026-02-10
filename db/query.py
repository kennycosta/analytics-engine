"""
Query validation and safe execution layer.

Ensures only read-only SQL queries are executed against the database.
Acts as the single guarded entry point for user-provided SQL.
"""

from typing import Set
import pandas as pd

from db.connection import DatabaseClient


FORBIDDEN_KEYWORDS: Set[str] = {
    "delete",
    "drop",
    "truncate",
    "update",
    "alter",
    "insert",
    "merge",
    "exec",
    "execute",
    "create",
    "grant",
    "revoke",
}


def validate_read_only_query(query: str) -> None:
    """
    Validate that a SQL query is read-only.

    Allowed:
    - SELECT
    - WITH ... SELECT (CTEs)

    Raises:
        ValueError if the query is unsafe.
    """
    if not isinstance(query, str):
        raise ValueError("Query must be a string.")

    normalized = query.strip().lower()

    if not normalized:
        raise ValueError("Query is empty.")

    if not (normalized.startswith("select") or normalized.startswith("with")):
        raise ValueError("Only SELECT queries are allowed.")

    for keyword in FORBIDDEN_KEYWORDS:
        if keyword in normalized:
            raise ValueError(f"Forbidden SQL keyword detected: '{keyword}'")


def run_safe_query(db: DatabaseClient, query: str) -> pd.DataFrame:
    """
    Validate and execute a read-only SQL query.

    Args:
        db: DatabaseClient instance
        query: SQL SELECT query

    Returns:
        Query results as a pandas DataFrame
    """
    validate_read_only_query(query)
    return db.connection.sql_to_df(query)
