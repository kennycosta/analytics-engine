from typing import List, Optional
import pandas as pd

from db.connection import DatabaseClient


def load_table(
    db: DatabaseClient,
    table_name: str,
    columns: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load table data into a pandas DataFrame.

    Args:
        table_name: schema.table
        columns: list of columns to select
        limit: optional row limit

    Returns:
        DataFrame
    """
    col_str = ", ".join(columns) if columns else "*"
    query = f"SELECT {col_str} FROM {table_name}"

    if limit:
        query += f" ORDER BY (SELECT NULL) OFFSET 0 ROWS FETCH NEXT {limit} ROWS ONLY"

    return db.connection.sql_to_df(query)


def run_query(db: DatabaseClient, query: str) -> pd.DataFrame:
    """
    Execute an arbitrary SELECT query and return results as DataFrame.
    """
    return db.connection.sql_to_df(query)
