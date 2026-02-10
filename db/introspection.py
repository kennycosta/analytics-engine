from typing import List, Dict, Any
import pandas as pd

from db.connection import DatabaseClient


def get_tables(db: DatabaseClient) -> List[str]:
    """
    Return all accessible tables in the database.
    """
    query = """
    SELECT TABLE_SCHEMA + '.' + TABLE_NAME AS table_name
    FROM INFORMATION_SCHEMA.TABLES
    WHERE TABLE_TYPE = 'BASE TABLE'
    ORDER BY TABLE_SCHEMA, TABLE_NAME
    """
    df = db.connection.sql_to_df(query)
    return df["table_name"].tolist()


def get_table_columns(db: DatabaseClient, table_name: str) -> List[Dict[str, Any]]:
    """
    Return column metadata for a given table.
    """
    schema, table = table_name.split(".")

    query = f"""
    SELECT
        COLUMN_NAME,
        DATA_TYPE,
        IS_NULLABLE
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = '{schema}'
      AND TABLE_NAME = '{table}'
    ORDER BY ORDINAL_POSITION
    """

    df = db.connection.sql_to_df(query)

    return [
        {
            "name": row["COLUMN_NAME"],
            "type": row["DATA_TYPE"],
            "nullable": row["IS_NULLABLE"] == "YES",
        }
        for _, row in df.iterrows()
    ]


def get_row_count(db: DatabaseClient, table_name: str) -> int:
    """Return total row count for a table."""
    query = f"SELECT COUNT(*) AS row_count FROM {table_name}"
    df = db.connection.sql_to_df(query)
    return int(df["row_count"].iloc[0])
