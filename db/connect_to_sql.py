"""
Utility for connecting to SQL Server using pyodbc and SQLAlchemy.

Provided by the user; moved into the db/ module for use in the Streamlit UI.
"""

from pathlib import Path
from string import Template
from typing import Union
import warnings

import pandas as pd
import pyodbc
import sqlalchemy

warnings.simplefilter(action="ignore", category=UserWarning)


class MySqlConnection:
    """Convenience wrapper around pyodbc/SQLAlchemy for SQL Server."""

    def __init__(self, server_name: str, database_name: str):
        self.server_name = server_name
        self.database_name = database_name
        self.params = (
            "DRIVER={ODBC Driver 17 for SQL Server}; "
            "SERVER=%s; DATABASE=%s; Trusted_Connection=Yes" % (self.server_name, self.database_name)
        )
        self.connection = pyodbc.connect(self.params)
        self.connection_cursor = self.connection.cursor()
        self.connection_engine = sqlalchemy.create_engine(
            "mssql+pyodbc://%s/%s?driver=ODBC+Driver+17+for+SQL+Server" % (self.server_name, self.database_name),
            fast_executemany=True,
        )

    def get_connection_information(self) -> None:
        print(f"Server: {self.server_name}")
        print(f"Database: {self.database_name}")

    def sql_to_df(self, query: str) -> pd.DataFrame:
        """Read a query into a DataFrame."""
        return pd.read_sql_query(query, self.connection)

    def df_to_sql(self, df: pd.DataFrame, table_name: str, schema: str) -> None:
        """Append a DataFrame to a table."""
        df.to_sql(
            name=table_name,
            con=self.connection_engine,
            if_exists="append",
            index=False,
            schema=schema,
            chunksize=20000,
        )

    def truncate_table(self, table_name: str, schema: str, reseed: bool = False) -> None:
        """Truncate a table and optionally reseed identity."""
        query = f"TRUNCATE TABLE {schema}.{table_name}"
        self.connection_cursor.execute(query)
        self.connection.commit()

        if reseed:
            query = f"DBCC CHECKIDENT ('{schema}.{table_name}', RESEED, 1);"
            self.connection_cursor.execute(query)
            self.connection.commit()

    def delete_with_conditions(self, table_name: str, schema: str, conditions: str) -> None:
        """Delete rows from a table with a custom WHERE clause."""
        assert isinstance(conditions, str)
        query_delete = f"""DELETE FROM {schema}.{table_name} WHERE {conditions}"""
        with self.connection_engine.begin() as conn:
            conn.execute(sqlalchemy.text(query_delete))

    def run_query(self, query: str, fetch_rows: Union[str, None] = None):
        """Run a custom query and optionally fetch results."""
        assert isinstance(query, str)
        with self.connection_engine.begin() as conn:
            if fetch_rows is None:
                conn.execute(sqlalchemy.text(query))
                results = None
            elif fetch_rows == "one":
                results = conn.execute(sqlalchemy.text(query)).fetchone()
                cols = ["ReturnCode", "ReturnMessage"]
                results = pd.DataFrame([tuple(results)])
                results.columns = cols
            elif fetch_rows == "all":
                results = conn.execute(sqlalchemy.text(query)).fetchall()
            else:
                raise ValueError("Unsupported fetch_rows mode")
            return results

    def close_connection(self) -> None:
        """Close the SQL connection."""
        self.connection.close()

    def df_to_new_table(self, df: pd.DataFrame, table_name: str, schema: str, **kwargs) -> None:
        """Create a new table and upload a DataFrame."""
        len_varchar = kwargs.get("varchar", 255)

        print(
            f"Creating new table {self.server_name}.{self.database_name}.{schema}.{table_name} "
            f"and uploading {len(df)} rows"
        )

        dtypedict = {}
        for col_name, dtype in zip(df.columns, df.dtypes):
            dtype_str = str(dtype)
            if "object" in dtype_str:
                dtypedict[col_name] = sqlalchemy.types.NVARCHAR(length=len_varchar)
            if "datetime" in dtype_str:
                dtypedict[col_name] = sqlalchemy.types.DateTime()
            if "float" in dtype_str:
                dtypedict[col_name] = sqlalchemy.types.Float(precision=3, asdecimal=True)
            if "int" in dtype_str:
                dtypedict[col_name] = sqlalchemy.types.INT()

        df.to_sql(
            name=table_name,
            con=self.connection_engine,
            if_exists="replace",
            index=False,
            schema=schema,
            chunksize=100000,
            dtype=dtypedict,
        )

        print("All done! :D")

    def commit_custom_query(self, query: str) -> None:
        self.connection_cursor.execute(query)
        self.connection.commit()


def read_query(file: Union[Path, str], **kwds: Union[str, int]) -> str:
    """Read a SQL file and substitute template variables."""
    with open(file, mode="r", encoding="utf-8") as f:
        query: str = f.read()

    query = Template(query).substitute(**kwds)
    return query


def upload_to_sql(df2upload: pd.DataFrame, schema: str, table_name: str, sql_connection: MySqlConnection, flush_table: bool = True) -> None:
    """Upload a DataFrame to a SQL table, optionally truncating first."""
    assert isinstance(df2upload, pd.DataFrame), "Must be a pandas dataframe"
    assert isinstance(flush_table, bool), "Must be a boolean"
    assert isinstance(schema, str), "Must be a string"
    assert isinstance(table_name, str), "Must be a string"
    print(f"Flushing and uploading {len(df2upload)} row(s) to {table_name} ...")

    if flush_table:
        sql_connection.truncate_table(table_name, schema, reseed=False)

    sql_connection.df_to_sql(df2upload, table_name, schema)
    print("Upload successful! Connection remains open.")
