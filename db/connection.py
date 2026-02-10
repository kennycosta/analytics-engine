from typing import Optional
import pandas as pd

from config.settings import DatabaseConfig
from db.connect_to_sql import MySqlConnection


class DatabaseClient:
    """
    Thin wrapper around the company SQL Server connection module.

    This class abstracts pyodbc / SQLAlchemy details away from
    Streamlit and analytics layers.
    """

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._connection: Optional[MySqlConnection] = None

    @property
    def connection(self) -> MySqlConnection:
        """
        Lazy-load the SQL connection.

        Uses Trusted Connection as per company standards.
        """
        if self._connection is None:
            self._connection = MySqlConnection(
                server_name=self.config.host,
                database_name=self.config.database,
            )
        return self._connection

    def test_connection(self) -> bool:
        """Verify database connectivity."""
        try:
            self.connection.get_connection_information()
            return True
        except Exception as exc:
            print(f"Database connection failed: {exc}")
            return False

    def close(self) -> None:
        """Close database connection."""
        if self._connection:
            self._connection.close_connection()
            self._connection = None
