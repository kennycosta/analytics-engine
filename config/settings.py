"""
Centralized configuration management for the analytics platform.

Handles environment variables, database config, and application settings
with type safety and validation.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    
    host: str
    port: int
    database: str
    username: str
    password: str
    driver: str = "postgresql"  # postgresql, mysql, sqlite
    pool_size: int = 5
    max_overflow: int = 10
    
    @property
    def connection_string(self) -> str:
        """Generate SQLAlchemy connection string."""
        if self.driver == "sqlite":
            return f"sqlite:///{self.database}"
        
        return (
            f"{self.driver}://{self.username}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )
    
    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Load database config from environment variables."""
        return cls(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "analytics"),
            username=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
            driver=os.getenv("DB_DRIVER", "postgresql"),
            pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
            max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "10"))
        )


@dataclass
class AppConfig:
    """Application-level configuration."""
    
    title: str = "AI Analytics Engine"
    page_icon: str = "📊"
    layout: str = "wide"
    
    # Data handling
    max_rows_display: int = 10000
    sample_size: int = 1000
    
    # Statistical thresholds
    correlation_threshold: float = 0.3
    p_value_threshold: float = 0.05
    outlier_iqr_multiplier: float = 1.5
    
    # Performance
    cache_ttl: int = 3600  # seconds
    enable_profiling: bool = False
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load app config from environment variables."""
        return cls(
            title=os.getenv("APP_TITLE", "AI Analytics Engine"),
            max_rows_display=int(os.getenv("MAX_ROWS_DISPLAY", "10000")),
            sample_size=int(os.getenv("SAMPLE_SIZE", "1000")),
            correlation_threshold=float(os.getenv("CORRELATION_THRESHOLD", "0.3")),
            p_value_threshold=float(os.getenv("P_VALUE_THRESHOLD", "0.05")),
        )


class Config:
    """Global configuration manager."""
    
    _instance: Optional["Config"] = None
    
    def __init__(self):
        self.db = DatabaseConfig.from_env()
        self.app = AppConfig.from_env()
        self.root_dir = Path(__file__).parent.parent
        self.data_dir = self.root_dir / "data"
        self.logs_dir = self.root_dir / "logs"
    
    @classmethod
    def load(cls) -> "Config":
        """Singleton pattern - load or return existing config."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
