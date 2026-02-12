"""
In-memory session store keyed by UUID.

Each session holds config, database client, loaded data, and profile.
Sessions expire after 1 hour of inactivity.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from config.settings import Config, DatabaseConfig
from db.connection import DatabaseClient
from core.profiling import DatasetProfile


@dataclass
class SessionData:
    """Per-session state mirroring the old Streamlit session_state."""

    config: Config = field(default_factory=Config.load)
    db_client: Optional[DatabaseClient] = None
    available_tables: List[str] = field(default_factory=list)
    raw_data: Optional[pd.DataFrame] = None
    current_data: Optional[pd.DataFrame] = None
    dataset_profile: Optional[DatasetProfile] = None
    dataset_name: Optional[str] = None
    active_filters: List[Dict[str, Any]] = field(default_factory=list)
    last_accessed: float = field(default_factory=time.time)


# Global store: session_id -> SessionData
_sessions: Dict[str, SessionData] = {}

SESSION_TTL_SECONDS = 3600  # 1 hour


def create_session() -> str:
    """Create a new session and return its ID."""
    session_id = str(uuid.uuid4())
    _sessions[session_id] = SessionData()
    return session_id


def get_session(session_id: str) -> Optional[SessionData]:
    """Return the session if it exists and is not expired."""
    session = _sessions.get(session_id)
    if session is None:
        return None
    if time.time() - session.last_accessed > SESSION_TTL_SECONDS:
        del _sessions[session_id]
        return None
    session.last_accessed = time.time()
    return session


def cleanup_expired() -> int:
    """Remove expired sessions. Returns number removed."""
    now = time.time()
    expired = [
        sid for sid, s in _sessions.items()
        if now - s.last_accessed > SESSION_TTL_SECONDS
    ]
    for sid in expired:
        del _sessions[sid]
    return len(expired)
