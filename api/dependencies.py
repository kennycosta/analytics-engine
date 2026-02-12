"""
FastAPI dependency-injection helpers.
"""

from fastapi import Request, HTTPException

from api.session_store import SessionData, get_session


def get_session_data(request: Request) -> SessionData:
    """Return the current session, creating one if needed."""
    session_id: str = request.state.session_id
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=401, detail="Session expired")
    return session


def require_data(request: Request) -> SessionData:
    """Like get_session_data but also asserts a dataset is loaded."""
    session = get_session_data(request)
    if session.current_data is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")
    return session
