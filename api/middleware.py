"""
Session cookie middleware.

Sets / reads a `session_id` HTTP-only cookie on every request so that
each browser tab gets its own analytics workspace.
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from api.session_store import create_session, get_session


class SessionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        session_id = request.cookies.get("session_id")

        # Create a new session when the cookie is missing or expired
        if session_id is None or get_session(session_id) is None:
            session_id = create_session()

        # Attach to request state so dependency injection can use it
        request.state.session_id = session_id

        response: Response = await call_next(request)

        # Set/refresh the cookie
        response.set_cookie(
            key="session_id",
            value=session_id,
            httponly=True,
            samesite="lax",
            max_age=3600,
        )
        return response
