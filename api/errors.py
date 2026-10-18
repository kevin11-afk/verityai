from __future__ import annotations


class ApiError(Exception):
    """An expected, user-presentable failure: becomes {"error": {"code", "message"}}."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
