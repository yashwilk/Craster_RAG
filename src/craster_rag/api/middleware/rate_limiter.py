"""
# 10 requests per minute per IP
# returns friendly error if exceeded
{"error": "Too many requests. Please wait a moment."}
"""


"""
Limits:
    chat endpoint    10 requests per minute per IP
    admin endpoints  5 requests per minute per IP
"""

from fastapi import Request, Response

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded


def rate_limit_exceeded_handler(request: Request,
                                exc: RateLimitExceeded) -> Response:

    return Response(
        content     = '{"error": "Too many requests. Please wait a moment before trying again."}',
        status_code = 429,
        media_type  = "application/json",
    )

# uses client IP address as the key
# one limit counter per IP address
limiter = Limiter(key_func=get_remote_address)
