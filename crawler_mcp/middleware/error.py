"""
Error handling middleware for FastMCP operations.
"""

import logging
import traceback
from collections.abc import Awaitable, Callable

from fastmcp.exceptions import ToolError
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class ErrorHandlingMiddleware:
    """
    Middleware to handle errors in FastMCP operations with proper logging
    and client-friendly error messages.
    """

    def __init__(self, app: Callable[[Request], Awaitable[Response]]):
        self.app = app

    async def __call__(self, request: Request) -> Response:
        """
        Handle incoming requests and catch any unhandled exceptions.
        """
        try:
            return await self.app(request)

        except ToolError as e:
            # These are expected errors that should be returned to client
            logger.warning(f"Tool error in {request.url.path}: {e}")
            raise  # Re-raise to let FastMCP handle it

        except ConnectionError as e:
            logger.error(f"Connection error in {request.url.path}: {e}")
            raise ToolError(
                f"Connection failed: {e!s}. Please check if external services are "
                f"running."
            ) from e

        except TimeoutError as e:
            logger.error(f"Timeout error in {request.url.path}: {e}")
            raise ToolError(
                f"Operation timed out: {e!s}. Please try again with smaller batch "
                f"size or increased timeout."
            ) from e

        except ValueError as e:
            logger.error(f"Validation error in {request.url.path}: {e}")
            raise ToolError(
                f"Invalid input: {e!s}. Please check your parameters and try again."
            ) from e

        except FileNotFoundError as e:
            logger.error(f"File not found error in {request.url.path}: {e}")
            raise ToolError(
                f"File or directory not found: {e!s}. Please check the path exists."
            ) from e

        except PermissionError as e:
            logger.error(f"Permission error in {request.url.path}: {e}")
            raise ToolError(
                f"Permission denied: {e!s}. Please check file/directory permissions."
            ) from e

        except Exception as e:
            # Unexpected errors - log full traceback but return generic message
            error_id = f"error_{hash(str(e)) % 10000:04d}"
            logger.error(
                f"Unexpected error [{error_id}] in {request.url.path}: {e}\n"
                f"Traceback: {traceback.format_exc()}"
            )
            raise ToolError(
                f"An unexpected error occurred [{error_id}]. "
                f"Please check the server logs for more details."
            ) from e
