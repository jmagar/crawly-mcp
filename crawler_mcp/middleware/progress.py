"""
Progress tracking middleware for long-running FastMCP operations.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class ProgressTracker:
    """
    Tracks progress for individual operations.
    """

    def __init__(self, operation_id: str):
        self.operation_id = operation_id
        self.start_time = time.time()
        self.current_step = 0
        self.total_steps = 0
        self.status = "starting"
        self.message = ""
        self.metadata: dict[str, Any] = {}
        self.last_update = time.time()

    def update(
        self,
        current: int,
        total: int,
        status: str = "running",
        message: str = "",
        **metadata: Any,
    ) -> None:
        """Update progress information."""
        self.current_step = current
        self.total_steps = total
        self.status = status
        self.message = message
        self.metadata.update(metadata)
        self.last_update = time.time()

    @property
    def progress_percentage(self) -> float:
        """Calculate progress as percentage."""
        if self.total_steps == 0:
            return 0.0
        return (self.current_step / self.total_steps) * 100.0

    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time

    @property
    def estimated_time_remaining(self) -> float | None:
        """Estimate remaining time based on current progress."""
        if self.current_step == 0 or self.total_steps == 0:
            return None

        elapsed = self.elapsed_time
        progress_ratio = self.current_step / self.total_steps

        if progress_ratio > 0:
            estimated_total_time = elapsed / progress_ratio
            return estimated_total_time - elapsed

        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert progress to dictionary."""
        return {
            "operation_id": self.operation_id,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "progress_percentage": self.progress_percentage,
            "status": self.status,
            "message": self.message,
            "elapsed_time": self.elapsed_time,
            "estimated_time_remaining": self.estimated_time_remaining,
            "metadata": self.metadata,
            "last_update": self.last_update,
        }


class ProgressMiddleware:
    """
    Middleware to track and manage progress for long-running operations.
    """

    def __init__(self, app: Callable[[Request], Awaitable[Response]]):
        self.app = app
        self._active_operations: dict[str, ProgressTracker] = {}
        self._cleanup_task: asyncio.Task[None] | None = None

    async def __call__(self, request: Request) -> Response:
        """
        Process request with progress tracking support.
        """
        # Start cleanup task if not already running
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_stale_operations())

        # Process the request normally
        return await self.app(request)

    def create_tracker(self, operation_id: str) -> ProgressTracker:
        """Create a new progress tracker for an operation."""
        tracker = ProgressTracker(operation_id)
        self._active_operations[operation_id] = tracker

        logger.debug(f"Created progress tracker for operation: {operation_id}")
        return tracker

    def get_tracker(self, operation_id: str) -> ProgressTracker | None:
        """Get progress tracker for an operation."""
        return self._active_operations.get(operation_id)

    def remove_tracker(self, operation_id: str) -> ProgressTracker | None:
        """Remove and return progress tracker for an operation."""
        tracker = self._active_operations.pop(operation_id, None)
        if tracker:
            logger.debug(f"Removed progress tracker for operation: {operation_id}")
        return tracker

    def list_active_operations(self) -> dict[str, dict[str, Any]]:
        """Get information about all active operations."""
        return {
            op_id: tracker.to_dict()
            for op_id, tracker in self._active_operations.items()
        }

    async def _cleanup_stale_operations(self) -> None:
        """
        Background task to clean up stale progress trackers.
        """
        while True:
            try:
                current_time = time.time()
                stale_ops = []

                for op_id, tracker in self._active_operations.items():
                    # Remove operations that haven't been updated in 5 minutes
                    if current_time - tracker.last_update > 300:
                        stale_ops.append(op_id)

                for op_id in stale_ops:
                    removed_tracker = self.remove_tracker(op_id)
                    if removed_tracker:
                        logger.info(
                            f"Cleaned up stale progress tracker: {op_id} "
                            f"(last update: {current_time - removed_tracker.last_update:.1f}s ago)"
                        )

                # Sleep for 60 seconds before next cleanup
                await asyncio.sleep(60)

            except asyncio.CancelledError:
                logger.debug("Progress cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in progress cleanup task: {e}")
                await asyncio.sleep(60)  # Continue after error


# Global progress middleware instance
async def _dummy_app(request: Request) -> Response:
    from fastapi import Response  # type: ignore[import-not-found]

    response = Response("Middleware not initialized", status_code=500)
    # Explicitly cast to fix MyPy no-any-return error
    return response  # type: ignore[no-any-return]


progress_middleware = ProgressMiddleware(_dummy_app)  # Will be set by app
