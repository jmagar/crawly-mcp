"""
Comprehensive tests for middleware modules to maximize coverage.

This module focuses on testing all code paths in the middleware
to achieve high coverage on error, logging, and progress middleware.
"""

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp import Context
from fastmcp.exceptions import ToolError

from crawler_mcp.middleware.error import ErrorHandlingMiddleware
from crawler_mcp.middleware.logging import LoggingMiddleware
from crawler_mcp.middleware.progress import (
    ProgressMiddleware,
    ProgressTracker,
    progress_middleware,
)


class TestErrorHandlingMiddleware:
    """Comprehensive tests for ErrorHandlingMiddleware."""

    @pytest.fixture
    def middleware(self):
        """Create ErrorHandlingMiddleware instance."""
        return ErrorHandlingMiddleware()

    @pytest.fixture
    def mock_context(self):
        """Create mock FastMCP context."""
        mock_ctx = MagicMock(spec=Context)
        mock_ctx.info = AsyncMock()
        mock_ctx.error = AsyncMock()
        return mock_ctx

    @pytest.mark.asyncio
    async def test_successful_operation(self, middleware, mock_context):
        """Test middleware with successful operation."""

        async def success_handler(ctx: Context) -> dict[str, Any]:
            return {"result": "success"}

        result = await middleware.process(mock_context, success_handler)
        assert result == {"result": "success"}
        mock_context.info.assert_not_called()
        mock_context.error.assert_not_called()

    @pytest.mark.asyncio
    async def test_tool_error_handling(self, middleware, mock_context):
        """Test handling of ToolError exceptions."""

        async def error_handler(ctx: Context) -> dict[str, Any]:
            raise ToolError("Test tool error")

        with pytest.raises(ToolError, match="Test tool error"):
            await middleware.process(mock_context, error_handler)

        # Should log the error
        mock_context.error.assert_called_once()
        call_args = mock_context.error.call_args[0]
        assert "Tool error occurred" in call_args[0]

    @pytest.mark.asyncio
    async def test_generic_exception_handling(self, middleware, mock_context):
        """Test handling of generic exceptions."""

        async def error_handler(ctx: Context) -> dict[str, Any]:
            raise ValueError("Test generic error")

        with pytest.raises(ToolError, match="Operation failed"):
            await middleware.process(mock_context, error_handler)

        # Should log the error
        mock_context.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_keyboard_interrupt_handling(self, middleware, mock_context):
        """Test handling of KeyboardInterrupt."""

        async def error_handler(ctx: Context) -> dict[str, Any]:
            raise KeyboardInterrupt()

        with pytest.raises(ToolError, match="Operation was cancelled"):
            await middleware.process(mock_context, error_handler)

        mock_context.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_asyncio_cancelled_error_handling(self, middleware, mock_context):
        """Test handling of asyncio.CancelledError."""

        async def error_handler(ctx: Context) -> dict[str, Any]:
            raise asyncio.CancelledError()

        with pytest.raises(ToolError, match="Operation was cancelled"):
            await middleware.process(mock_context, error_handler)

        mock_context.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_timeout_error_handling(self, middleware, mock_context):
        """Test handling of TimeoutError."""

        async def error_handler(ctx: Context) -> dict[str, Any]:
            raise TimeoutError("Operation timed out")

        with pytest.raises(ToolError, match="Operation timed out"):
            await middleware.process(mock_context, error_handler)

        mock_context.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_error_handling(self, middleware, mock_context):
        """Test handling of ConnectionError."""

        async def error_handler(ctx: Context) -> dict[str, Any]:
            raise ConnectionError("Connection failed")

        with pytest.raises(ToolError, match="Connection error"):
            await middleware.process(mock_context, error_handler)

        mock_context.error.assert_called_once()


class TestLoggingMiddleware:
    """Comprehensive tests for LoggingMiddleware."""

    @pytest.fixture
    def middleware(self):
        """Create LoggingMiddleware instance."""
        return LoggingMiddleware()

    @pytest.fixture
    def mock_context(self):
        """Create mock FastMCP context."""
        mock_ctx = MagicMock(spec=Context)
        mock_ctx.info = AsyncMock()
        mock_ctx.error = AsyncMock()
        return mock_ctx

    @pytest.mark.asyncio
    async def test_successful_operation_logging(self, middleware, mock_context):
        """Test logging for successful operations."""

        async def success_handler(ctx: Context) -> dict[str, Any]:
            return {"result": "success", "data": "test"}

        with patch("crawler_mcp.middleware.logging.logger") as mock_logger:
            result = await middleware.process(mock_context, success_handler)

            assert result == {"result": "success", "data": "test"}

            # Check that operation was logged
            assert mock_logger.info.call_count >= 2  # Start and end logs

            # Check start log
            start_call = mock_logger.info.call_args_list[0]
            assert "Operation started" in start_call[0][0]

            # Check end log
            end_call = mock_logger.info.call_args_list[-1]
            assert "Operation completed" in end_call[0][0]
            assert "success_handler" in end_call[0][0]

    @pytest.mark.asyncio
    async def test_error_operation_logging(self, middleware, mock_context):
        """Test logging for failed operations."""

        async def error_handler(ctx: Context) -> dict[str, Any]:
            raise ValueError("Test error")

        with patch("crawler_mcp.middleware.logging.logger") as mock_logger:
            with pytest.raises(ValueError):
                await middleware.process(mock_context, error_handler)

            # Check that error was logged
            assert mock_logger.error.called
            error_call = mock_logger.error.call_args
            assert "Operation failed" in error_call[0][0]
            assert "error_handler" in error_call[0][0]

    @pytest.mark.asyncio
    async def test_performance_logging(self, middleware, mock_context):
        """Test performance timing logging."""

        async def slow_handler(ctx: Context) -> dict[str, Any]:
            await asyncio.sleep(0.1)  # Simulate slow operation
            return {"result": "slow"}

        with patch("crawler_mcp.middleware.logging.logger") as mock_logger:
            result = await middleware.process(mock_context, slow_handler)

            assert result == {"result": "slow"}

            # Check that timing was logged
            end_call = mock_logger.info.call_args_list[-1]
            assert "completed in" in end_call[0][0]
            assert "ms" in end_call[0][0]


class TestProgressMiddleware:
    """Comprehensive tests for ProgressMiddleware and ProgressTracker."""

    @pytest.fixture
    def middleware(self):
        """Create ProgressMiddleware instance."""
        return ProgressMiddleware()

    @pytest.fixture
    def mock_context(self):
        """Create mock FastMCP context."""
        mock_ctx = MagicMock(spec=Context)
        mock_ctx.report_progress = AsyncMock()
        return mock_ctx

    def test_progress_tracker_creation(self):
        """Test ProgressTracker creation and properties."""
        tracker = ProgressTracker("test-op-123", "Test Operation")

        assert tracker.operation_id == "test-op-123"
        assert tracker.name == "Test Operation"
        assert tracker.current_step == 0
        assert tracker.total_steps == 0
        assert tracker.status == "pending"
        assert tracker.start_time is not None
        assert tracker.end_time is None

    def test_progress_tracker_update(self):
        """Test ProgressTracker update functionality."""
        tracker = ProgressTracker("test-op", "Test")

        # Test update with progress
        tracker.update_progress(5, 10, "Processing...")
        assert tracker.current_step == 5
        assert tracker.total_steps == 10
        assert tracker.status == "running"
        assert tracker.message == "Processing..."

        # Test completion
        tracker.complete("All done!")
        assert tracker.status == "completed"
        assert tracker.message == "All done!"
        assert tracker.end_time is not None

    def test_progress_tracker_error(self):
        """Test ProgressTracker error handling."""
        tracker = ProgressTracker("test-op", "Test")

        tracker.error("Something went wrong")
        assert tracker.status == "error"
        assert tracker.message == "Something went wrong"
        assert tracker.end_time is not None

    def test_progress_tracker_to_dict(self):
        """Test ProgressTracker serialization."""
        tracker = ProgressTracker("test-op", "Test")
        tracker.update_progress(3, 5, "Working...")

        data = tracker.to_dict()
        assert data["operation_id"] == "test-op"
        assert data["name"] == "Test"
        assert data["current_step"] == 3
        assert data["total_steps"] == 5
        assert data["status"] == "running"
        assert data["message"] == "Working..."
        assert "start_time" in data
        assert "progress_percentage" in data

    def test_progress_middleware_create_tracker(self, middleware):
        """Test creating progress trackers."""
        tracker = middleware.create_tracker("operation-1", "Test Operation")

        assert tracker.operation_id == "operation-1"
        assert tracker.name == "Test Operation"
        assert "operation-1" in middleware.active_operations

    def test_progress_middleware_get_tracker(self, middleware):
        """Test getting existing trackers."""
        # Create a tracker
        tracker1 = middleware.create_tracker("op-1", "Operation 1")

        # Get the same tracker
        tracker2 = middleware.get_tracker("op-1")
        assert tracker1 is tracker2

        # Try to get non-existent tracker
        tracker3 = middleware.get_tracker("non-existent")
        assert tracker3 is None

    def test_progress_middleware_remove_tracker(self, middleware):
        """Test removing trackers."""
        # Create a tracker
        tracker = middleware.create_tracker("removable", "Removable Op")
        assert "removable" in middleware.active_operations

        # Remove it
        middleware.remove_tracker("removable")
        assert "removable" not in middleware.active_operations

        # Remove non-existent tracker (should not error)
        middleware.remove_tracker("non-existent")

    def test_progress_middleware_list_operations(self, middleware):
        """Test listing active operations."""
        # Start with no operations
        operations = middleware.list_operations()
        assert len(operations) == 0

        # Add some operations
        middleware.create_tracker("op-1", "Operation 1")
        middleware.create_tracker("op-2", "Operation 2")

        operations = middleware.list_operations()
        assert len(operations) == 2
        assert any(op["operation_id"] == "op-1" for op in operations)
        assert any(op["operation_id"] == "op-2" for op in operations)

    def test_progress_middleware_cleanup_completed(self, middleware):
        """Test cleanup of completed operations."""
        # Create and complete some operations
        tracker1 = middleware.create_tracker("completed-1", "Completed 1")
        tracker1.complete("Done")

        tracker2 = middleware.create_tracker("running-1", "Running 1")
        tracker2.update_progress(1, 2, "In progress")

        tracker3 = middleware.create_tracker("error-1", "Error 1")
        tracker3.error("Failed")

        # Should have 3 operations
        assert len(middleware.active_operations) == 3

        # Cleanup completed/errored operations
        middleware.cleanup_completed()

        # Should only have the running operation
        assert len(middleware.active_operations) == 1
        assert "running-1" in middleware.active_operations

    @pytest.mark.asyncio
    async def test_progress_middleware_process(self, middleware, mock_context):
        """Test progress middleware processing."""
        operation_id = None

        async def tracked_handler(ctx: Context) -> dict[str, Any]:
            nonlocal operation_id
            # Simulate creating progress tracking
            operation_id = "test-op"
            tracker = middleware.create_tracker(operation_id, "Test Operation")
            tracker.update_progress(1, 3, "Step 1")
            await asyncio.sleep(0.01)
            tracker.update_progress(2, 3, "Step 2")
            await asyncio.sleep(0.01)
            tracker.complete("Finished")
            return {"result": "success"}

        result = await middleware.process(mock_context, tracked_handler)

        assert result == {"result": "success"}
        assert operation_id in middleware.active_operations

        # Check tracker state
        tracker = middleware.get_tracker(operation_id)
        assert tracker.status == "completed"
        assert tracker.current_step == 2
        assert tracker.total_steps == 3

    def test_global_progress_middleware_instance(self):
        """Test the global progress middleware instance."""
        # Test the global instance
        assert progress_middleware is not None
        assert isinstance(progress_middleware, ProgressMiddleware)

        # Test creating tracker through global instance
        tracker = progress_middleware.create_tracker("global-test", "Global Test")
        assert tracker.operation_id == "global-test"

        # Cleanup
        progress_middleware.remove_tracker("global-test")

    def test_progress_tracker_edge_cases(self):
        """Test edge cases in ProgressTracker."""
        tracker = ProgressTracker("edge-case", "Edge Case Test")

        # Test updating with zero total steps
        tracker.update_progress(0, 0, "No steps")
        assert tracker.progress_percentage == 0.0

        # Test updating with current > total
        tracker.update_progress(10, 5, "Over limit")
        assert tracker.progress_percentage == 100.0  # Should cap at 100%

        # Test multiple completions
        tracker.complete("First completion")
        first_end_time = tracker.end_time

        time.sleep(0.001)  # Ensure different timestamp
        tracker.complete("Second completion")
        # End time should not change after first completion
        assert tracker.end_time == first_end_time

    def test_progress_middleware_concurrent_operations(self, middleware):
        """Test handling multiple concurrent operations."""
        # Create multiple trackers
        trackers = {}
        for i in range(5):
            op_id = f"concurrent-{i}"
            trackers[op_id] = middleware.create_tracker(op_id, f"Concurrent Op {i}")

        # Update them in different ways
        trackers["concurrent-0"].update_progress(1, 2, "Running")
        trackers["concurrent-1"].complete("Done")
        trackers["concurrent-2"].error("Failed")
        trackers["concurrent-3"].update_progress(3, 5, "In progress")
        trackers["concurrent-4"].update_progress(2, 2, "Almost done")

        # Verify states
        operations = middleware.list_operations()
        assert len(operations) == 5

        # Check specific states
        assert middleware.get_tracker("concurrent-0").status == "running"
        assert middleware.get_tracker("concurrent-1").status == "completed"
        assert middleware.get_tracker("concurrent-2").status == "error"
        assert middleware.get_tracker("concurrent-3").status == "running"
        assert middleware.get_tracker("concurrent-4").status == "running"

        # Cleanup all
        for op_id in trackers:
            middleware.remove_tracker(op_id)
