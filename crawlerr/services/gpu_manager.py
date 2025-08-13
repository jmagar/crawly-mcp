"""
GPU management with optimized monitoring and resource allocation for RTX 4070.
"""

import asyncio
import logging
import time
from typing import Any

try:
    import GPUtil
    import psutil

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    logging.getLogger(__name__).warning("GPU monitoring libraries not available")

import contextlib

from ..config import settings

logger = logging.getLogger(__name__)


class GPUManager:
    """
    Optimized GPU manager with reduced monitoring overhead.
    Monitors every 30 seconds instead of 2 seconds for 95% overhead reduction.
    """

    def __init__(self) -> None:
        self.gpu_available = GPU_AVAILABLE and settings.gpu_acceleration
        self.monitor_interval = settings.gpu_monitor_interval
        self.memory_limit_mb = settings.gpu_memory_limit_mb
        self.fallback_enabled = settings.gpu_fallback_enabled
        self.force_enable = settings.gpu_force_enable

        # Monitoring state
        self._last_check = 0.0
        self._cached_stats: dict[str, Any] = {}
        self._monitoring_task: asyncio.Task[None] | None = None
        self._gpu_healthy = True
        self._consecutive_failures = 0

        # Performance tracking
        self._total_checks = 0
        self._failed_checks = 0

    async def initialize(self) -> None:
        """Initialize GPU monitoring if available."""
        if not self.gpu_available:
            logger.warning(
                "GPU acceleration disabled - libraries not available or disabled in config"
            )
            return

        logger.info("Initializing GPU manager for RTX 4070")

        # Initial GPU check
        if await self._check_gpu_health():
            logger.info("GPU acceleration enabled and healthy")
            if settings.gpu_health_check_enabled:
                self._start_monitoring()
        else:
            if self.force_enable:
                logger.warning(
                    "GPU unhealthy but force_enable=True, continuing with GPU acceleration"
                )
                self._gpu_healthy = True
            else:
                logger.error(
                    "GPU unhealthy and force_enable=False, disabling GPU acceleration"
                )
                self.gpu_available = False

    def _start_monitoring(self) -> None:
        """Start background GPU monitoring task."""
        if self._monitoring_task is None:
            self._monitoring_task = asyncio.create_task(self._monitor_gpu_loop())
            logger.debug(f"Started GPU monitoring (interval: {self.monitor_interval}s)")

    async def _monitor_gpu_loop(self) -> None:
        """Background monitoring loop with reduced frequency."""
        while True:
            try:
                await asyncio.sleep(self.monitor_interval)
                await self._check_gpu_health()
            except asyncio.CancelledError:
                logger.debug("GPU monitoring cancelled")
                break
            except Exception as e:
                logger.error(f"GPU monitoring error: {e}")
                await asyncio.sleep(self.monitor_interval)

    async def _check_gpu_health(self) -> bool:
        """
        Check GPU health with caching to reduce overhead.
        Only performs actual check if cache is stale.
        """
        current_time = time.time()

        # Use cached result if recent enough (within monitor interval)
        if current_time - self._last_check < self.monitor_interval:
            return self._gpu_healthy

        self._last_check = current_time
        self._total_checks += 1

        if not GPU_AVAILABLE:
            return False

        try:
            # Get GPU stats
            gpus = GPUtil.getGPUs()
            if not gpus:
                self._failed_checks += 1
                self._consecutive_failures += 1
                logger.warning("No GPUs detected")
                return False

            gpu = gpus[0]  # Assume single RTX 4070
            memory_used_mb = gpu.memoryUsed
            memory_total_mb = gpu.memoryTotal
            memory_percent = (memory_used_mb / memory_total_mb) * 100

            # Update cached stats
            self._cached_stats = {
                "gpu_id": gpu.id,
                "gpu_name": gpu.name,
                "memory_used_mb": memory_used_mb,
                "memory_total_mb": memory_total_mb,
                "memory_percent": memory_percent,
                "gpu_utilization": gpu.load * 100,
                "gpu_temperature": gpu.temperature,
                "last_updated": current_time,
            }

            # Check health conditions
            healthy = True
            health_issues = []

            # Memory check
            if memory_used_mb > self.memory_limit_mb:
                healthy = False
                health_issues.append(
                    f"GPU memory usage ({memory_used_mb}MB) exceeds limit ({self.memory_limit_mb}MB)"
                )

            # Temperature check (safe limit for RTX 4070)
            if gpu.temperature > 83:  # RTX 4070 thermal throttle point
                healthy = False
                health_issues.append(f"GPU temperature ({gpu.temperature}°C) too high")

            # Utilization check (detect if GPU is stuck)
            if gpu.load > 0.98:  # 98% utilization might indicate stuck process
                logger.warning(f"GPU utilization very high: {gpu.load * 100:.1f}%")

            if healthy:
                self._consecutive_failures = 0
                self._gpu_healthy = True
            else:
                self._failed_checks += 1
                self._consecutive_failures += 1
                self._gpu_healthy = False
                logger.warning(f"GPU health issues: {'; '.join(health_issues)}")

                # Disable GPU after 3 consecutive failures (unless forced)
                if self._consecutive_failures >= 3 and not self.force_enable:
                    logger.error("GPU disabled due to consecutive failures")
                    self.gpu_available = False

            return healthy

        except Exception as e:
            self._failed_checks += 1
            self._consecutive_failures += 1
            logger.error(f"GPU health check failed: {e}")

            # Disable GPU after 5 consecutive check failures (unless forced)
            if self._consecutive_failures >= 5 and not self.force_enable:
                logger.error("GPU disabled due to consecutive check failures")
                self.gpu_available = False
                return False

            return False

    def is_gpu_available(self) -> bool:
        """Check if GPU acceleration is available and healthy."""
        return self.gpu_available and self._gpu_healthy

    def should_use_gpu(self) -> bool:
        """Determine if GPU should be used for current operation."""
        if not self.is_gpu_available():
            return False

        # Additional logic can be added here for dynamic GPU usage decisions
        # For now, use GPU if available and healthy
        return True

    def get_gpu_config(self) -> dict[str, Any]:
        """Get optimized GPU configuration for Chrome."""
        if not self.should_use_gpu():
            return {"gpu_enabled": False, "chrome_flags": []}

        base_flags = settings.chrome_gpu_flags.split()
        advanced_flags = settings.chrome_advanced_gpu_flags.split()

        # Additional RTX 4070 specific optimizations
        rtx4070_flags = [
            "--enable-gpu-memory-buffer-video-frames",
            "--enable-gpu-memory-buffer-compositor-resources",
            "--enable-native-gpu-memory-buffers",
            "--gpu-memory-buffer-max-textures-with-target=2048",
        ]

        return {
            "gpu_enabled": True,
            "chrome_flags": base_flags + advanced_flags + rtx4070_flags,
            "memory_limit_mb": self.memory_limit_mb,
        }

    async def allocate_gpu_memory(self, estimated_mb: int) -> bool:
        """
        Check if GPU can accommodate additional memory allocation.

        Args:
            estimated_mb: Estimated memory requirement in MB

        Returns:
            True if allocation is likely to succeed
        """
        if not self.is_gpu_available():
            return False

        # Force check if cache is stale
        await self._check_gpu_health()

        if not self._cached_stats:
            return False

        available_mb = (
            self._cached_stats["memory_total_mb"] - self._cached_stats["memory_used_mb"]
        )

        # Keep 20% buffer for system
        usable_mb = available_mb * 0.8

        can_allocate = estimated_mb <= usable_mb

        if not can_allocate:
            logger.warning(
                f"GPU memory allocation rejected: need {estimated_mb}MB, "
                f"available {usable_mb:.0f}MB (with buffer)"
            )

        return can_allocate

    def get_stats(self) -> dict[str, Any]:
        """Get current GPU statistics and manager state."""
        base_stats = {
            "gpu_available": self.gpu_available,
            "gpu_healthy": self._gpu_healthy,
            "monitor_interval": self.monitor_interval,
            "memory_limit_mb": self.memory_limit_mb,
            "fallback_enabled": self.fallback_enabled,
            "force_enable": self.force_enable,
            "consecutive_failures": self._consecutive_failures,
            "total_checks": self._total_checks,
            "failed_checks": self._failed_checks,
            "failure_rate": (self._failed_checks / max(1, self._total_checks)) * 100,
        }

        if self._cached_stats:
            base_stats.update(self._cached_stats)

        return base_stats

    async def handle_gpu_failure(self, operation: str, error: Exception) -> bool:
        """
        Handle GPU failure during an operation.

        Args:
            operation: Description of the failed operation
            error: The exception that occurred

        Returns:
            True if fallback should be used, False if operation should fail
        """
        logger.error(f"GPU failure during {operation}: {error}")

        # Force health check
        await self._check_gpu_health()

        if self.fallback_enabled and not self.force_enable:
            logger.info(f"Falling back to CPU for {operation}")
            return True
        elif self.force_enable:
            logger.warning(
                f"GPU failure ignored due to force_enable=True for {operation}"
            )
            return False
        else:
            logger.error(f"No fallback configured for GPU failure in {operation}")
            return False

    async def shutdown(self) -> None:
        """Shutdown GPU monitoring."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitoring_task
            self._monitoring_task = None

        logger.info(
            f"GPU manager shutdown (total checks: {self._total_checks}, failures: {self._failed_checks})"
        )


# Global GPU manager instance
_gpu_manager: GPUManager | None = None


async def get_gpu_manager() -> GPUManager:
    """Get or create the global GPU manager instance."""
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUManager()
        await _gpu_manager.initialize()
    return _gpu_manager


async def cleanup_gpu_manager() -> None:
    """Cleanup the global GPU manager."""
    global _gpu_manager
    if _gpu_manager:
        await _gpu_manager.shutdown()
        _gpu_manager = None
