"""
GitHub Webhook Server Module for Crawler MCP

This module provides GitHub organization webhook processing for automatic
AI prompt extraction from PR comments and reviews.
"""

from .server import app, config, processor

__all__ = ["app", "config", "processor"]
