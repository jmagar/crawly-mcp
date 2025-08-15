"""
Crawling strategies for different source types.
"""

from .base import BaseCrawlStrategy
from .directory import DirectoryCrawlStrategy, DirectoryRequest
from .repository import RepositoryCrawlStrategy, RepositoryRequest
from .web import WebCrawlStrategy

__all__ = [
    "BaseCrawlStrategy",
    "DirectoryCrawlStrategy",
    "DirectoryRequest",
    "RepositoryCrawlStrategy",
    "RepositoryRequest",
    "WebCrawlStrategy",
]
