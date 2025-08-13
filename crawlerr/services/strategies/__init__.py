"""
Crawling strategies for different source types.
"""

from .base_strategy import BaseCrawlStrategy
from .directory_strategy import DirectoryCrawlStrategy, DirectoryRequest
from .repository_strategy import RepositoryCrawlStrategy, RepositoryRequest
from .web_strategy import WebCrawlStrategy

__all__ = [
    "BaseCrawlStrategy",
    "DirectoryCrawlStrategy",
    "DirectoryRequest",
    "RepositoryCrawlStrategy",
    "RepositoryRequest",
    "WebCrawlStrategy",
]
