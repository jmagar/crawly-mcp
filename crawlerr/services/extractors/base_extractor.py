"""
Base content extraction strategy.
"""

import logging
from abc import ABC, abstractmethod

from crawl4ai.extraction_strategy import ExtractionStrategy

logger = logging.getLogger(__name__)


class BaseExtractor(ABC):
    """
    Base class for content extractors with common functionality.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def get_strategy(self) -> ExtractionStrategy | None:
        """
        Get the Crawl4AI extraction strategy instance.

        Returns:
            ExtractionStrategy instance or None for default extraction
        """
        pass

    @abstractmethod
    def supports_content_type(self, content_type: str) -> bool:
        """
        Check if this extractor supports the given content type.

        Args:
            content_type: MIME type or content description

        Returns:
            True if this extractor can handle the content type
        """
        pass
