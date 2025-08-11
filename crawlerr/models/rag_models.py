"""
Data models for RAG (Retrieval-Augmented Generation) operations.
"""
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ConfigDict


class DocumentChunk(BaseModel):
    model_config = ConfigDict()

    """A chunk of document content with metadata."""
    id: str
    content: str
    embedding: Optional[List[float]] = None
    source_url: str
    source_title: Optional[str] = None
    chunk_index: int = 0
    word_count: int = 0
    char_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @field_validator("word_count", mode="before")
    @classmethod
    def calculate_word_count(cls, v, info):
        if v == 0 and info.data and "content" in info.data:
            content = info.data["content"]
            if content:
                return len(content.split())
        return v
    
    @field_validator("char_count", mode="before")
    @classmethod
    def calculate_char_count(cls, v, info):
        if v == 0 and info.data and "content" in info.data:
            content = info.data["content"]
            if content:
                return len(content)
        return v


class SearchMatch(BaseModel):
    model_config = ConfigDict()

    """A search result match with similarity score."""
    document: DocumentChunk
    score: float = Field(ge=0.0, le=1.0)
    relevance: str = Field(default="medium")  # low, medium, high
    highlighted_content: Optional[str] = None
    
    @field_validator("relevance", mode="before")
    @classmethod
    def calculate_relevance(cls, v, info):
        if info.data and "score" in info.data:
            score = info.data["score"]
            if score >= 0.8:
                return "high"
            elif score >= 0.6:
                return "medium"
            else:
                return "low"
        return v


class RagQuery(BaseModel):
    model_config = ConfigDict()

    """Query for RAG search operations."""
    query: str = Field(min_length=1, max_length=1000)
    limit: int = Field(default=10, ge=1, le=100)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    source_filters: Optional[List[str]] = None
    date_range: Optional[tuple[datetime, datetime]] = None
    include_content: bool = True
    include_metadata: bool = True
    rerank: bool = True
    
    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class EmbeddingResult(BaseModel):
    model_config = ConfigDict()

    """Result of embedding generation."""
    text: str
    embedding: List[float]
    model: str
    dimensions: int
    processing_time: float = 0.0
    
    @field_validator("dimensions", mode="before")
    @classmethod
    def calculate_dimensions(cls, v, info):
        if v == 0 and info.data and "embedding" in info.data:
            embedding = info.data["embedding"]
            if embedding:
                return len(embedding)
        return v


class RagResult(BaseModel):
    model_config = ConfigDict()

    """Result of a RAG query operation."""
    query: str
    matches: List[SearchMatch] = Field(default_factory=list)
    total_matches: int = 0
    processing_time: float = 0.0
    embedding_time: float = 0.0
    search_time: float = 0.0
    rerank_time: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @property
    def has_high_confidence_matches(self) -> bool:
        """Check if any matches have high confidence."""
        return any(match.relevance == "high" for match in self.matches)
    
    @property
    def average_score(self) -> float:
        """Calculate average similarity score."""
        if not self.matches:
            return 0.0
        return sum(match.score for match in self.matches) / len(self.matches)
    
    @property
    def best_match_score(self) -> float:
        """Get the highest similarity score."""
        if not self.matches:
            return 0.0
        return max(match.score for match in self.matches)