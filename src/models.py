from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class SentimentLabel(str, Enum):
    """Possible sentiment categories."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"

class AnalysisRequest(BaseModel):
    """What the user sends to us."""
    text: str = Field(
        ...,
        min_length=10,
        max_length=10000,
        description="Text to analyze (10-10,000 characters)"
    )
    language: str = Field(default="auto")

class SummaryResponse(BaseModel):
    """What we send back for summarization."""
    summary: str
    key_points: List[str]
    original_length: int
    summary_length: int
    compression_ratio: float

class SentimentResponse(BaseModel):
    """What we send back for sentiment analysis."""
    score: float = Field(ge=-1.0, le=1.0)
    label: SentimentLabel
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str

class FullAnalysisResponse(BaseModel):
    """Combined analysis response."""
    summary: SummaryResponse
    sentiment: SentimentResponse
    processing_time_ms: float
