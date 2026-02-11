import json
from .llm_service import llm_service
from .models import SummaryResponse, SentimentResponse, SentimentLabel
import logging

logger = logging.getLogger(__name__)

class AnalysisService:

    async def summarize(self, text: str) -> SummaryResponse:
        """Generate a summary with key points."""

        system_msg = (
            "You are a professional document analyst. "
            "Return your response as JSON with these exact keys: "
            "summary (string, 2-3 sentences), "
            "key_points (list of 3-5 strings). "
            "Return ONLY valid JSON, no other text."
        )

        prompt = f"Analyze and summarize this text:\n\n{text}"

        response = llm_service.generate(
            prompt=prompt,
            system_message=system_msg,
            temperature=0.3,  # Low temperature for factual accuracy
        )

        # Parse the JSON response from the AI
        data = json.loads(response)

        return SummaryResponse(
            summary=data["summary"],
            key_points=data["key_points"],
            original_length=len(text),
            summary_length=len(data["summary"]),
            compression_ratio=len(data["summary"]) / len(text),
        )

    async def analyze_sentiment(self, text: str) -> SentimentResponse:
        """Analyze the emotional tone of the text."""

        system_msg = (
            "You are a sentiment analysis expert. "
            "Return JSON with these exact keys: "
            "score (float from -1.0 to 1.0), "
            "label (one of: positive, negative, neutral, mixed), "
            "confidence (float from 0.0 to 1.0), "
            "explanation (one sentence why). "
            "Return ONLY valid JSON."
        )

        prompt = f"Analyze the sentiment of this text:\n\n{text}"

        response = llm_service.generate(
            prompt=prompt,
            system_message=system_msg,
            temperature=0.1,  # Very low for consistent analysis
        )

        data = json.loads(response)

        return SentimentResponse(
            score=data["score"],
            label=SentimentLabel(data["label"]),
            confidence=data["confidence"],
            explanation=data["explanation"],
        )

analysis_service = AnalysisService()
