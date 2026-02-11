import time
from fastapi import FastAPI, HTTPException
from .models import (
    AnalysisRequest, SummaryResponse,
    SentimentResponse, FullAnalysisResponse
)
from .analysis_service import analysis_service
from .config import settings
import logging

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    version="0.2.0",
)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/summarize", response_model=SummaryResponse)
async def summarize_text(request: AnalysisRequest):
    try:
        return await analysis_service.summarize(request.text)
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        raise HTTPException(status_code=500, detail="Analysis failed")

@app.post("/sentiment", response_model=SentimentResponse)
async def sentiment_analysis(request: AnalysisRequest):
    try:
        return await analysis_service.analyze_sentiment(request.text)
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Analysis failed")

@app.post("/analyze", response_model=FullAnalysisResponse)
async def full_analysis(request: AnalysisRequest):
    start = time.time()
    try:
        summary = await analysis_service.summarize(request.text)
        sentiment = await analysis_service.analyze_sentiment(request.text)
        elapsed = (time.time() - start) * 1000
        return FullAnalysisResponse(
            summary=summary,
            sentiment=sentiment,
            processing_time_ms=round(elapsed, 2),
        )
    except Exception as e:
        logger.error(f"Full analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Analysis failed")
