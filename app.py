from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field
from lingua import Language, LanguageDetectorBuilder

app = FastAPI(title="Lingua Language Detector")

# Build detector once at startup
detector = LanguageDetectorBuilder.from_all_languages().build()

# Confidence threshold - text with English confidence above this is considered English
ENGLISH_THRESHOLD = 0.5


# Request/Response schemas matching guardrails-detectors
class ContentAnalysisHttpRequest(BaseModel):
    contents: List[str] = Field(min_length=1)
    detector_params: Optional[Dict[str, Any]] = None


class ContentAnalysisResponse(BaseModel):
    start: int
    end: int
    text: str
    detection: str
    detection_type: str
    score: float
    evidences: List[Any] = []
    metadata: Dict[str, Any] = {}


def detect_language(text: str, threshold: float = ENGLISH_THRESHOLD) -> List[ContentAnalysisResponse]:
    """
    Detect the primary language of text.
    Returns empty list if text is English (above threshold).
    Returns detection if text is non-English.
    """
    if not text or not text.strip():
        return []

    # Get English confidence for the whole text
    english_confidence = detector.compute_language_confidence(text, Language.ENGLISH)

    # If English confidence is high enough, no detection (it's allowed)
    if english_confidence >= threshold:
        return []

    # Otherwise, find the most likely language
    detected_lang = detector.detect_language_of(text)
    if detected_lang is None:
        return []

    # Get confidence for the detected language
    detected_confidence = detector.compute_language_confidence(text, detected_lang)

    return [ContentAnalysisResponse(
        start=0,
        end=len(text),
        text=text,
        detection=detected_lang.name,
        detection_type="language",
        score=detected_confidence,
        evidences=[],
        metadata={
            "detected_language": detected_lang.name,
            "english_confidence": english_confidence
        }
    )]


@app.get("/health")
def health():
    return "ok"


@app.post("/api/v1/text/contents", response_model=List[List[ContentAnalysisResponse]])
def analyze_contents(request: ContentAnalysisHttpRequest):
    """
    Analyze text contents for language detection.
    Returns empty array for each content that is English.
    Returns detection for non-English content.
    """
    threshold = ENGLISH_THRESHOLD
    if request.detector_params and "threshold" in request.detector_params:
        threshold = float(request.detector_params["threshold"])

    response = []
    for content in request.contents:
        detections = detect_language(content, threshold)
        response.append(detections)

    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
