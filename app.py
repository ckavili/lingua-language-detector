import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from lingua import Language, LanguageDetectorBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Lingua Language Detector")

# Exclude languages that cause false positives on short English text
EXCLUDED_LANGUAGES = [
    Language.SHONA,
    Language.XHOSA,
    Language.SOTHO,
    Language.TSONGA,
    Language.TSWANA,
    Language.GANDA,
]

# Build detector from all languages except excluded ones
supported_languages = [lang for lang in Language.all() if lang not in EXCLUDED_LANGUAGES]
detector = (
    LanguageDetectorBuilder
    .from_languages(*supported_languages)
    .with_preloaded_language_models()
    # TODO: If performance is still insufficient after testing, add:
    # .with_low_accuracy_mode()
    # This trades accuracy for ~50% faster detection (uses smaller n-gram models)
    .build()
)

# Minimum confidence threshold for a detection to be considered valid
MIN_CONFIDENCE_THRESHOLD = 0.1


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


def detect_language(text: str) -> List[ContentAnalysisResponse]:
    """
    Detect the primary language of text.
    Returns empty list if text is English.
    Returns detection only if text is non-English.
    """
    if not text or not text.strip():
        return []

    # Detect the primary language
    detected_lang = detector.detect_language_of(text)

    # If can't detect, allow it
    if detected_lang is None:
        return []

    # If English, allow it
    if detected_lang == Language.ENGLISH:
        return []

    # Get confidence for detected language
    detected_confidence = detector.compute_language_confidence(text, detected_lang)

    # If detected language confidence is below threshold, treat as uncertain
    if detected_confidence < MIN_CONFIDENCE_THRESHOLD:
        return []

    # Non-English detected with sufficient confidence - return detection
    english_confidence = detector.compute_language_confidence(text, Language.ENGLISH)
    score = 1.0 - english_confidence

    return [ContentAnalysisResponse(
        start=0,
        end=len(text),
        text=text,
        detection="non_english",
        detection_type="language_detection",
        score=score,
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
    return [detect_language(content) for content in request.contents]


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def catch_all(path: str, request: Request):
    """Catch-all to log any unhandled routes."""
    body = await request.body()
    logger.warning(f"Unhandled route: {request.method} /{path} - body: {body}")
    return {"error": f"Unknown endpoint: /{path}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
