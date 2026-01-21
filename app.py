import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from lingua import Language, LanguageDetectorBuilder
import re
from fast_langdetect import detect as fast_detect

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
    Language.LATIN,      # Dead language, causes false positives on English words with Latin roots
    Language.ESPERANTO,  # Constructed language, causes false positives on English
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
MIN_CONFIDENCE_THRESHOLD = 0.15  # Require reasonable confidence before flagging
# Detected language must be this many times more confident than English
MIN_CONFIDENCE_RATIO = 2.5  # Lowered from 3.0 to catch clearer non-English like "comment vas-tu?"


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
    # MIN_CONFIDENCE_THRESHOLD = 0.10  # Require reasonable confidence before flagging
    # # Detected language must be this many times more confident than English
    # MIN_CONFIDENCE_RATIO = 1.5  # Lowered from 3.0 to catch clearer non-English like "comment vas-tu?"
      
    # TODO: Remove debug logging before production
    if not text or not text.strip():
        return []

    words = re.findall(r'\b\w+\b', text)
    if len(words) == 1:
        word = words[0]

        # Get lingua English probability
        lingua_conf = detector.compute_language_confidence_values(word)
        lingua_english_prob = 0
        for c in lingua_conf:
            if c.language == Language.ENGLISH:
                lingua_english_prob = c.value
                break

        # Get fast-langdetect English probability
        # try:
        fast_result = fast_detect(word, k=10)  # Get all languages
        fast_english_prob = 0
        for r in fast_result:
            if r['lang'] == 'en':
                fast_english_prob = r['score']
                break
        # except:
        #     fast_english_prob = 0.5  # Neutral if fails


        #   # Average the two probabilities
        avg_english_prob = (lingua_english_prob + fast_english_prob) / 2

        logger.info(f"avg_english_prob: '{avg_english_prob}'")

        if avg_english_prob >= 0.11:
            logger.info(f"Allowing '{text}'")
            return []
        else:
            logger.info(f"Blocking '{text}'")
            detected_lang = detector.detect_language_of(text)
            resp = [ContentAnalysisResponse(
                start=0,
                end=len(text),
                text=text,
                detection="non_english",
                detection_type="language_detection",
                score=0.0,
                evidences=[],
                metadata={
                    "detected_language": detected_lang.name,
                    "english_confidence": 0.0
                }
            )]
            logger.info(f"Sending: {resp}")
            return resp

    # Detect the primary language
    detected_lang = detector.detect_language_of(text)

    # If can't detect, allow it
    if detected_lang is None:
        logger.info(f"Text: '{text}' | No language detected")
        return []

    # If English, allow it
    if detected_lang == Language.ENGLISH:
        logger.info(f"Text: '{text}' | English detected, allowing")
        return []

    # Get confidence scores
    detected_confidence = detector.compute_language_confidence(text, detected_lang)
    english_confidence = detector.compute_language_confidence(text, Language.ENGLISH)

    ratio = detected_confidence / english_confidence if english_confidence > 0 else float('inf')
    logger.info(f"Text: '{text}' | Detected: {detected_lang.name} ({detected_confidence:.3f}) vs English ({english_confidence:.3f}) | Ratio: {ratio:.2f}x")

    # If detected language confidence is below threshold, treat as uncertain
    if detected_confidence < MIN_CONFIDENCE_THRESHOLD:
        logger.info(f"  -> Ignored: confidence {detected_confidence:.3f} < {MIN_CONFIDENCE_THRESHOLD}")
        return []

    # If detected language isn't significantly more confident than English, allow it
    # This handles short ambiguous text like "hello" that could be either
    if english_confidence > 0 and detected_confidence < (english_confidence * MIN_CONFIDENCE_RATIO):
        logger.info(f"  -> Ignored: ratio < {MIN_CONFIDENCE_RATIO}x")
        return []
    
    logger.info(f"  -> Flagged as non-English")

    # Non-English detected with sufficient confidence - return detection
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
