import logging
from typing import Any, Dict, List, Optional
import re

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from lingua import Language, LanguageDetectorBuilder
from fast_langdetect import detect as fast_detect

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Lingua Language Detector")


class DiscriminativeEnglishDetector:
    def __init__(self):
        self.lingua_detector = LanguageDetectorBuilder.from_languages(
            Language.ENGLISH,
            Language.SPANISH,
            Language.FRENCH,
            Language.GERMAN,
            Language.SWEDISH,
            Language.ITALIAN,
            Language.PORTUGUESE,
            Language.DUTCH,
            Language.DANISH
        ).build()

    def is_english(self, text, threshold=0.5):
        """
        True discriminative approach:
        - Compute P(English | text)
        - If P(English) >= threshold, it's English
        - Otherwise, it's not English

        Uses both lingua and fast-langdetect for voting
        """
        text = text.strip()
        if not text or len(text) < 2:
            return True

        words = re.findall(r'\b\w+\b', text)

        # For very short text (1 word), check word-level
        if len(words) == 1:
            word = words[0]

            # Get lingua English probability
            lingua_conf = self.lingua_detector.compute_language_confidence_values(word)
            lingua_english_prob = 0
            for c in lingua_conf:
                if c.language == Language.ENGLISH:
                    lingua_english_prob = c.value
                    break

            # Get fast-langdetect English probability
            try:
                fast_result = fast_detect(word, k=10)  # Get all languages
                fast_english_prob = 0
                for r in fast_result:
                    if r['lang'] == 'en':
                        fast_english_prob = r['score']
                        break
            except:
                fast_english_prob = 0.5  # Neutral if fails

            # Average the two probabilities
            avg_english_prob = (lingua_english_prob + fast_english_prob) / 2

            logger.debug(f"lingua_english_prob: {lingua_english_prob}")
            logger.debug(f"fast_english_prob: {fast_english_prob}")
            logger.debug(f"avg_english_prob: {avg_english_prob}")

            return avg_english_prob >= threshold, avg_english_prob

        # For multi-word text, check each word and use majority voting
        english_votes = 0
        total_votes = 0
        lowest_word_english_prob = 1

        for word in words:
            if len(word) < 2:
                continue

            total_votes += 1

            # Lingua probability
            lingua_conf = self.lingua_detector.compute_language_confidence_values(word)
            lingua_english_prob = 0
            for c in lingua_conf:
                if c.language == Language.ENGLISH:
                    lingua_english_prob = c.value
                    break

            # Fast-langdetect probability
            try:
                fast_result = fast_detect(word, k=10)
                fast_english_prob = 0
                for r in fast_result:
                    if r['lang'] == 'en':
                        fast_english_prob = r['score']
                        break
            except:
                fast_english_prob = 0.5

            # Average probability for this word
            word_english_prob = (lingua_english_prob + fast_english_prob) / 2

            if word_english_prob < lowest_word_english_prob:
                lowest_word_english_prob = word_english_prob

            # Vote: is this word English?
            if word_english_prob >= threshold:
                english_votes += 1

            logger.debug(f"Word: {word}")
            logger.debug(f"lingua_english_prob: {lingua_english_prob}")
            logger.debug(f"fast_english_prob: {fast_english_prob}")
            logger.debug(f"avg_english_prob: {word_english_prob}")

        # All words must be English for text to be English
        if total_votes == 0:
            return True

        # Strict: ALL words must be classified as English
        return english_votes == total_votes, lowest_word_english_prob


# Build detector once at startup
detector = DiscriminativeEnglishDetector()


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


def detect_language(text: str, threshold: float = 0.1) -> List[ContentAnalysisResponse]:
    """
    Detect the primary language of text using discriminative English detection.
    Returns empty list if text is English.
    Returns detection only if text is non-English.
    """
    if not text or not text.strip():
        return []

    # Use the discriminative detector
    is_eng, score = detector.is_english(text, threshold)

    # If it's English, allow it (no detection)
    if is_eng:
        return []

    return [ContentAnalysisResponse(
        start=0,
        end=len(text),
        text=text,
        detection="non_english",
        detection_type="language_detection",
        score=score,
        evidences=[],
        metadata={}
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
    logger.info(f"Received request: contents={request.contents}, params={request.detector_params}")

    response = []
    for content in request.contents:
        detections = detect_language(content)
        response.append(detections)

    logger.info(f"Returning response: {response}")
    return response


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def catch_all(path: str, request: Request):
    """Catch-all to log any unhandled routes."""
    body = await request.body()
    logger.warning(f"Unhandled route: {request.method} /{path} - body: {body}")
    return {"error": f"Unknown endpoint: /{path}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
