from lingua import Language, LanguageDetectorBuilder
from fast_langdetect import detect as fast_detect
import re

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

      # self.lingua_detector = LanguageDetectorBuilder.from_all_languages().build()

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

          print(f"lingua_english_prob: {lingua_english_prob}")
          print(f"fast_english_prob: {fast_english_prob}")
          print(f"avg_english_prob: {avg_english_prob}")

          return avg_english_prob >= threshold

      # For multi-word text, check each word and use majority voting
      english_votes = 0
      total_votes = 0

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

          # Vote: is this word English?
          if word_english_prob >= threshold:
              english_votes += 1

          print(f"Word: {word}")
          print(f"lingua_english_prob: {lingua_english_prob}")
          print(f"fast_english_prob: {fast_english_prob}")
          print(f"avg_english_prob: {word_english_prob}")

      # All words must be English for text to be English
      if total_votes == 0:
          return True

      # Strict: ALL words must be classified as English
      return english_votes == total_votes

# Test with different thresholds
detector = DiscriminativeEnglishDetector()

print("True Discriminative Model (P(English) >= threshold):\n")

threshold = 0.1

print(f"\n{'='*60}")
print(f"Threshold: {threshold}")
print(f"{'='*60}\n")

correct = 0
total = 0

for text, expected, category in test_suite:
  result = detector.is_english(text, threshold=threshold)
  status = "✓" if result == expected else "✗"
  correct += (result == expected)
  total += 1

  print(f"{status} '{text}': {'English' if result else 'Not English'}")

print(f"\nAccuracy: {correct}/{total} ({100*correct/total:.1f}%)")