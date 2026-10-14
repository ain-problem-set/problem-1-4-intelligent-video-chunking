"""Lightweight user-query intent classifier.

Labels:
- whole_video: asks about the video globally or requests a summary.
- frame: asks about a specific frame, timestamp, or visible detail at a moment.
- clip: asks for an event, action, or section spanning a time interval.

The built-in examples are a runnable baseline only. For a real submission, expand
and evaluate with a separately labeled query dataset.
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

EXAMPLES = [
    ("Summarize the whole video", "whole_video"),
    ("What is this video about overall?", "whole_video"),
    ("Give me the main points from beginning to end", "whole_video"),
    ("Describe the overall story of this footage", "whole_video"),
    ("What are the key events in the entire recording?", "whole_video"),
    ("What is visible at 00:02:15?", "frame"),
    ("Describe the frame at 30 seconds", "frame"),
    ("What object is shown at timestamp 1:12?", "frame"),
    ("Is the person wearing a red shirt at this moment?", "frame"),
    ("Read the text visible in this frame", "frame"),
    ("Find the part where the speaker discusses pricing", "clip"),
    ("When does the player score a goal?", "clip"),
    ("Show the segment where the dog jumps into the water", "clip"),
    ("Find every scene where the car appears", "clip"),
    ("When does the presenter switch to the next slide?", "clip"),
    ("Locate the moment the person opens the door", "clip"),
    ("Which section contains the product demonstration?", "clip"),
    ("Find the interval when the alarm starts ringing", "clip"),
]


def build_model() -> Pipeline:
    texts, labels = zip(*EXAMPLES)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), sublinear_tf=True,
            max_features=10000, strip_accents="unicode"
        )),
        ("classifier", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        )),
    ])
    model.fit(list(texts), list(labels))
    return model


def predict_intent(model: Pipeline, query: str) -> dict:
    if not query or not query.strip():
        raise ValueError("query must not be empty")
    probabilities = model.predict_proba([query])[0]
    index = int(probabilities.argmax())
    return {
        "intent": str(model.classes_[index]),
        "confidence": float(probabilities[index]),
        "probabilities": {
            str(label): float(prob)
            for label, prob in zip(model.classes_, probabilities)
        },
    }
