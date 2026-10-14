import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chunking import create_chunks
from intent_classifier import build_model, predict_intent
from router import choose_retrieval_strategy


def test_chunking_covers_end():
    chunks = create_chunks(95, 30, 5)
    assert chunks[0] == {"start": 0.0, "end": 30}
    assert chunks[-1]["end"] == 95


def test_invalid_chunking_rejected():
    try:
        create_chunks(10, 0, 0)
    except ValueError:
        return
    assert False, "Expected ValueError"


def test_intent_classifier_returns_valid_label():
    model = build_model()
    result = predict_intent(model, "Summarize the whole video")
    assert result["intent"] in {"whole_video", "frame", "clip"}
    assert 0.0 <= result["confidence"] <= 1.0


def test_router_uses_video_type():
    result = choose_retrieval_strategy("clip", "sports")
    assert result["strategy"] == "event_and_temporal_search"
