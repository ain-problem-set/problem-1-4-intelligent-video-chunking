"""CLI for the Intelligent Video Chunking baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from intent_classifier import build_model, predict_intent
from chunking import create_chunks
from router import choose_retrieval_strategy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")

    intent = sub.add_parser("intent", help="Classify a user query")
    intent.add_argument("--query", required=True)
    intent.add_argument("--video-type", default="other")

    chunks = sub.add_parser("chunks", help="Create fallback temporal chunks")
    chunks.add_argument("--duration", type=float, required=True)
    chunks.add_argument("--chunk-seconds", type=float, default=30)
    chunks.add_argument("--overlap-seconds", type=float, default=5)

    args = parser.parse_args()
    if args.command == "intent":
        prediction = predict_intent(build_model(), args.query)
        result = {**prediction, "routing": choose_retrieval_strategy(
            prediction["intent"], args.video_type
        )}
        print(json.dumps(result, indent=2))
    elif args.command == "chunks":
        print(json.dumps(create_chunks(
            args.duration, args.chunk_seconds, args.overlap_seconds
        ), indent=2))
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
