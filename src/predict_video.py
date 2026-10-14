from __future__ import annotations
import argparse
import json
from video_classifier import predict_video

def main():
    parser = argparse.ArgumentParser(description="Predict a video category.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--video", required=True)
    args = parser.parse_args()
    print(json.dumps(predict_video(args.model, args.video), indent=2))

if __name__ == "__main__":
    main()
