# Problem 1.4: Intelligent Video Chunking

This repository contains a lightweight starter baseline for video-type classification, query-intent classification, temporal chunking, and retrieval routing.

## Setup
```bash
pip install -r requirements.txt
```

## Run
```bash
python src/main.py --help
python src/main.py intent --query "When does the player score a goal?" --video-type sports
python src/main.py chunks --duration 95 --chunk-seconds 30 --overlap-seconds 5
pytest -q
```

## Train the video classifier
Create a CSV manifest with `path,label` columns, with paths relative to the CSV or absolute paths. Include representative labeled examples for at least two categories.

```bash
python src/train_video_classifier.py --manifest data/videos.csv --model artifacts/video_classifier.joblib
python src/predict_video.py --model artifacts/video_classifier.joblib --video path/to/video.mp4
```

The training and prediction scripts are provided for local labeled data. Use real validation data and report measured metrics; do not treat the built-in query examples or hand-written strategy mapping as validated performance.

See `docs/README.md` for architecture, feature engineering, evaluation, deployment, and limitations.
