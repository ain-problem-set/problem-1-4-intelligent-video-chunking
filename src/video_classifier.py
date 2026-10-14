"""Video feature extraction and lightweight video-type classifier.

Training requires a CSV manifest with columns: path,label. Labels should use
categories relevant to chunking, e.g. sports, podcast, gameplay, wildlife,
egocentric, meeting, surveillance, concert, other.

This baseline uses inexpensive visual/metadata features. It is not expected to
reliably identify semantic categories without representative labeled training
videos. Audio/transcript features can be added for speech-heavy classes.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

FEATURE_NAMES = [
    "duration_seconds", "fps", "width", "height", "aspect_ratio",
    "mean_brightness", "brightness_std", "mean_saturation",
    "mean_edge_density", "mean_frame_difference",
]


def extract_features(video_path: str, samples: int = 12) -> dict[str, float]:
    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0)
        height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)
        duration = frame_count / fps if fps > 0 else 0.0
        if frame_count <= 0 or width <= 0 or height <= 0:
            raise ValueError(f"Missing video metadata: {video_path}")

        indices = np.linspace(0, max(0, int(frame_count) - 1),
                              num=max(1, min(samples, int(frame_count))),
                              dtype=int)
        brightness, saturation, edge_density, frame_diffs = [], [], [], []
        previous = None
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            brightness.append(float(gray.mean() / 255.0))
            saturation.append(float(hsv[:, :, 1].mean() / 255.0))
            edges = cv2.Canny(gray, 80, 160)
            edge_density.append(float((edges > 0).mean()))
            if previous is not None:
                frame_diffs.append(float(cv2.absdiff(gray, previous).mean() / 255.0))
            previous = gray

        if not brightness:
            raise ValueError(f"Could not decode sampled frames: {video_path}")
        return {
            "duration_seconds": duration,
            "fps": fps,
            "width": width,
            "height": height,
            "aspect_ratio": width / height if height else 0.0,
            "mean_brightness": float(np.mean(brightness)),
            "brightness_std": float(np.std(brightness)),
            "mean_saturation": float(np.mean(saturation)),
            "mean_edge_density": float(np.mean(edge_density)),
            "mean_frame_difference": float(np.mean(frame_diffs)) if frame_diffs else 0.0,
        }
    finally:
        cap.release()


def _matrix(rows: Iterable[dict[str, float]]) -> np.ndarray:
    return np.asarray([[row[name] for name in FEATURE_NAMES] for row in rows], dtype=float)


def train_from_manifest(manifest_path: str, model_path: str) -> dict:
    manifest = Path(manifest_path)
    with manifest.open(newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    if not records or not {"path", "label"}.issubset(records[0]):
        raise ValueError("Manifest CSV must contain columns: path,label")

    rows, labels = [], []
    for record in records:
        video_path = Path(record["path"])
        if not video_path.is_absolute():
            video_path = manifest.parent / video_path
        rows.append(extract_features(str(video_path)))
        labels.append(record["label"].strip())
    if len(set(labels)) < 2:
        raise ValueError("Training requires at least two distinct labels")

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1500, class_weight="balanced", random_state=42
        )),
    ])
    model.fit(_matrix(rows), labels)
    target = Path(model_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES}, target)
    return {"videos": len(rows), "classes": sorted(set(labels)), "model_path": str(target)}


def predict_video(model_path: str, video_path: str) -> dict:
    artifact = joblib.load(model_path)
    features = extract_features(video_path)
    model = artifact["model"]
    probabilities = model.predict_proba(_matrix([features]))[0]
    idx = int(probabilities.argmax())
    label = str(model.classes_[idx])
    strategies = {
        "podcast": "speech_or_speaker_turns",
        "meeting": "speech_and_screen_scene_boundaries",
        "sports": "event_based_temporal_chunks",
        "gameplay": "action_changes_plus_temporal_windows",
        "wildlife": "scene_and_activity_segments",
        "egocentric": "activity_transition_chunks",
        "surveillance": "time_windows_plus_event_boundaries",
        "concert": "song_or_performance_segments",
    }
    return {
        "video_type": label,
        "confidence": float(probabilities[idx]),
        "recommended_chunking": strategies.get(label, "scene_based_fallback"),
        "features": features,
    }
