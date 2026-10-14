"""Simple timestamp-based fallback chunking strategy."""
from __future__ import annotations


def create_chunks(duration_seconds: float, chunk_seconds: float = 30.0,
                  overlap_seconds: float = 5.0) -> list[dict]:
    if duration_seconds < 0:
        raise ValueError("duration_seconds cannot be negative")
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")
    if overlap_seconds < 0 or overlap_seconds >= chunk_seconds:
        raise ValueError("overlap_seconds must be >= 0 and smaller than chunk_seconds")
    chunks, start, step = [], 0.0, chunk_seconds - overlap_seconds
    while start < duration_seconds:
        end = min(start + chunk_seconds, duration_seconds)
        chunks.append({"start": round(start, 3), "end": round(end, 3)})
        start += step
    return chunks
