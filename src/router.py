"""Maps predicted query intent and video type to retrieval strategy."""
from __future__ import annotations


def choose_retrieval_strategy(intent: str, video_type: str = "other") -> dict:
    if intent == "frame":
        strategy = "timestamp_or_frame_search"
    elif intent == "whole_video":
        strategy = "global_video_summary_or_aggregate_search"
    elif intent == "clip":
        strategy = {
            "podcast": "transcript_semantic_search",
            "meeting": "transcript_and_screen_search",
            "sports": "event_and_temporal_search",
            "gameplay": "action_and_temporal_search",
            "wildlife": "activity_and_temporal_search",
            "egocentric": "activity_search",
            "surveillance": "event_and_time_window_search",
        }.get(video_type, "semantic_chunk_search")
    else:
        strategy = "semantic_chunk_search"
    return {"intent": intent, "video_type": video_type, "strategy": strategy}
