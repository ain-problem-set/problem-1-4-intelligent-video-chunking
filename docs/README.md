# Intelligent Video Chunking — design notes

## Goal
Build two small routing components: (1) classify an incoming video into a category useful for chunking, and (2) classify a natural-language query as whole-video, frame-based, or section/clip-based.

## Proposed pipeline
1. Read cheap metadata (duration, FPS, resolution).
2. Sample a small number of frames uniformly. Resize each to 160×90 before computing brightness, saturation, edge density, and frame-to-frame difference.
3. Train a small Logistic Regression classifier on labeled video examples. This is a low-cost baseline; the features are deliberately inexpensive but cannot reliably distinguish all semantic classes by themselves.
4. Use a TF-IDF + Logistic Regression query classifier as a low-latency baseline.
5. Route by intent and video type. Use temporal windows as a fallback; add transcript boundaries for speech-heavy videos and event/scene boundaries when those signals are available.

## Feature engineering
Video features include duration, frame rate, dimensions, aspect ratio, sampled-frame brightness and saturation, edge density, and average adjacent-frame difference. These capture basic visual and temporal characteristics, not deep semantic meaning. A production-quality version should compare a compact pretrained visual embedding model against this baseline, and optionally add speech/activity features for podcasts and meetings.

Query features are word and bigram TF-IDF features. This is compact and interpretable, but its quality depends on labeled examples and may be brittle for paraphrases. Build a diverse, human-labeled query set and include ambiguous queries and out-of-distribution examples.

## Evaluation plan
- Classification: macro-F1, per-class precision/recall, confusion matrix.
- Runtime: median and p95 latency, throughput, model artifact size, and memory.
- Retrieval routing: routing accuracy and, once an indexed corpus exists, Recall@K / MRR.
- Use video-level splits to prevent near-duplicate frames from leaking across train and test.
- Report results by class and dataset slice, not just a single aggregate score.

## VideoMME note
The repository instructions require benchmarking on the 10% VideoMME sample. This starter ZIP does not include VideoMME or category labels for the proposed video taxonomy. Do not claim benchmark results until the official sample is obtained and the evaluation protocol is implemented. VideoMME question-answering accuracy alone does not directly measure custom video-type classification, so document the relationship between the task labels, query types, and benchmark.

## Deployment
Package the feature extractor and trained model artifact together. Run inference as a stateless service or worker, cache extracted features by video checksum, and process videos asynchronously. Bound frame sampling to control cost; use confidence thresholds and an `other` fallback for uncertain predictions. Track latency, errors, class distribution drift, and model version.

## Limitations and next steps
The included query examples are a smoke-test baseline, not a representative training corpus. The video classifier must be trained from a labeled `path,label` CSV before use. Next: create documented training/evaluation commands, add a diverse labeled dataset, compare compact pretrained embeddings, and measure end-to-end performance on the required benchmark.
