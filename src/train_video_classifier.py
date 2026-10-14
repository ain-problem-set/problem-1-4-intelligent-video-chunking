from __future__ import annotations
import argparse
from video_classifier import train_from_manifest

def main():
    parser = argparse.ArgumentParser(description="Train video classifier from a path,label CSV.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    print(train_from_manifest(args.manifest, args.model))

if __name__ == "__main__":
    main()
