#!/usr/bin/env python3
"""
Merge Romanian NLP datasets into a unified corpus for RO-ASI benchmark.

Datasets:
- LaRoSeDa: Product reviews (sentiment)
- PoPreRo: News popularity prediction
- RED v1: Tweets with single-label emotions
- RED v2: Tweets with multi-label emotions
- RoSent: Sentiment analysis (reviews)

Output: merged_corpus.jsonl with unified schema
"""

import json
import csv
from pathlib import Path
from typing import Dict, Generator, Any
import hashlib


# Base path for datasets
BASE_PATH = Path(__file__).parent.parent.parent / "small_datasets"


def generate_id(source: str, original_id: str) -> str:
    """Generate unique ID combining source and original ID."""
    return f"{source}_{original_id}"


def process_laroseda(base_path: Path) -> Generator[Dict[str, Any], None, None]:
    """
    Process LaRoSeDa dataset (product reviews).

    Fields: index, title, content, starRating
    """
    source = "laroseda"

    for split in ["train", "test"]:
        file_path = base_path / "LaRoSeDa" / f"laroseda_{split}.json"
        if not file_path.exists():
            print(f"Warning: {file_path} not found")
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        reviews = data.get("reviews", [])
        for review in reviews:
            # Combine title and content for full text
            title = review.get("title", "").strip()
            content = review.get("content", "").strip()

            if title and content:
                text = f"{title}\n{content}"
            else:
                text = title or content

            if not text:
                continue

            yield {
                "id": generate_id(source, str(review.get("index", ""))),
                "text": text,
                "source": source,
                "split": split,
                "original_labels": {
                    "star_rating": review.get("starRating"),
                    "title": title,
                }
            }


def process_poprero(base_path: Path) -> Generator[Dict[str, Any], None, None]:
    """
    Process PoPreRo dataset (news articles).

    Fields: index, full_text, label
    """
    source = "poprero"

    for split in ["train", "validation", "test"]:
        file_path = base_path / "PoPreRo" / "Dataset" / f"{split}.csv"
        if not file_path.exists():
            print(f"Warning: {file_path} not found")
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get("full_text", "").strip()
                if not text:
                    continue

                yield {
                    "id": generate_id(source, str(row.get("index", ""))),
                    "text": text,
                    "source": source,
                    "split": split,
                    "original_labels": {
                        "popularity_label": row.get("label"),
                    }
                }


def process_red_v1(base_path: Path) -> Generator[Dict[str, Any], None, None]:
    """
    Process RED v1 dataset (tweets with single emotion label).

    Fields: Tweet, Emotion
    Emotions: Neutru, Bucurie, Frica, Furie, Tristete
    """
    source = "red_v1"

    for split in ["train", "val", "test"]:
        file_path = base_path / "RED" / "REDv1" / "data" / f"{split}.csv"
        if not file_path.exists():
            print(f"Warning: {file_path} not found")
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                text = row.get("Tweet", "").strip()
                if not text:
                    continue

                # Use row index as ID since CSV may have unnamed index columns
                yield {
                    "id": generate_id(source, f"{split}_{idx}"),
                    "text": text,
                    "source": source,
                    "split": "validation" if split == "val" else split,
                    "original_labels": {
                        "emotion": row.get("Emotion"),
                    }
                }


def process_red_v2(base_path: Path) -> Generator[Dict[str, Any], None, None]:
    """
    Process RED v2 dataset (tweets with multi-label emotions).

    Fields: text, text_id, agreed_labels, sum_labels, procentual_labels
    Emotions order: ['Tristețe', 'Surpriză', 'Frică', 'Furie', 'Neutru', 'Încredere', 'Bucurie']
    """
    source = "red_v2"
    emotion_names = ['tristete', 'surpriza', 'frica', 'furie', 'neutru', 'incredere', 'bucurie']

    for split in ["train", "valid", "test"]:
        file_path = base_path / "RED" / "REDv2" / "data" / f"{split}.json"
        if not file_path.exists():
            print(f"Warning: {file_path} not found")
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            text = item.get("text", "").strip()
            if not text:
                continue

            # Convert agreed_labels to emotion names
            agreed_labels = item.get("agreed_labels", [])
            emotions = [emotion_names[i] for i, v in enumerate(agreed_labels) if v == 1]

            yield {
                "id": generate_id(source, str(item.get("text_id", ""))),
                "text": text,
                "source": source,
                "split": "validation" if split == "valid" else split,
                "original_labels": {
                    "emotions": emotions,
                    "agreed_labels": agreed_labels,
                    "procentual_labels": item.get("procentual_labels"),
                }
            }


def process_rosent(base_path: Path) -> Generator[Dict[str, Any], None, None]:
    """
    Process RoSent dataset (sentiment analysis).

    Fields: index, text, label
    Labels: 0 (negative), 1 (positive)
    """
    source = "rosent"

    for split in ["train", "test"]:
        file_path = base_path / "RoSent" / f"{split}.csv"
        if not file_path.exists():
            print(f"Warning: {file_path} not found")
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get("text", "").strip()
                if not text:
                    continue

                yield {
                    "id": generate_id(source, str(row.get("index", ""))),
                    "text": text,
                    "source": source,
                    "split": split,
                    "original_labels": {
                        "sentiment": int(row.get("label", 0)),
                    }
                }


def merge_all_datasets(base_path: Path, output_path: Path) -> Dict[str, int]:
    """
    Merge all datasets into a single JSONL file.

    Returns statistics about the merged corpus.
    """
    stats = {
        "total": 0,
        "by_source": {},
        "unique_texts": 0,
        "duplicates_removed": 0,
    }

    # Track unique texts by hash to avoid exact duplicates
    seen_hashes = set()

    processors = [
        ("laroseda", process_laroseda),
        ("poprero", process_poprero),
        ("red_v1", process_red_v1),
        ("red_v2", process_red_v2),
        ("rosent", process_rosent),
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as out_f:
        for source_name, processor in processors:
            source_count = 0

            for record in processor(base_path):
                # Check for duplicates using text hash
                text_hash = hashlib.md5(record["text"].encode()).hexdigest()

                if text_hash in seen_hashes:
                    stats["duplicates_removed"] += 1
                    continue

                seen_hashes.add(text_hash)

                # Write to JSONL
                out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
                source_count += 1
                stats["total"] += 1

            stats["by_source"][source_name] = source_count
            print(f"Processed {source_name}: {source_count} records")

    stats["unique_texts"] = len(seen_hashes)

    return stats


def load_merged_corpus(corpus_path: Path) -> Generator[Dict[str, Any], None, None]:
    """Load merged corpus from JSONL file."""
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


if __name__ == "__main__":
    base_path = BASE_PATH
    output_path = Path(__file__).parent.parent.parent / "data" / "merged_corpus.jsonl"

    print(f"Base path: {base_path}")
    print(f"Output path: {output_path}")
    print()

    stats = merge_all_datasets(base_path, output_path)

    print()
    print("=" * 50)
    print("Merge Statistics:")
    print(f"  Total records: {stats['total']}")
    print(f"  Unique texts: {stats['unique_texts']}")
    print(f"  Duplicates removed: {stats['duplicates_removed']}")
    print()
    print("By source:")
    for source, count in stats["by_source"].items():
        print(f"  {source}: {count}")
    print()
    print(f"Output saved to: {output_path}")
