#!/usr/bin/env python3
"""Sample SWE-bench Verified into a CodeBench task file.

Loads SWE-bench/SWE-bench_Verified from Hugging Face and writes a small
JSONL sample (default 5 tasks) to data/swebench_verified_sample.jsonl.

Usage:
    python scripts/prepare_swebench_sample.py
    python scripts/prepare_swebench_sample.py --limit 30 --difficulty easy
    python scripts/prepare_swebench_sample.py --limit 50 --output data/my_sample.jsonl

    # repo-stratified, seeded sample that skips an earlier pilot
    python scripts/prepare_swebench_sample.py --limit 30 --stratify-repo --seed 0 \
        --max-per-repo 4 --exclude-ids-from data/swebench_verified_sample.jsonl \
        --output data/swebench_aes30.jsonl
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from codebench.swebench_adapter import write_swebench_sample

DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(__file__), "..", "data", "swebench_verified_sample.jsonl"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of tasks to sample (default: 5; try 30 or 50 for larger runs)",
    )
    parser.add_argument(
        "--difficulty",
        default=None,
        help="Filter by difficulty: easy/medium/hard or a raw SWE-bench label",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output JSONL path (default: data/swebench_verified_sample.jsonl)",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split (default: test)",
    )
    parser.add_argument(
        "--stratify-repo", action="store_true",
        help="Draw a seeded sample round-robin across repos instead of the first N rows",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for --stratify-repo")
    parser.add_argument(
        "--max-per-repo", type=int, default=None,
        help="Cap tasks per repo (with --stratify-repo)",
    )
    parser.add_argument(
        "--exclude-ids-from", default=None,
        help="Sample JSONL whose task ids are excluded (e.g. the 5-task pilot)",
    )
    parser.add_argument(
        "--exclude-ids", nargs="*", default=None,
        help="Instance ids to exclude (with --stratify-repo)",
    )
    args = parser.parse_args()

    exclude = list(args.exclude_ids or [])
    if args.exclude_ids_from:
        from codebench.swebench_adapter import read_swebench_sample
        exclude += [r["task"]["task_id"] for r in read_swebench_sample(args.exclude_ids_from)]
    exclude = exclude or None
    records = write_swebench_sample(
        output_path=args.output,
        limit=args.limit,
        difficulty=args.difficulty,
        split=args.split,
        stratify_repo=args.stratify_repo,
        seed=args.seed,
        max_per_repo=args.max_per_repo,
        exclude_ids=exclude,
    )
    print(f"Wrote {len(records)} SWE-bench Verified tasks to {os.path.abspath(args.output)}")
    for record in records:
        task = record["task"]
        print(f"  - {task['task_id']} ({task['difficulty']}) {task['repo']}")


if __name__ == "__main__":
    main()
