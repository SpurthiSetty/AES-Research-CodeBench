#!/usr/bin/env python3
"""Run real agent attempts against a SWE-bench Verified sample.

For each task x attempt: checks out the target repo at base_commit in an
isolated git worktree, runs the agent command there, captures git diff as
model_patch, and writes one official predictions JSONL per attempt index.

The agent only ever sees the issue description, repo name, base commit,
difficulty, and tags — never the gold patch or hidden tests.

Usage:
    # dry-run the pipeline (no real agent, empty patches)
    python scripts/run_swebench_agent.py --agent-cmd noop --run-name dryrun

    # 3 attempts per task with Claude Code
    python scripts/run_swebench_agent.py \\
        --model-name anote-code --attempts 3 --run-name v1 \\
        --agent-cmd 'claude -p "$(cat {prompt_file})" --permission-mode acceptEdits'
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from codebench.swebench_runner import (
    DEFAULT_GIT_BASE_URL,
    DEFAULT_TIMEOUT_S,
    run_swebench_attempts,
)

ROOT = os.path.join(os.path.dirname(__file__), "..")
DEFAULT_SAMPLE = os.path.join(ROOT, "data", "swebench_verified_sample.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample",
        default=DEFAULT_SAMPLE,
        help="Task sample JSONL from prepare_swebench_sample.py",
    )
    parser.add_argument(
        "--agent-cmd",
        default="noop",
        help=(
            "Shell command template with {prompt_file}/{workdir}/{instance_id}/{repo} "
            "placeholders, or 'noop' for a dry run (default: noop)"
        ),
    )
    parser.add_argument("--model-name", default="anote-code", help="model_name_or_path value")
    parser.add_argument("--attempts", type=int, default=1, help="Attempts per task (default: 1)")
    parser.add_argument("--run-name", default="run", help="Name for this run's artifacts")
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT_S,
        help=f"Per-attempt agent timeout in seconds (default: {DEFAULT_TIMEOUT_S})",
    )
    parser.add_argument(
        "--workspaces-dir", default=os.path.join(ROOT, "workspaces"),
        help="Where repo clones and worktrees live",
    )
    parser.add_argument(
        "--attempts-dir", default=os.path.join(ROOT, "attempts"),
        help="Where per-attempt records (prompt/log/patch/meta) are written",
    )
    parser.add_argument(
        "--predictions-dir", default=os.path.join(ROOT, "predictions"),
        help="Where prediction JSONL files are written",
    )
    parser.add_argument(
        "--git-base-url", default=DEFAULT_GIT_BASE_URL,
        help="Base URL for cloning task repos (default: https://github.com)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip (task, attempt) pairs that already have a recorded meta.json",
    )
    parser.add_argument(
        "--keep-workspaces", action="store_true",
        help="Keep per-attempt worktrees for debugging",
    )
    parser.add_argument(
        "--shard", default=None,
        help="Run one shard, e.g. 2/4. Shards split whole repos, so they can run in "
             "parallel terminals. Afterwards rerun once without --shard and with "
             "--resume to write merged prediction files.",
    )
    parser.add_argument(
        "--merge-only", action="store_true",
        help="Write merged prediction files from recorded attempts without running the agent; "
             "fails if any attempt is missing",
    )
    args = parser.parse_args()

    shard = num_shards = None
    if args.shard:
        shard, num_shards = (int(x) for x in args.shard.split("/"))

    prediction_files = run_swebench_attempts(
        sample_path=args.sample,
        agent_cmd=args.agent_cmd,
        model_name=args.model_name,
        attempts=args.attempts,
        run_name=args.run_name,
        workspaces_dir=args.workspaces_dir,
        attempts_dir=args.attempts_dir,
        predictions_dir=args.predictions_dir,
        timeout_s=args.timeout,
        resume=args.resume,
        keep_workspaces=args.keep_workspaces,
        git_base_url=args.git_base_url,
        shard=shard,
        num_shards=num_shards,
        merge_only=args.merge_only,
    )
    print(f"Run '{args.run_name}' complete: {len(prediction_files)} prediction file(s)")
    for attempt, path in sorted(prediction_files.items()):
        print(f"  attempt {attempt}: {os.path.abspath(path)}")
    print(
        "\nNext: evaluate each file with the official harness, e.g.\n"
        "  python -m swebench.harness.run_evaluation "
        "--dataset_name SWE-bench/SWE-bench_Verified \\\n"
        f"      --predictions_path {os.path.abspath(prediction_files[1])} \\\n"
        f"      --max_workers 2 --run_id {args.run_name}-attempt1"
    )


if __name__ == "__main__":
    main()
