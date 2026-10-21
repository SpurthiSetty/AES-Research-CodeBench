"""Adapter for SWE-bench Verified (https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified).

Converts SWE-bench Verified instances into CodeBench :class:`CodeTask` objects
and prepares prediction files for the official SWE-bench evaluation harness.

Leakage policy
--------------
The gold ``patch``, ``test_patch``, ``FAIL_TO_PASS``, and ``PASS_TO_PASS``
fields must NEVER reach a coding agent's prompt. The :class:`CodeTask`
produced here is fully agent-safe: ``test_file`` and ``reference_solution``
hold placeholders, never gold data. Gold data is only available through
:func:`extract_hidden_reference` and is stored under the clearly-marked
``hidden_reference`` key in sample files, for sanity checks — not inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from .core import CodeTask, TaskDifficulty

SWEBENCH_DATASET_NAME = "SWE-bench/SWE-bench_Verified"

#: Gold fields that must never appear in agent-facing output.
HIDDEN_FIELDS = ("patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS")

#: Instance fields the agent is allowed to see.
AGENT_VISIBLE_FIELDS = ("instance_id", "repo", "base_commit", "problem_statement")

#: Placeholder values — CodeTask requires these fields, but for SWE-bench the
#: real tests and gold patch live in the hidden reference, not on the task.
TEST_FILE_PLACEHOLDER = "swebench://hidden-tests (evaluated by the SWE-bench harness)"
REFERENCE_SOLUTION_PLACEHOLDER = (
    "swebench://hidden-gold-patch (reference-only; stored separately, never for inference)"
)

#: SWE-bench Verified annotates difficulty as estimated fix time.
SWEBENCH_DIFFICULTY_MAP: Dict[str, TaskDifficulty] = {
    "<15 min fix": TaskDifficulty.EASY,
    "15 min - 1 hour": TaskDifficulty.MEDIUM,
    "1-4 hours": TaskDifficulty.HARD,
    ">4 hours": TaskDifficulty.HARD,
}

#: Required keys for the official SWE-bench prediction format.
PREDICTION_KEYS = ("instance_id", "model_name_or_path", "model_patch")


def map_swebench_difficulty(raw: Optional[str]) -> TaskDifficulty:
    """Map a SWE-bench Verified difficulty label to :class:`TaskDifficulty`.

    Unknown or missing labels default to MEDIUM.
    """
    if raw is None:
        return TaskDifficulty.MEDIUM
    label = str(raw).strip()
    if label in SWEBENCH_DIFFICULTY_MAP:
        return SWEBENCH_DIFFICULTY_MAP[label]
    try:
        return TaskDifficulty(label.lower())
    except ValueError:
        return TaskDifficulty.MEDIUM


def _load_raw_dataset(split: str) -> Iterable[Dict[str, Any]]:
    """Load the raw SWE-bench Verified split from Hugging Face."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised only without extras
        raise ImportError(
            "The 'datasets' package is required to load SWE-bench Verified. "
            "Install it with: pip install datasets"
        ) from exc
    return load_dataset(SWEBENCH_DATASET_NAME, split=split)


def load_swebench_verified(
    split: str = "test",
    limit: Optional[int] = None,
    difficulty: Optional[Union[str, TaskDifficulty]] = None,
) -> List[Dict[str, Any]]:
    """Load SWE-bench Verified instances as plain dicts.

    Args:
        split: Dataset split (SWE-bench Verified only publishes "test").
        limit: Keep at most this many instances (applied after filtering).
        difficulty: Filter by difficulty — accepts a CodeBench level
            ("easy"/"medium"/"hard") or a raw SWE-bench label ("<15 min fix", ...).

    Returns:
        Raw instance dicts, including gold fields. Callers preparing
        agent-facing data must go through :func:`swebench_instance_to_codetask`
        or :func:`agent_prompt_payload`, never hand these dicts to an agent.
    """
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")

    wanted: Optional[TaskDifficulty] = None
    if difficulty is not None:
        wanted = (
            difficulty
            if isinstance(difficulty, TaskDifficulty)
            else map_swebench_difficulty(str(difficulty))
        )

    instances: List[Dict[str, Any]] = []
    for row in _load_raw_dataset(split):
        instance = dict(row)
        if wanted is not None and map_swebench_difficulty(instance.get("difficulty")) != wanted:
            continue
        instances.append(instance)
        if limit is not None and len(instances) >= limit:
            break
    return instances


def swebench_instance_to_codetask(instance: Dict[str, Any]) -> CodeTask:
    """Convert one SWE-bench Verified instance into an agent-safe CodeTask.

    The returned task never contains gold data: ``test_file`` and
    ``reference_solution`` are placeholders, so the task can be serialized
    into prompts without leaking the solution.
    """
    difficulty = map_swebench_difficulty(instance.get("difficulty"))
    tags = ["swebench", instance["repo"], difficulty.value]
    if instance.get("base_commit"):
        tags.append(f"base_commit:{instance['base_commit']}")
    return CodeTask(
        task_id=instance["instance_id"],
        repo=instance["repo"],
        description=instance["problem_statement"],
        difficulty=difficulty,
        test_file=TEST_FILE_PLACEHOLDER,
        reference_solution=REFERENCE_SOLUTION_PLACEHOLDER,
        tags=tags,
    )


def extract_hidden_reference(instance: Dict[str, Any]) -> Dict[str, Any]:
    """Extract gold fields for internal sanity checks. NEVER show to agents."""
    hidden = {field: instance.get(field) for field in HIDDEN_FIELDS}
    hidden["_note"] = (
        "Reference-only gold data for sanity checks and harness evaluation. "
        "Must never be included in agent prompts."
    )
    return hidden


def agent_prompt_payload(instance: Dict[str, Any]) -> Dict[str, Any]:
    """Build the agent-facing view of an instance: only allowed fields.

    Includes the issue description, repo name, base commit, and safe metadata.
    """
    task = swebench_instance_to_codetask(instance)
    return {
        "task_id": task.task_id,
        "repo": task.repo,
        "base_commit": instance.get("base_commit"),
        "description": task.description,
        "difficulty": task.difficulty.value,
        "tags": task.tags,
    }


def stratified_sample(
    instances: List[Dict[str, Any]],
    n: int,
    seed: int = 0,
    max_per_repo: Optional[int] = None,
    exclude_ids: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """Seeded, repo-stratified sample of ``n`` instances.

    Instances are shuffled within each repo (seeded), then drawn round-robin
    across repos in sorted order, so every repo contributes before any repo
    contributes twice. ``max_per_repo`` caps any single repo; ``exclude_ids``
    removes instances (e.g. an earlier pilot) before sampling.
    """
    import random

    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    excluded = set(exclude_ids or ())
    by_repo: Dict[str, List[Dict[str, Any]]] = {}
    for inst in instances:
        if inst["instance_id"] in excluded:
            continue
        by_repo.setdefault(inst["repo"], []).append(inst)
    rng = random.Random(seed)
    for repo in sorted(by_repo):
        by_repo[repo].sort(key=lambda x: x["instance_id"])
        rng.shuffle(by_repo[repo])
    chosen: List[Dict[str, Any]] = []
    depth = 0
    while len(chosen) < n:
        added = False
        for repo in sorted(by_repo):
            if max_per_repo is not None and depth >= max_per_repo:
                continue
            if depth < len(by_repo[repo]):
                chosen.append(by_repo[repo][depth])
                added = True
                if len(chosen) >= n:
                    break
        if not added:
            break
        depth += 1
    return chosen


def write_swebench_sample(
    output_path: Union[str, Path],
    limit: int = 5,
    difficulty: Optional[Union[str, TaskDifficulty]] = None,
    split: str = "test",
    stratify_repo: bool = False,
    seed: int = 0,
    max_per_repo: Optional[int] = None,
    exclude_ids: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """Sample SWE-bench Verified and write a JSONL task file.

    Each line holds an agent-safe ``task`` (CodeTask fields), the
    ``base_commit`` / ``environment_setup_commit`` needed to check out the
    repo, and a clearly-marked ``hidden_reference`` block with gold data for
    internal sanity checks only.

    With ``stratify_repo=True`` the sample is drawn with
    :func:`stratified_sample` instead of taking the first ``limit`` rows.

    Returns the written records.
    """
    if stratify_repo:
        pool = load_swebench_verified(split=split, limit=None, difficulty=difficulty)
        instances = stratified_sample(
            pool, limit, seed=seed, max_per_repo=max_per_repo, exclude_ids=exclude_ids
        )
    else:
        instances = load_swebench_verified(split=split, limit=limit, difficulty=difficulty)
        if exclude_ids:
            raise ValueError("exclude_ids requires stratify_repo=True")
    records: List[Dict[str, Any]] = []
    for instance in instances:
        task = swebench_instance_to_codetask(instance)
        records.append(
            {
                "task": task.model_dump(mode="json"),
                "base_commit": instance.get("base_commit"),
                "environment_setup_commit": instance.get("environment_setup_commit"),
                "hidden_reference": extract_hidden_reference(instance),
            }
        )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


def read_swebench_sample(sample_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read a JSONL sample file written by :func:`write_swebench_sample`."""
    records: List[Dict[str, Any]] = []
    with Path(sample_path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_predictions_jsonl(
    predictions: Iterable[Dict[str, Any]],
    output_path: Union[str, Path],
) -> Path:
    """Write predictions in the official SWE-bench harness JSONL format.

    Each prediction must provide ``instance_id``, ``model_name_or_path``,
    and ``model_patch``. Extra keys are dropped so the output stays
    harness-compatible.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for prediction in predictions:
            missing = [key for key in PREDICTION_KEYS if key not in prediction]
            if missing:
                raise ValueError(
                    f"Prediction for {prediction.get('instance_id', '<unknown>')!r} "
                    f"is missing required keys: {missing}"
                )
            row = {key: prediction[key] for key in PREDICTION_KEYS}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path
