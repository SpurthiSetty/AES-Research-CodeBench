"""Runner for real SWE-bench Verified agent attempts.

Orchestrates: load agent-safe tasks -> check out each target repo at
``base_commit`` -> run a coding agent in the checkout -> capture ``git diff``
as ``model_patch`` -> export official SWE-bench prediction JSONL files
(one per attempt index, since the harness allows one prediction per
instance_id per file).

Leakage policy
--------------
:func:`load_agent_safe_tasks` is the single choke point that reads sample
files written by :func:`codebench.swebench_adapter.write_swebench_sample`.
It keeps only agent-safe fields; ``hidden_reference`` (gold patch,
test_patch, FAIL_TO_PASS, PASS_TO_PASS) is dropped at parse time and never
reaches any prompt, log, or attempt record. The agent works in a clean
checkout of the *target* repo — no benchmark files are present there.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from .swebench_adapter import write_predictions_jsonl

#: Sentinel value for --agent-cmd: run no agent, produce an empty patch.
NOOP_AGENT = "noop"

DEFAULT_GIT_BASE_URL = "https://github.com"
DEFAULT_TIMEOUT_S = 1800


class AgentSafeTask(BaseModel):
    """The only view of a SWE-bench task the runner ever holds.

    Constructed exclusively by :func:`load_agent_safe_tasks`, which drops
    gold fields at parse time.
    """

    task_id: str
    repo: str
    base_commit: str
    description: str
    difficulty: str
    tags: List[str] = Field(default_factory=list)


class AttemptRecord(BaseModel):
    """Outcome of one agent attempt on one task."""

    task_id: str
    attempt: int
    model_name: str
    exit_code: Optional[int]
    timed_out: bool
    latency_ms: float
    model_patch: str
    record_dir: str


def load_agent_safe_tasks(sample_path: Union[str, Path]) -> List[AgentSafeTask]:
    """Load tasks from a sample JSONL, keeping only agent-safe fields.

    Gold data (``hidden_reference``) and infra-only fields are discarded
    here and never stored on any runner object.
    """
    tasks: List[AgentSafeTask] = []
    with Path(sample_path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            task = record["task"]
            tasks.append(
                AgentSafeTask(
                    task_id=task["task_id"],
                    repo=task["repo"],
                    base_commit=record["base_commit"],
                    description=task["description"],
                    difficulty=task["difficulty"],
                    tags=list(task.get("tags", [])),
                )
            )
    return tasks


def build_agent_prompt(task: AgentSafeTask) -> str:
    """Build the agent-facing prompt from agent-safe fields only."""
    return (
        f"You are working in a checkout of the repository `{task.repo}` "
        f"at commit `{task.base_commit}` (the current working directory).\n\n"
        f"Resolve the following GitHub issue by modifying the source code. "
        f"Focus on the root cause rather than editing tests. "
        f"Do not create git commits; leave your changes in the working tree.\n\n"
        f"--- ISSUE ({task.task_id}, difficulty: {task.difficulty}) ---\n\n"
        f"{task.description}\n"
    )


def _git(args: List[str], cwd: Union[str, Path]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed in {cwd}: {result.stderr.strip()}"
        )
    return result.stdout


def ensure_repo_clone(
    repo: str,
    workspaces_dir: Union[str, Path],
    git_base_url: str = DEFAULT_GIT_BASE_URL,
) -> Path:
    """Clone ``repo`` once into the workspace cache, or fetch if present."""
    clone_path = Path(workspaces_dir) / "repos" / repo.replace("/", "__")
    if clone_path.exists():
        _git(["fetch", "--all", "--quiet"], cwd=clone_path)
    else:
        clone_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{git_base_url.rstrip('/')}/{repo}"
        result = subprocess.run(
            ["git", "clone", "--quiet", url, str(clone_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"git clone {url} failed: {result.stderr.strip()}")
    return clone_path


def create_worktree(
    clone_path: Union[str, Path],
    base_commit: str,
    worktree_path: Union[str, Path],
) -> Path:
    """Create a detached worktree of the cached clone at ``base_commit``."""
    worktree_path = Path(worktree_path)
    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    _git(
        ["worktree", "add", "--detach", str(worktree_path), base_commit],
        cwd=clone_path,
    )
    return worktree_path


def remove_worktree(clone_path: Union[str, Path], worktree_path: Union[str, Path]) -> None:
    """Remove a worktree created by :func:`create_worktree`."""
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(worktree_path)],
        cwd=str(clone_path),
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "worktree", "prune"],
        cwd=str(clone_path),
        capture_output=True,
        text=True,
    )


def capture_model_patch(worktree_path: Union[str, Path], base_commit: str) -> str:
    """Capture the agent's work as a unified diff against ``base_commit``.

    ``git add -A`` stages new untracked files; diffing the index against the
    explicit base commit is robust even if the agent created commits.
    """
    _git(["add", "-A"], cwd=worktree_path)
    patch = _git(["diff", "--cached", base_commit], cwd=worktree_path)
    for line in patch.splitlines():
        if line.startswith("diff --git"):
            parts = line.split()
            paths = [p[2:] for p in parts[2:4] if len(p) > 2]
            if any(p.startswith(("/", "..")) or p.startswith(".git/") for p in paths):
                raise RuntimeError(f"Refusing patch touching unsafe path: {line}")
    return patch


def run_agent_command(
    agent_cmd: str,
    prompt_file: Union[str, Path],
    workdir: Union[str, Path],
    task: AgentSafeTask,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    log_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Run one agent attempt and return exit metadata.

    ``agent_cmd`` is a shell template with ``{prompt_file}``, ``{workdir}``,
    ``{instance_id}``, and ``{repo}`` placeholders. The sentinel ``"noop"``
    runs nothing (empty patch), for dry runs and tests.
    """
    if agent_cmd.strip() == NOOP_AGENT:
        return {"exit_code": 0, "timed_out": False, "latency_ms": 0.0}

    command = agent_cmd.format(
        prompt_file=shlex.quote(str(prompt_file)),
        workdir=shlex.quote(str(workdir)),
        instance_id=shlex.quote(task.task_id),
        repo=shlex.quote(task.repo),
    )
    start = time.monotonic()
    timed_out = False
    exit_code: Optional[int] = None
    stdout = stderr = ""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        exit_code = result.returncode
        stdout, stderr = result.stdout, result.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = exc.stdout or "" if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr or "" if isinstance(exc.stderr, str) else ""
    latency_ms = (time.monotonic() - start) * 1000.0

    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            f"$ {command}\n\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}\n",
            encoding="utf-8",
        )
    return {"exit_code": exit_code, "timed_out": timed_out, "latency_ms": latency_ms}


def _run_single_attempt(
    task: AgentSafeTask,
    attempt: int,
    agent_cmd: str,
    model_name: str,
    record_dir: Path,
    clone_path: Path,
    worktree_path: Path,
    timeout_s: int,
    keep_workspaces: bool,
) -> AttemptRecord:
    record_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = record_dir / "prompt.md"
    prompt_file.write_text(build_agent_prompt(task), encoding="utf-8")

    create_worktree(clone_path, task.base_commit, worktree_path)
    try:
        outcome = run_agent_command(
            agent_cmd,
            prompt_file=prompt_file,
            workdir=worktree_path,
            task=task,
            timeout_s=timeout_s,
            log_path=record_dir / "agent.log",
        )
        if outcome["timed_out"] or (outcome["exit_code"] not in (0, None)):
            model_patch = ""  # failed attempt still counts as a rollout
        else:
            model_patch = capture_model_patch(worktree_path, task.base_commit)
    finally:
        if not keep_workspaces:
            remove_worktree(clone_path, worktree_path)

    record = AttemptRecord(
        task_id=task.task_id,
        attempt=attempt,
        model_name=model_name,
        exit_code=outcome["exit_code"],
        timed_out=outcome["timed_out"],
        latency_ms=outcome["latency_ms"],
        model_patch=model_patch,
        record_dir=str(record_dir),
    )
    (record_dir / "model.patch").write_text(model_patch, encoding="utf-8")
    (record_dir / "meta.json").write_text(
        json.dumps(record.model_dump(exclude={"model_patch"}), indent=2),
        encoding="utf-8",
    )
    return record


def shard_tasks(tasks: List[AgentSafeTask], shard: int, num_shards: int) -> List[AgentSafeTask]:
    """Return the tasks for ``shard`` (1-based) of ``num_shards``.

    Whole repos are assigned to shards (greedy, largest repo first), so two
    shards never share a repo clone or create worktrees in the same repo
    concurrently.
    """
    if num_shards < 1 or not 1 <= shard <= num_shards:
        raise ValueError(f"invalid shard {shard}/{num_shards}")
    counts: Dict[str, int] = {}
    for t in tasks:
        counts[t.repo] = counts.get(t.repo, 0) + 1
    load = [0] * num_shards
    owner: Dict[str, int] = {}
    for repo in sorted(counts, key=lambda r: (-counts[r], r)):
        i = min(range(num_shards), key=lambda j: (load[j], j))
        owner[repo] = i + 1
        load[i] += counts[repo]
    return [t for t in tasks if owner[t.repo] == shard]


def run_swebench_attempts(
    sample_path: Union[str, Path],
    agent_cmd: str,
    model_name: str = "anote-code",
    attempts: int = 1,
    run_name: str = "run",
    workspaces_dir: Union[str, Path] = "workspaces",
    attempts_dir: Union[str, Path] = "attempts",
    predictions_dir: Union[str, Path] = "predictions",
    timeout_s: int = DEFAULT_TIMEOUT_S,
    resume: bool = False,
    keep_workspaces: bool = False,
    git_base_url: str = DEFAULT_GIT_BASE_URL,
    shard: Optional[int] = None,
    num_shards: Optional[int] = None,
    merge_only: bool = False,
) -> Dict[int, Path]:
    """Run ``attempts`` independent agent attempts per sampled task.

    With ``shard``/``num_shards`` only that shard's tasks are run and the
    prediction files get a ``_s{shard}of{num_shards}`` suffix. After all
    shards finish, rerun without a shard and with ``resume=True`` to write
    the merged prediction files from the recorded attempts.

    Returns a mapping of attempt index -> written predictions JSONL path
    (one file per attempt, in the official SWE-bench prediction format).
    """
    if attempts < 1:
        raise ValueError(f"attempts must be >= 1, got {attempts}")

    tasks = load_agent_safe_tasks(sample_path)
    if not tasks:
        raise ValueError(f"No tasks found in {sample_path}")
    suffix = ""
    if shard is not None or num_shards is not None:
        if shard is None or num_shards is None:
            raise ValueError("shard and num_shards must be given together")
        tasks = shard_tasks(tasks, shard, num_shards)
        suffix = f"_s{shard}of{num_shards}"
        if not tasks:
            raise ValueError(f"Shard {shard}/{num_shards} has no tasks")

    if merge_only:
        missing = [
            f"{t.task_id} attempt {a}"
            for t in tasks
            for a in range(1, attempts + 1)
            if not (Path(attempts_dir) / run_name / t.task_id / f"attempt-{a}" / "meta.json").exists()
        ]
        if missing:
            raise RuntimeError(
                f"merge_only: {len(missing)} attempt(s) not recorded yet; finish the shards first:\n  "
                + "\n  ".join(missing)
            )
        resume = True

    records: Dict[int, List[AttemptRecord]] = {k: [] for k in range(1, attempts + 1)}
    for task in tasks:
        clone_path = None if merge_only else ensure_repo_clone(task.repo, workspaces_dir, git_base_url)
        for attempt in range(1, attempts + 1):
            record_dir = Path(attempts_dir) / run_name / task.task_id / f"attempt-{attempt}"
            meta_path = record_dir / "meta.json"
            if resume and meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                patch = (record_dir / "model.patch").read_text(encoding="utf-8")
                records[attempt].append(AttemptRecord(**meta, model_patch=patch))
                continue
            worktree_path = (
                Path(workspaces_dir) / "runs" / run_name / f"{task.task_id}__a{attempt}"
            )
            records[attempt].append(
                _run_single_attempt(
                    task=task,
                    attempt=attempt,
                    agent_cmd=agent_cmd,
                    model_name=model_name,
                    record_dir=record_dir,
                    clone_path=clone_path,
                    worktree_path=worktree_path,
                    timeout_s=timeout_s,
                    keep_workspaces=keep_workspaces,
                )
            )

    prediction_files: Dict[int, Path] = {}
    for attempt, attempt_records in records.items():
        predictions = [
            {
                "instance_id": r.task_id,
                "model_name_or_path": model_name,
                "model_patch": r.model_patch,
            }
            for r in attempt_records
        ]
        out = Path(predictions_dir) / f"{run_name}{suffix}_attempt{attempt}.jsonl"
        prediction_files[attempt] = write_predictions_jsonl(predictions, out)
    return prediction_files
