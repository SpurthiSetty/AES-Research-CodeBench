"""Offline tests for stratified sampling, sharding, and acceptance-event analysis."""

import importlib.util
import json
import os
from pathlib import Path

import pytest

from codebench.swebench_adapter import stratified_sample
from codebench.swebench_runner import AgentSafeTask, shard_tasks

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "analyze_swebench_acceptance", ROOT / "scripts" / "analyze_swebench_acceptance.py"
)
asa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(asa)


def inst(repo, i):
    return {"instance_id": f"{repo.split('/')[1]}-{i}", "repo": repo}


POOL = [inst("a/a", i) for i in range(10)] + [inst("b/b", i) for i in range(3)] + [inst("c/c", 0)]


def test_stratified_sample_covers_repos_first():
    s = stratified_sample(POOL, 3, seed=0)
    assert sorted(x["repo"] for x in s) == ["a/a", "b/b", "c/c"]


def test_stratified_sample_is_seeded_and_respects_caps_and_exclusions():
    assert stratified_sample(POOL, 6, seed=1) == stratified_sample(POOL, 6, seed=1)
    capped = stratified_sample(POOL, 20, seed=0, max_per_repo=2)
    assert len(capped) == 5  # 2 + 2 + 1
    excl = stratified_sample(POOL, 20, seed=0, exclude_ids=["c-0"])
    assert "c-0" not in {x["instance_id"] for x in excl}
    assert len(excl) == 13


def test_stratified_sample_rejects_bad_n():
    with pytest.raises(ValueError):
        stratified_sample(POOL, 0)


def task(repo, i):
    return AgentSafeTask(task_id=f"{repo}-{i}", repo=repo, base_commit="abc",
                         description="d", difficulty="medium")


def test_shard_tasks_partitions_by_whole_repo():
    tasks = [task("x/big", i) for i in range(4)] + [task("y/mid", i) for i in range(2)] + [task("z/s", 0)]
    shards = [shard_tasks(tasks, s, 2) for s in (1, 2)]
    assert sorted(t.task_id for sh in shards for t in sh) == sorted(t.task_id for t in tasks)
    for repo in {"x/big", "y/mid", "z/s"}:
        owners = [i for i, sh in enumerate(shards) if any(t.repo == repo for t in sh)]
        assert len(owners) == 1
    with pytest.raises(ValueError):
        shard_tasks(tasks, 3, 2)


def write_report(logs, run_id, model, iid, f2p, p2p, resolved, applied=True):
    d = logs / run_id / model / iid
    d.mkdir(parents=True)
    (d / "report.json").write_text(json.dumps({iid: {
        "patch_successfully_applied": applied, "resolved": resolved,
        "tests_status": {"FAIL_TO_PASS": {"success": ["t"] * f2p[0], "failure": ["t"] * (f2p[1] - f2p[0])},
                         "PASS_TO_PASS": {"success": ["t"] * p2p[0], "failure": ["t"] * (p2p[1] - p2p[0])}}}}))


def test_analyze_acceptance_events_and_profile(tmp_path):
    sample = tmp_path / "s.jsonl"
    sample.write_text("\n".join(json.dumps({"task": {"task_id": t, "repo": r}}) for t, r in
                                [("t1", "a/a"), ("t2", "b/b")]) + "\n")
    logs = tmp_path / "logs"
    # t1: attempt1 resolved, attempt2 fixed-but-regressed. t2: attempt1 missed-clean, attempt2 no report.
    write_report(logs, "p-attempt1", "m", "t1", (2, 2), (5, 5), True)
    write_report(logs, "p-attempt2", "m", "t1", (2, 2), (4, 5), False)
    write_report(logs, "p-attempt1", "m", "t2", (0, 1), (3, 3), False)
    res = asa.analyze(str(sample), {"cfg": ("m", "p")}, attempts=2, logs_dir=str(logs), n_boot=200)
    c = res["configs"]["cfg"]
    assert c["failure_profile"] == {"resolved": 1, "fixed_but_regressed": 1, "missed_clean": 1,
                                    "missed_regressed": 0, "no_patch": 1, "harness_error": 0}
    assert c["resolved"]["rel@1"] == pytest.approx((0.5 + 0.0) / 2)
    assert c["f2p"]["rel@1"] == pytest.approx((1.0 + 0.0) / 2)
    assert c["resolved"]["rel@2"] == pytest.approx(0.5)
    assert c["resolved"]["pass^2"] == pytest.approx(0.0)
    assert c["f2p"]["pass^2"] == pytest.approx(0.5)
    assert "| cfg | resolved |" in asa.to_markdown(res)


def test_harness_errors_are_separated(tmp_path):
    sample = tmp_path / "s.jsonl"
    sample.write_text(json.dumps({"task": {"task_id": "t1", "repo": "a/a"}}) + "\n")
    (tmp_path / "m.p-attempt1.json").write_text(json.dumps({"error_ids": ["t1"]}))
    res = asa.analyze(str(sample), {"cfg": ("m", "p")}, attempts=1, logs_dir=str(tmp_path / "logs"),
                      n_boot=50, summary_dir=str(tmp_path))
    assert res["configs"]["cfg"]["failure_profile"]["harness_error"] == 1
    assert res["configs"]["cfg"]["harness_errors"] == [("t1", "p-attempt1")]


def test_estimators():
    assert asa.rel_at_k(5, 1, 5) == 1.0
    assert asa.rel_at_k(5, 0, 1) == 0.0
    assert asa.pass_hat_k(5, 5, 5) == 1.0
    assert asa.pass_hat_k(5, 4, 5) == 0.0
    assert asa.pass_hat_k(4, 2, 2) == pytest.approx(1 / 6)


def test_agent_cost_parsed_from_json_log(tmp_path):
    d = tmp_path / "att" / "p" / "t1" / "attempt-1"
    d.mkdir(parents=True)
    (d / "agent.log").write_text("noise\n" + json.dumps(
        {"type": "result", "total_cost_usd": 0.42, "modelUsage": {"claude-x": {}}}) + "\n")
    got = asa.agent_cost_and_models(tmp_path / "att", "p", "t1", 1)
    assert got == {"cost_usd": 0.42, "models": ["claude-x"]}
    assert asa.agent_cost_and_models(tmp_path / "att", "p", "t2", 1)["cost_usd"] is None


def test_merge_only_refuses_missing_attempts(tmp_path):
    from codebench.swebench_runner import run_swebench_attempts
    sample = tmp_path / "s.jsonl"
    sample.write_text(json.dumps({"task": {"task_id": "t1", "repo": "a/a", "description": "d",
                                           "difficulty": "medium"}, "base_commit": "abc"}) + "\n")
    with pytest.raises(RuntimeError, match="t1 attempt 1"):
        run_swebench_attempts(sample_path=sample, agent_cmd="noop", attempts=1, run_name="r",
                              attempts_dir=tmp_path / "att", predictions_dir=tmp_path / "pred",
                              merge_only=True)
    d = tmp_path / "att" / "r" / "t1" / "attempt-1"
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"task_id": "t1", "attempt": 1, "model_name": "m",
                                             "exit_code": 0, "timed_out": False, "latency_ms": 1.0,
                                             "record_dir": str(d)}))
    (d / "model.patch").write_text("diff")
    files = run_swebench_attempts(sample_path=sample, agent_cmd="noop", attempts=1, run_name="r",
                                  attempts_dir=tmp_path / "att", predictions_dir=tmp_path / "pred",
                                  merge_only=True)
    line = json.loads(Path(files[1]).read_text().strip())
    assert line["instance_id"] == "t1" and line["model_patch"] == "diff"
