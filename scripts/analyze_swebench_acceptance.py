#!/usr/bin/env python3
"""Score SWE-bench rollouts under several acceptance events (AES paper, Sec. 5).

Reads the official harness's per-instance ``report.json`` files directly, so
FAIL_TO_PASS and PASS_TO_PASS outcomes stay separate. For each configuration
and acceptance event it reports reliability@1, reliability@k and pass^k with
95% task-bootstrap intervals, the partial-credit test pass rate, and a
failure profile of every rollout.

Acceptance events:
  f2p       every FAIL_TO_PASS test passes (the issue is fixed; regressions ignored)
  resolved  every FAIL_TO_PASS and PASS_TO_PASS test passes (official "resolved")

Usage (run ids are ``{prefix}-attempt{1..k}``, as in docs/swebench_experiment.md):

    python scripts/analyze_swebench_acceptance.py \\
        --sample data/swebench_aes30.jsonl --attempts 5 \\
        --config sonnet=cc-sonnet:aes-sonnet \\
        --config haiku=cc-haiku:aes-haiku \\
        --output data/swebench_aes30_acceptance.json

``--config`` is ``LABEL=MODEL_NAME:RUN_PREFIX`` where MODEL_NAME is the
``--model-name`` passed to run_swebench_agent.py. A rollout with no report
(empty patch, failed apply, or missing run) counts as a failure. Harness errors
listed in the run summary are reported separately so they can be re-run.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from math import comb
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from codebench.swebench_adapter import read_swebench_sample  # noqa: E402

EVENTS = ("f2p", "resolved")
CATEGORIES = ("resolved", "fixed_but_regressed", "missed_clean", "missed_regressed", "no_patch", "harness_error")


def harness_error_ids(summary_dir: Optional[Path], run_id: str, model_name: str) -> set:
    """Instance ids the harness itself failed on (image build, container, timeout)."""
    if summary_dir is None:
        return set()
    path = Path(summary_dir) / f"{model_name.replace('/', '__')}.{run_id}.json"
    if not path.exists():
        return set()
    return set(json.loads(path.read_text(encoding="utf-8")).get("error_ids", []))


def load_rollout(logs_dir: Path, run_id: str, model_name: str, instance_id: str,
                 errors: Optional[set] = None) -> Dict[str, Any]:
    """One rollout's outcome from its harness report (or a no-patch failure)."""
    path = logs_dir / run_id / model_name.replace("/", "__") / instance_id / "report.json"
    out: Dict[str, Any] = {"instance_id": instance_id, "run_id": run_id, "has_report": False,
                           "f2p_pass": 0, "f2p_total": 0, "p2p_pass": 0, "p2p_total": 0,
                           "resolved": False}
    if errors and instance_id in errors:
        out["category"] = "harness_error"
        return out
    if not path.exists():
        out["category"] = "no_patch"
        return out
    report = json.loads(path.read_text(encoding="utf-8"))
    entry = report.get(instance_id, report)
    status = entry.get("tests_status", {})
    f2p, p2p = status.get("FAIL_TO_PASS", {}), status.get("PASS_TO_PASS", {})
    out.update(
        has_report=bool(entry.get("patch_successfully_applied", True)),
        f2p_pass=len(f2p.get("success", [])),
        f2p_total=len(f2p.get("success", [])) + len(f2p.get("failure", [])),
        p2p_pass=len(p2p.get("success", [])),
        p2p_total=len(p2p.get("success", [])) + len(p2p.get("failure", [])),
        resolved=bool(entry.get("resolved", False)),
    )
    if not out["has_report"]:
        out["category"] = "no_patch"
        return out
    fixed = out["f2p_total"] > 0 and out["f2p_pass"] == out["f2p_total"]
    regressed = out["p2p_pass"] < out["p2p_total"]
    if out["resolved"]:
        out["category"] = "resolved"
    elif fixed and regressed:
        out["category"] = "fixed_but_regressed"
    elif regressed:
        out["category"] = "missed_regressed"
    else:
        out["category"] = "missed_clean"
    return out


def agent_cost_and_models(attempts_dir: Optional[Path], run_prefix: str, instance_id: str,
                          attempt: int) -> Dict[str, Any]:
    """Cost and model ids from a ``claude -p --output-format json`` agent.log, if present."""
    out: Dict[str, Any] = {"cost_usd": None, "models": []}
    if attempts_dir is None:
        return out
    log = Path(attempts_dir) / run_prefix / instance_id / f"attempt-{attempt}" / "agent.log"
    if not log.exists():
        return out
    for line in reversed(log.read_text(encoding="utf-8", errors="replace").splitlines()):
        line = line.strip()
        if not (line.startswith("{") and "total_cost_usd" in line):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        out["cost_usd"] = obj.get("total_cost_usd")
        out["models"] = sorted((obj.get("modelUsage") or {}).keys())
        break
    return out


def accepted(r: Dict[str, Any], event: str) -> bool:
    if event == "resolved":
        return bool(r["resolved"])
    if event == "f2p":
        return r["has_report"] and r["f2p_total"] > 0 and r["f2p_pass"] == r["f2p_total"]
    raise ValueError(event)


def rel_at_k(n: int, c: int, k: int) -> float:
    """Chen et al. (2021) unbiased estimate of P(at least one of k succeeds)."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """Unbiased estimate of P(all k succeed) (pass^k, Yao et al. 2025)."""
    return comb(c, k) / comb(n, k) if c >= k else 0.0


def per_task_scores(rollouts: List[Dict[str, Any]], tasks: List[str], event: str, k: int) -> Dict[str, Dict[str, float]]:
    scores = {}
    for t in tasks:
        rs = [r for r in rollouts if r["instance_id"] == t]
        n, c = len(rs), sum(accepted(r, event) for r in rs)
        scores[t] = {"rel1": c / n, "relk": rel_at_k(n, c, min(k, n)), "passk": pass_hat_k(n, c, min(k, n))}
    return scores


def bootstrap_ci(values: List[float], rng: random.Random, b: int) -> List[float]:
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(b))
    return [means[int(0.025 * b)], means[min(b - 1, int(0.975 * b))]]


def wilson(c: int, n: int, z: float = 1.96) -> List[float]:
    if n == 0:
        return [0.0, 1.0]
    p = c / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [max(0.0, centre - half), min(1.0, centre + half)]


def analyze(sample: str, configs: Dict[str, tuple], attempts: int, logs_dir: str,
            k: Optional[int] = None, n_boot: int = 10000, seed: int = 0,
            summary_dir: Optional[str] = ".", attempts_dir: Optional[str] = None) -> Dict[str, Any]:
    k = k or attempts
    records = read_swebench_sample(sample)
    tasks = [r["task"]["task_id"] for r in records]
    repo_of = {r["task"]["task_id"]: r["task"]["repo"] for r in records}
    rng = random.Random(seed)
    out: Dict[str, Any] = {"n_tasks": len(tasks), "attempts": attempts, "k": k,
                           "repos": sorted(set(repo_of.values())), "configs": {}}
    per_task_all: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for label, (model_name, prefix) in configs.items():
        errs = {a: harness_error_ids(Path(summary_dir) if summary_dir else None,
                                     f"{prefix}-attempt{a}", model_name)
                for a in range(1, attempts + 1)}
        rollouts = [load_rollout(Path(logs_dir), f"{prefix}-attempt{a}", model_name, t, errs[a])
                    for t in tasks for a in range(1, attempts + 1)]
        costs = [agent_cost_and_models(Path(attempts_dir) if attempts_dir else None, prefix, t, a)
                 for t in tasks for a in range(1, attempts + 1)]
        known = [c["cost_usd"] for c in costs if c["cost_usd"] is not None]
        cfg: Dict[str, Any] = {"model_name": model_name, "run_prefix": prefix,
                               "agent_cost_usd": {"total": sum(known), "runs_with_cost": len(known),
                                                  "mean_per_run": sum(known) / len(known) if known else None},
                               "agent_models_seen": sorted({m for c in costs for m in c["models"]}),
                               "n_rollouts": len(rollouts),
                               "n_with_report": sum(r["has_report"] for r in rollouts),
                               "failure_profile": {c: sum(r["category"] == c for r in rollouts) for c in CATEGORIES},
                               "harness_errors": sorted({(r["instance_id"], r["run_id"]) for r in rollouts
                                                         if r["category"] == "harness_error"})}
        tp = sum(r["f2p_pass"] + r["p2p_pass"] for r in rollouts)
        tt = sum(r["f2p_total"] + r["p2p_total"] for r in rollouts)
        with_tests = [r for r in rollouts if r["f2p_total"] + r["p2p_total"] > 0]
        cfg["partial_credit_test_pass_rate"] = {
            "macro_over_rollouts_with_report": (sum((r["f2p_pass"] + r["p2p_pass"]) / (r["f2p_total"] + r["p2p_total"])
                                                     for r in with_tests) / len(with_tests)) if with_tests else None,
            "pooled": tp / tt if tt else None,
        }
        per_task_all[label] = {}
        for ev in EVENTS:
            pt = per_task_scores(rollouts, tasks, ev, k)
            per_task_all[label][ev] = pt
            c_total = sum(accepted(r, ev) for r in rollouts)
            cfg[ev] = {
                "accepted_rollouts": c_total,
                "rel@1": sum(v["rel1"] for v in pt.values()) / len(pt),
                "rel@1_ci_task_bootstrap": bootstrap_ci([v["rel1"] for v in pt.values()], rng, n_boot),
                "rel@1_ci_wilson_rollouts": wilson(c_total, len(rollouts)),
                f"rel@{k}": sum(v["relk"] for v in pt.values()) / len(pt),
                f"rel@{k}_ci": bootstrap_ci([v["relk"] for v in pt.values()], rng, n_boot),
                f"pass^{k}": sum(v["passk"] for v in pt.values()) / len(pt),
                f"pass^{k}_ci": bootstrap_ci([v["passk"] for v in pt.values()], rng, n_boot),
                "tasks_solved_at_least_once": sum(v["relk"] > 0 for v in pt.values()) if k == attempts else None,
            }
        cfg["per_repo_resolved_rel@1"] = {
            repo: sum(per_task_all[label]["resolved"][t]["rel1"] for t in tasks if repo_of[t] == repo)
            / sum(1 for t in tasks if repo_of[t] == repo)
            for repo in out["repos"]
        }
        out["configs"][label] = cfg
    labels = list(configs)
    if len(labels) >= 2:
        out["paired"] = {}
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = labels[i], labels[j]
                for ev in EVENTS:
                    for metric in ("rel1", "relk"):
                        diffs = [per_task_all[a][ev][t][metric] - per_task_all[b][ev][t][metric] for t in tasks]
                        boots = []
                        for _ in range(n_boot):
                            idx = [rng.randrange(len(tasks)) for _ in tasks]
                            boots.append(sum(diffs[x] for x in idx) / len(idx))
                        boots.sort()
                        out["paired"][f"{a}-{b}|{ev}|{metric}"] = {
                            "mean_diff": sum(diffs) / len(diffs),
                            "ci": [boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]],
                            "p_a_gt_b": sum(x > 0 for x in boots) / n_boot,
                        }
    out["rankings"] = {ev: sorted(labels, key=lambda l: -out["configs"][l][ev][f"rel@{k}"]) for ev in EVENTS}
    return out


def to_markdown(res: Dict[str, Any]) -> str:
    k = res["k"]
    lines = [f"Tasks: {res['n_tasks']} from {len(res['repos'])} repos · attempts per task: {res['attempts']}", "",
             f"| Config | Event | rel@1 [95% CI] | rel@{k} [95% CI] | pass^{k} [95% CI] |", "|---|---|---|---|---|"]
    f = lambda v, ci: f"{v:.3f} [{ci[0]:.2f}, {ci[1]:.2f}]"
    for label, c in res["configs"].items():
        for ev in EVENTS:
            e = c[ev]
            lines.append(f"| {label} | {ev} | {f(e['rel@1'], e['rel@1_ci_task_bootstrap'])} | "
                         f"{f(e[f'rel@{k}'], e[f'rel@{k}_ci'])} | {f(e[f'pass^{k}'], e[f'pass^{k}_ci'])} |")
    lines += ["", "| Config | " + " | ".join(CATEGORIES) + " | partial-credit pass rate |",
              "|---|" + "---|" * (len(CATEGORIES) + 1)]
    for label, c in res["configs"].items():
        pc = c["partial_credit_test_pass_rate"]["macro_over_rollouts_with_report"]
        pc_text = f"{pc:.3f}" if pc is not None else "n/a"
        lines.append(f"| {label} | " + " | ".join(str(c["failure_profile"][x]) for x in CATEGORIES)
                     + f" | {pc_text} |")
    lines += ["", "| Config | agent cost (USD, runs with cost) | models seen |", "|---|---|---|"]
    for label, c in res["configs"].items():
        ac = c["agent_cost_usd"]
        lines.append(f"| {label} | {ac['total']:.2f} ({ac['runs_with_cost']}) | {', '.join(c['agent_models_seen']) or 'n/a'} |")
    lines += ["", "Rankings by rel@k: " + "; ".join(f"{ev}: {' > '.join(r)}" for ev, r in res["rankings"].items())]
    for key, v in res.get("paired", {}).items():
        lines.append(f"- paired {key}: diff {v['mean_diff']:+.3f} [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}], "
                     f"P(first > second) = {v['p_a_gt_b']:.3f}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", required=True, help="Sample JSONL used for the runs")
    parser.add_argument("--config", action="append", required=True,
                        help="LABEL=MODEL_NAME:RUN_PREFIX (repeat for each configuration)")
    parser.add_argument("--attempts", type=int, required=True, help="Attempts per task in each run")
    parser.add_argument("-k", type=int, default=None, help="k for rel@k and pass^k (default: attempts)")
    parser.add_argument("--logs-dir", default=os.path.join("logs", "run_evaluation"),
                        help="Harness logs directory (default: logs/run_evaluation)")
    parser.add_argument("--summary-dir", default=".",
                        help="Where the harness wrote {model}.{run_id}.json summaries (default: .)")
    parser.add_argument("--attempts-dir", default="attempts",
                        help="Runner attempts dir, used to read agent cost/model from agent.log")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None, help="Write full results JSON here")
    args = parser.parse_args()

    configs = {}
    for spec in args.config:
        label, rest = spec.split("=", 1)
        model_name, prefix = rest.rsplit(":", 1)
        configs[label] = (model_name, prefix)
    res = analyze(args.sample, configs, args.attempts, args.logs_dir, k=args.k,
                  n_boot=args.bootstrap, seed=args.seed, summary_dir=args.summary_dir, attempts_dir=args.attempts_dir)
    print(to_markdown(res))
    for label, c in res["configs"].items():
        if c["harness_errors"]:
            print(f"\nWARNING {label}: {len(c['harness_errors'])} harness errors (counted as failures). "
                  "Re-run the harness for these before reporting:")
            for iid, rid in c["harness_errors"]:
                print(f"  {rid}  {iid}")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
