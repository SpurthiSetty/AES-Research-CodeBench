# CodeBench

> **Measuring reliable, secure code generation by AI coding agents — beyond inflated pass@k scores.**

CodeBench is an evaluation framework that identifies and corrects a systematic
inflation in how AI coding agent benchmarks report `pass@k`, proposes a
corrected metric (`reliability@k`), and extends it to account for code security.
Results are validated on both synthetic multi-rollout experiments and real
SWE-bench Verified tasks.

📄 [Blog post](docs/blog/beyond-pass-k-reliability-security.md) ·
🤗 [Datasets](https://huggingface.co/anote-ai/codebench) ·
🖼 [Figures](figures/)

---

## Key Findings

| | Finding | Result |
|---|---|---|
| **H1** | Test-suite size alone changes the broken score | Two agents with identical 40% true reliability scored **0.976 vs 1.000** on broken pass@5 — differing only in number of tests |
| **H2** | Inflation magnitude | Broken pass@5 ≈ **0.96–0.97** collapsed to reliability@5 ≈ **0.00–0.12** once the category error was corrected |
| **H3** | Single-rollout proxy validity | A composite proxy (pass rate × regression penalty × tool efficiency) correlated with reliability@5 at only **Spearman ρ = 0.42** — below the 0.70 usefulness threshold; ≥5 rollouts are necessary |
| **H4** | Security screening gap | `security_adjusted_reliability@k` proposed and measured; **codex produced 24 insecure rollouts** vs 4–5 for Anthropic-backed agents |
| **Pilot** | Real-world SWE-bench gap | On 5 real astropy tasks: mean hidden-test pass rate = **0.80**, strict resolve rate = **0.20** — a 4× inflation consistent with H1/H2 |

---

## The Problem

`pass@k` (Chen et al., 2021) estimates the probability that at least one of `k`
**independent attempts** at a task succeeds. The formula requires `n` independent
samples and `c` correct ones.

Some agentic benchmarks silently substitute:
- `n` → number of **unit tests** in one submission
- `c` → number of **tests that passed**

Unit tests inside one submission are *not* independent attempts — they are
correlated sub-results of a single run. The substitution makes the score depend
on test-suite size, not agent quality.

## The Fix: `reliability@k`

Same formula, correct inputs:

- `n` = number of **independent rollout attempts** per (task, agent)
- `c` = rollouts where **all tests pass** (binary, full-task success)

```python
from codebench.evaluate import reliability_at_k

# results: List[ExecutionResult] with execution_success per rollout
score = reliability_at_k(results, k=5)
```

For security-sensitive tasks, add a security gate:

```python
from codebench.evaluate import security_adjusted_reliability_at_k

score = security_adjusted_reliability_at_k(results, submissions, k=5, security_threshold=0.80)
```

---

## Quickstart

```bash
pip install -e .
python scripts/run_demo.py          # leaderboard on 10 synthetic tasks
python scripts/run_experiments.py   # reproduce H1–H3
python scripts/run_security_exp.py  # reproduce H4 (real API calls, ~$0.50)
python scripts/plot_results.py      # regenerate all figures
```

---

## H4 Security Results (live API, 240 rollouts)

3 agents × 10 tasks (5 algorithmic + 5 security-sensitive) × 8 rollouts each.

| Agent | reliability@5 | security\_adj@5 | mean security score | insecure rollouts |
|---|---|---|---|---|
| anote-code | 0.988 | 0.988 | 0.991 | 5 / 80 |
| claude-code | 0.793 | 0.793 | 0.993 | 4 / 80 |
| codex (gpt-4o) | **1.000** | **1.000** | 0.957 | **24 / 80** |

Security-sensitive tasks targeted: `eval()`, `exec()`, `os.system()`,
`yaml.load()` without `Loader`, and `__import__()`.

---

## SWE-bench Verified Pilot

5 real `astropy/astropy` tasks, 1 attempt each, judged by the official harness.

| Task | Difficulty | Resolved | Hidden tests | Regressions |
|---|---|---|---|---|
| astropy-12907 | medium | ✅ | 15/15 | 0 |
| astropy-13033 | medium | ❌ | 20/21 | 0 |
| astropy-13236 | medium | ❌ | 644/646 | 0 |
| astropy-13398 | hard | ❌ | 63/72 | 5 |
| astropy-13453 | medium | ❌ | 2/10 | 7 |

**reliability@1 = 0.20 · mean pass rate = 0.80** — a 4× gap on real tasks.

See [docs/swebench_pilot_results.md](docs/swebench_pilot_results.md) for full
methodology and [docs/swebench_experiment.md](docs/swebench_experiment.md) to
run your own agent on SWE-bench tasks.

---

## Metrics

| Metric | Description |
|---|---|
| `reliability@k` | P(≥1 of k rollouts fully solves the task) — correct pass@k operationalization |
| `security_adjusted_reliability@k` | Same, counting only rollouts that are correct **and** secure |
| `single_rollout_proxy` | pass\_rate × (1 − regression\_rate) × tool\_efficiency — cheap but insufficient (H3) |
| `pass_rate` | tests\_passed / tests\_total — within-submission partial credit |
| `regression_rate` | newly-broken tests / tests\_total |

---

## Project Structure

```
src/codebench/
  core.py               # data models (CodeTask, ExecutionResult, AgentSubmission, …)
  data.py               # synthetic task factory (make_benchmark, make_rollout_benchmark)
  evaluate.py           # all metrics including reliability_at_k
  swebench_adapter.py   # leakage-safe SWE-bench task loader
  swebench_runner.py    # multi-attempt agent runner
scripts/
  run_experiments.py    # H1–H3 synthetic experiments
  run_security_exp.py   # H4 live-API security experiment
  plot_results.py       # figure generation
  run_swebench_agent.py # SWE-bench agent runner
data/
  h4_results.json                             # H4 experiment raw rollouts (240 rows)
  swebench_results_claude-code_smoke-v1.json  # SWE-bench pilot aggregates
docs/
  blog/beyond-pass-k-reliability-security.md  # full write-up
  swebench_experiment.md                      # SWE-bench workflow
  swebench_pilot_results.md                   # 5-task pilot analysis
figures/                                      # fig1–fig5 PNGs
```

---

## Hugging Face

| Repo | Contents |
|---|---|
| [anote-ai/codebench-tasks](https://huggingface.co/datasets/anote-ai/codebench-tasks) | 10 benchmark task definitions |
| [anote-ai/codebench-results](https://huggingface.co/datasets/anote-ai/codebench-results) | H4 security rollouts + SWE-bench pilot (`h4_security` and `swebench_smoke` configs) |
| [anote-ai/codebench](https://huggingface.co/anote-ai/codebench) | Model card with all figures and findings summary |

---


## Citation

```bibtex
@misc{codebench2026,
  title   = {CodeBench: Measuring Reliable and Secure Code Generation by AI Agents},
  author  = {Anote AI},
  year    = {2026},
  url     = {https://github.com/anote-ai/Research-CodeBench}
}
```
