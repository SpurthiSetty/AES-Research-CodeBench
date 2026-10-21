# AES scale-up runbook: 30 tasks × 5 attempts × 2 Claude Code configurations

Goal: replace the 5-task, 1-attempt pilot in the AES paper (Section 5) with a
sample that supports reliability@1 / reliability@5 / pass^5 with intervals,
and test whether the configuration ranking depends on the acceptance event
("fail-to-pass tests pass" vs. official "resolved").

Same agent setup as the pilot (Claude Code, headless, `acceptEdits`, no shell),
so the only thing that changes is the model.

| | Config A | Config B |
|---|---|---|
| `--model-name` | `cc-sonnet` | `cc-haiku` |
| Claude Code flag | `--model sonnet` | `--model haiku` |
| Run prefix | `aes-sonnet` | `aes-haiku` |

Run everything from the repo root on the Mac that ran the pilot (Docker running,
`claude` signed in).

## 0. Setup (once)

```bash
git checkout feature/aes-scaleup
pip install -e ".[eval]" datasets
claude --version          # record this in the paper
docker info >/dev/null && echo docker ok
python -m pytest -q       # 100 tests, all offline
```

## 1. Sample 30 tasks, stratified across repos, excluding the pilot

```bash
python scripts/prepare_swebench_sample.py --limit 30 --stratify-repo --seed 0 \
    --max-per-repo 4 \
    --exclude-ids astropy__astropy-12907 astropy__astropy-13033 \
        astropy__astropy-13236 astropy__astropy-13398 astropy__astropy-13453 \
    --output data/swebench_aes30.jsonl
```

It prints every task with its repo. Commit `data/swebench_aes30.jsonl` so the
sample is fixed.

## 2. Dry run (no model calls; checks clones and worktrees)

```bash
python scripts/run_swebench_agent.py --sample data/swebench_aes30.jsonl \
    --agent-cmd noop --run-name dryrun --shard 1/3
```

## 3. Generate patches — Config A, three shards in three terminals

Shards split whole repos, so parallel terminals never touch the same clone.

```bash
for S in 1 2 3; do   # run each line in its own terminal (or tmux pane)
python scripts/run_swebench_agent.py --sample data/swebench_aes30.jsonl \
    --model-name cc-sonnet --attempts 5 --timeout 1800 --run-name aes-sonnet \
    --resume --shard $S/3 \
    --agent-cmd 'claude -p "$(cat {prompt_file})" --permission-mode acceptEdits --model sonnet --output-format json --max-budget-usd 5'
done
```

`--resume` makes it safe to restart a shard after an interruption.
`--output-format json` puts cost and the exact model id in each `agent.log`.
`--max-budget-usd 5` caps any single runaway attempt.

When all three shards finish, write the merged prediction files (no agent runs):

```bash
python scripts/run_swebench_agent.py --sample data/swebench_aes30.jsonl \
    --model-name cc-sonnet --attempts 5 --run-name aes-sonnet --merge-only
# → predictions/aes-sonnet_attempt1.jsonl … attempt5.jsonl
# If it lists missing attempts, rerun that shard (with --resume) and try again.
```

## 4. Same for Config B

Repeat step 3 with `--model-name cc-haiku`, `--run-name aes-haiku`,
and `--model haiku`.

## 5. Evaluate with the official harness

```bash
for CFG in sonnet haiku; do for K in 1 2 3 4 5; do
python -m swebench.harness.run_evaluation \
    --dataset_name SWE-bench/SWE-bench_Verified \
    --predictions_path predictions/aes-${CFG}_attempt${K}.jsonl \
    --max_workers 2 --run_id aes-${CFG}-attempt${K}
done; done
```

The first pass builds the Docker images and is the slowest. If the analysis
in step 6 lists harness errors (common on Apple silicon for some repos),
re-run the harness for those instances with `--instance_ids ...` and the same `--run_id`.

## 6. Analyze

```bash
python scripts/analyze_swebench_acceptance.py \
    --sample data/swebench_aes30.jsonl --attempts 5 \
    --config sonnet=cc-sonnet:aes-sonnet \
    --config haiku=cc-haiku:aes-haiku \
    --output data/swebench_aes30_acceptance.json
```

Send back `data/swebench_aes30_acceptance.json` (and the printed tables).
That file is all the paper needs.

## If time runs short

- One configuration (Sonnet only) still gives reliability@1/@5/pass^5 with
  intervals and the failure profile; drop the ranking comparison.
- Fewer attempts: `--attempts 3` is the minimum for any pass^k / rel@k > 1 claim.
- Do not cut below 30 tasks; intervals over tasks get too wide to say anything.
