# CLAUDE.md: AES Fall 2026 resubmission of "Beyond Pass@k"

Work on branch `aes-2026`. Do not push to `anote-ai/research-codebench`; push only to this fork.

## Goal

The paper in `paper/` is a short paper for the Agent Evaluation Science symposium
(deadline Oct 25, 2026; 4–6 pages excluding references; appendices free; AES LaTeX
template; not anonymous). Its weakest part is Section 5, a 5-task, 1-attempt SWE-bench
pilot. Replace it with the scaled run described in `docs/aes_scaleup_runbook.md`
(30 tasks × 5 attempts × Claude Code with Sonnet and with Haiku), then update the paper
from the results and rebuild the PDF.

## Phase 1: run the experiment

Follow `docs/aes_scaleup_runbook.md` exactly.

- Agent shards and harness evaluation take hours. Start each in the background in its own
  tmux window, logging to `logs/aes/<name>.log`. Never run them as one blocking foreground
  command; your shell commands time out after about 10 minutes.
- Poll every ~10 minutes: count `attempts/<run>/*/attempt-*/meta.json`, tail the logs,
  and check `tmux ls`. If a shard died, restart it with the same command (it has `--resume`).
- Sonnet first, then Haiku. Merge each with `--merge-only`. Then evaluate all 10
  prediction files with the harness, two workers at a time.
- If the analysis reports harness errors, re-run the harness for only those instances
  with the same `--run_id`, then re-run the analysis.
- Never commit `data/swebench_aes30.jsonl`; it contains gold patches. Commit
  `data/swebench_aes30_ids.txt` (see the runbook) and `data/swebench_aes30_acceptance.json`.
- Never open the gold `hidden_reference` fields, and never pass them to the agent.

Do not change the sample, models, permission mode, timeout, attempt count or acceptance
events. Do not re-run any attempt to get a better result. If something forces a change
(e.g. a repo's Docker image cannot build), stop and tell the user what and why.

## Phase 2: update the paper

1. `python paper/make_swebench_tables.py data/swebench_aes30_acceptance.json`
   writes `paper/generated/swebench_{reliability,profile,macros}.tex`.
2. In `paper/main.tex`:
   - Load the macros in the preamble: `\input{generated/swebench_macros.tex}`.
   - Rewrite Section 5 around the scaled run, using `\input{generated/...}` tables and
     the `\swe...` macros. Every number in Section 5 must come from those files; never
     type a result by hand. Report reliability@1, reliability@5, pass^5 with intervals,
     the failure profile, and whether the Sonnet vs. Haiku ranking differs between the
     F2P and resolved acceptance events, using the paired-bootstrap results in the JSON.
   - Move the old 5-task pilot table to the appendix as a pilot, and say the scaled
     sample excludes those 5 tasks.
   - Update the abstract, the Section 1 bullet, the protocol table (Appendix A),
     Limitations and the Conclusion to match. Remove "we could not scale it before the
     deadline".
   - Record the Claude Code version and the model ids from `agent_models_seen` in the JSON.
3. Leave Sections 2–4 alone. Their numbers come from `analysis/` and do not change.
4. Build with `cd paper && latexmk -xelatex main.tex` (pdfLaTeX is not supported).
   Main text through the Conclusion must fit in 6 pages; check the rendered PDF.
5. Commit and push to `aes-2026`.

## Honesty rules for the write-up

- Report what the data show. If the ranking does not change, or intervals overlap, say so
  plainly. That is still a finding.
- Claims must match the intervals. Do not call a difference significant unless the
  paired-bootstrap interval excludes zero.
- Count every attempt, failures and empty patches included. Report harness errors that
  could not be fixed, with their number.
- Keep the existing caveats: one agent harness, `acceptEdits` with no shell, one seed for
  task sampling.

## Ask the user before

- changing the experimental design or the sample;
- agent spend beyond what the runbook implies (each attempt is capped at $5 by `--max-budget-usd`);
- deleting any data under `attempts/`, `predictions/`, `logs/` or `data/`;
- changing author names, affiliations or the title.

## Useful facts

- Tests: `python -m pytest -q` (all offline).
- Re-analysis for Sections 3–4: `analysis/README.md`.
- LaTeX on macOS: `brew install --cask mactex-no-gui`, then open a new terminal.
