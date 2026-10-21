# AES Fall 2026 re-analysis

Scripts behind Tables 2–3, B1 and Figure 1 of the AES short paper. No API calls; they
rescore the stored rollouts in `data/` and the seeded simulator.

```bash
python analysis/synthetic_common_estimand.py . analysis/results/synthetic.json
python analysis/h4_rescore.py data/h4_results.json . analysis/results/h4_rescore.json
python analysis/plot_ladder.py analysis/results/h4_rescore_ladder.json <dir with OpenSans ttf> figures/acceptance_ladder.pdf
```

For the scaled SWE-bench run, see `docs/aes_scaleup_runbook.md` and
`scripts/analyze_swebench_acceptance.py`.
