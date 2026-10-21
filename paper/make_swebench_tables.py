#!/usr/bin/env python3
"""Turn analyze_swebench_acceptance.py output into LaTeX for paper Section 5.

    python paper/make_swebench_tables.py data/swebench_aes30_acceptance.json

Writes paper/generated/swebench_reliability.tex, paper/generated/swebench_profile.tex
and paper/generated/swebench_macros.tex. Every number in Section 5 should come from
these files (\\input them, or use the macros) rather than being typed by hand.
"""

import json
import re
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "generated"
PROFILE = [("resolved", "Resolved"), ("fixed_but_regressed", "Fixed, regressed"),
           ("missed_clean", "Missed, clean"), ("missed_regressed", "Missed, regressed"),
           ("no_patch", "No usable patch"), ("harness_error", "Harness error")]


def esc(s: str) -> str:
    return s.replace("_", r"\_")


def ci(v, c):
    return f"{v:.3f} [{c[0]:.2f}, {c[1]:.2f}]"


def macro_name(*parts: str) -> str:
    words = re.split(r"[^A-Za-z]+", " ".join(parts))
    return "swe" + "".join(w.capitalize() for w in words if w)


def main(path: str) -> None:
    res = json.loads(Path(path).read_text())
    k = res["k"]
    OUT.mkdir(exist_ok=True)

    rows = []
    for label, c in res["configs"].items():
        for ev, evname in (("f2p", "F2P pass"), ("resolved", "Resolved")):
            e = c[ev]
            rows.append(f"{esc(label)} & {evname} & {ci(e['rel@1'], e['rel@1_ci_task_bootstrap'])} & "
                        f"{ci(e[f'rel@{k}'], e[f'rel@{k}_ci'])} & {ci(e[f'pass^{k}'], e[f'pass^{k}_ci'])} \\\\")
    (OUT / "swebench_reliability.tex").write_text(
        "\\begin{tabularx}{\\linewidth}{@{}l l X X X@{}}\n\\toprule\n"
        f"Config & Acceptance event & $\\mathrm{{rel}}@1$ & $\\mathrm{{rel}}@{k}$ & pass$^{k}$ \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabularx}\n")

    prow = []
    for label, c in res["configs"].items():
        fp = c["failure_profile"]
        prow.append(esc(label) + " & " + " & ".join(str(fp[key]) for key, _ in PROFILE) + " \\\\")
    (OUT / "swebench_profile.tex").write_text(
        "\\begin{tabularx}{\\linewidth}{@{}X " + "c" * len(PROFILE) + "@{}}\n\\toprule\n"
        "Config & " + " & ".join(n for _, n in PROFILE) + " \\\\\n\\midrule\n"
        + "\n".join(prow) + "\n\\bottomrule\n\\end{tabularx}\n")

    macros = [f"\\newcommand{{\\sweNTasks}}{{{res['n_tasks']}}}",
              f"\\newcommand{{\\sweNRepos}}{{{len(res['repos'])}}}",
              f"\\newcommand{{\\sweAttempts}}{{{res['attempts']}}}"]
    for label, c in res["configs"].items():
        for ev in ("f2p", "resolved"):
            e = c[ev]
            macros.append(f"\\newcommand{{\\{macro_name(label, ev, 'relone')}}}{{{e['rel@1']:.3f}}}")
            macros.append(f"\\newcommand{{\\{macro_name(label, ev, 'relk')}}}{{{e[f'rel@{k}']:.3f}}}")
            macros.append(f"\\newcommand{{\\{macro_name(label, ev, 'passk')}}}{{{e[f'pass^{k}']:.3f}}}")
        pc = c["partial_credit_test_pass_rate"]["macro_over_rollouts_with_report"]
        if pc is not None:
            macros.append(f"\\newcommand{{\\{macro_name(label, 'partial')}}}{{{pc:.3f}}}")
    (OUT / "swebench_macros.tex").write_text("\n".join(macros) + "\n")
    print("Wrote", ", ".join(p.name for p in sorted(OUT.glob("swebench_*.tex"))))
    print("Macros:", ", ".join(m.split("{")[1].rstrip("}") for m in macros))


if __name__ == "__main__":
    main(sys.argv[1])
