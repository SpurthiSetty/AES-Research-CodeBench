import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fd = sys.argv[2]
for f in ["OpenSans-Regular.ttf","OpenSans-Bold.ttf"]: fm.fontManager.addfont(f"{fd}/{f}")
plt.rcParams.update({"font.family":"Open Sans","font.size":8,"axes.edgecolor":"#52514e","axes.labelcolor":"#0b0b0b",
    "xtick.color":"#52514e","ytick.color":"#52514e","pdf.fonttype":42})
L = json.load(open(sys.argv[1]))
events = [("A0 tests pass","A0: tests pass"),("A2 + zero scanner findings","A1: A0 + zero scanner\nfindings"),
          ("A1 + no LLM-API delegation","A2: A0 + no LLM-API\ndelegation"),("A3 = A1 and A2","A3: A1 and A2")]
agents = [("anote-code","Sonnet-4.6 · Anote persona","#2a78d6"),("claude-code","Sonnet-4.6 · Claude Code persona","#eb6834"),("codex","GPT-4o · generic prompt","#1baf7a")]
fig, ax = plt.subplots(figsize=(6.3,2.5)); w=0.26; gap=0.02
for j,(key,lab,col) in enumerate(agents):
    xs=[i+(j-1)*(w+gap) for i in range(len(events))]
    v=[L[f"{e}|{key}"]["r5"] for e,_ in events]; ci=[L[f"{e}|{key}"]["ci"] for e,_ in events]
    ax.bar(xs, v, w, color=col, label=lab, zorder=2)
    ax.errorbar(xs, v, yerr=[[a-lo for a,(lo,hi) in zip(v,ci)],[hi-a for a,(lo,hi) in zip(v,ci)]],
                fmt="none", ecolor="#52514e", elinewidth=0.8, capsize=2, zorder=3)
    for x,a in zip(xs,v):
        inside = a >= 0.15
        ax.text(x, 0.03 if inside else a+0.02, f"{a:.2f}", ha="center", va="bottom", fontsize=6.5,
                color="white" if inside else "#0b0b0b", fontweight="bold", zorder=4)
ax.set_xticks(range(len(events))); ax.set_xticklabels([l for _,l in events])
ax.set_ylim(0,1.12); ax.set_yticks([0,.25,.5,.75,1]); ax.set_ylabel("reliability@5")
ax.grid(axis="y", color="#e4e3df", lw=0.6, zorder=0); ax.spines[["top","right"]].set_visible(False)
ax.legend(ncol=3, frameon=False, loc="upper right", bbox_to_anchor=(1,1.15), fontsize=7)
fig.tight_layout(); fig.savefig(sys.argv[3]); fig.savefig(sys.argv[3].replace(".pdf",".png"), dpi=200)
