import json, sys, re, random, itertools
from math import comb
sys.path.insert(0, sys.argv[2] + "/src")
from codebench.evaluate import _INSECURE_PATTERNS
recs = json.load(open(sys.argv[1]))
def findings(code): return [n for p,n in _INSECURE_PATTERNS if re.search(p, code or "")]
def passk(n,c,k):
    if n-c < k: return 1.0
    return 1 - comb(n-c,k)/comb(n,k)
agents = sorted({r["agent_name"] for r in recs}); tasks = sorted({r["task_id"] for r in recs})
for r in recs:
    r["f"] = findings(r["generated_code"])
    # check stored score consistent with recomputed
    assert abs(r["security_score"] - max(0,1-len(r["f"])/7)) < 1e-9, r["task_id"]
def score(rs, pred, k):
    g = {}
    for r in rs: g.setdefault(r["task_id"], []).append(r)
    per = {t: passk(len(v), sum(pred(x) for x in v), min(k,len(v))) for t,v in g.items()}
    return per
func = lambda r: r["execution_success"]
lenient_last = None
binding = lambda r: r["execution_success"] and len(r["f"]) == 0
lenient = lambda r: r["execution_success"] and r["security_score"] >= 0.80
random.seed(0)
def boot(per, B=10000):
    v = list(per.values()); m = [sum(random.choice(v) for _ in v)/len(v) for _ in range(B)]
    m.sort(); return m[int(.025*B)], m[int(.975*B)]
out = {}
for subset in ["all", "sec", "algo"]:
    print(f"\n== tasks: {subset} ==")
    print(f"{'agent':12} {'n':>3} {'func@1':>7} {'func@5':>7} {'lenient@5':>9} {'bind@1':>7} {'bind@5':>7} {'bind@5 95%CI':>14} {'pass+flag':>9}")
    for a in agents:
        rs = [r for r in recs if r["agent_name"]==a and (subset=="all" or r["task_id"].startswith(subset))]
        f1 = score(rs,func,1); f5=score(rs,func,5); l5=score(rs,lenient,5); b1=score(rs,binding,1); b5=score(rs,binding,5)
        mean = lambda d: sum(d.values())/len(d)
        lo,hi = boot(b5)
        pf = sum(1 for r in rs if r["execution_success"] and r["f"])
        print(f"{a:12} {len(rs):>3} {mean(f1):7.3f} {mean(f5):7.3f} {mean(l5):9.3f} {mean(b1):7.3f} {mean(b5):7.3f}  [{lo:.3f},{hi:.3f}] {pf:>9}")
        out[(subset,a)] = dict(func1=mean(f1),func5=mean(f5),len5=mean(l5),bind1=mean(b1),bind5=mean(b5),ci=(lo,hi),passflag=pf,n=len(rs))
print("\nper-task: functionally-correct rollouts that are flagged, by agent")
for t in tasks:
    row = []
    for a in agents:
        rs=[r for r in recs if r["agent_name"]==a and r["task_id"]==t]
        row.append(f"{a}: c={sum(func(r) for r in rs)}/8 flagged-pass={sum(1 for r in rs if func(r) and r['f'])} pats={sorted({x for r in rs for x in r['f']})}")
    print(t, " | ".join(row))
json.dump({f"{k[0]}|{k[1]}":v for k,v in out.items()}, open(sys.argv[3],"w"), indent=1)

print("\n=== acceptance-event ladder (all 10 tasks); reliability@1 = mean per-task c/n, reliability@5 via Chen estimator ===")
deleg = lambda r: bool(re.search(r'import (anthropic|openai)|from (anthropic|openai)', r["generated_code"]))
preds = {"A0 tests pass": func,
         "A1 + no LLM-API delegation": lambda r: func(r) and not deleg(r),
         "A2 + zero scanner findings": lambda r: func(r) and not r["f"],
         "A3 = A1 and A2": lambda r: func(r) and not deleg(r) and not r["f"]}
ladder = {}
for name, pr in preds.items():
    row=[]
    for a in agents:
        rs=[r for r in recs if r["agent_name"]==a]
        s1=score(rs,pr,1); s5=score(rs,pr,5); lo,hi=boot(s5)
        m=lambda d: sum(d.values())/len(d)
        row.append(f"{a}: @1={m(s1):.3f} @5={m(s5):.3f} [{lo:.2f},{hi:.2f}]")
        ladder[f"{name}|{a}"]=dict(r1=m(s1),r5=m(s5),ci=(lo,hi))
    print(f"{name:30}", " | ".join(row))
# paired bootstrap over tasks: P(codex < anote-code) under A2 and A0 at @5
for name in ["A0 tests pass","A2 + zero scanner findings","A3 = A1 and A2"]:
    per={a:score([r for r in recs if r["agent_name"]==a],preds[name],5) for a in agents}
    B=10000; wins=collections_c=0
    for _ in range(B):
        ts=[random.choice(tasks) for _ in tasks]
        d=sum(per["codex"][t]-per["anote-code"][t] for t in ts)/len(ts)
        wins+= d<0
    print(f"paired bootstrap {name}: P(codex < anote-code @5) = {wins/B:.3f}")
json.dump(ladder, open(sys.argv[3].replace(".json","_ladder.json"),"w"), indent=1)
