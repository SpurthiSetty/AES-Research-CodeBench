"""Score both estimators against the same ground truth.
Ground truth per (task, agent): q = P(pr >= 0.95) = 1 - Phi((0.95 - s)/sigma) under the
simulator's clamped Gaussian (clamping at 1.0 keeps pr >= 0.95), so true rel@k = 1-(1-q)^k
and true single-attempt reliability = q."""
import sys, random, json, math
sys.path.insert(0, sys.argv[1] + "/src")
from statistics import NormalDist
from scipy.stats import spearmanr, kendalltau
from codebench.data import make_rollout_benchmark
from codebench.evaluate import _estimate_pass_at_k, reliability_at_k, single_rollout_proxy
from codebench.core import pass_at_k
AG = ["anote-code","claude-code","codex"]; K=5
def latent(n_tasks, seed):
    # replay make_rollout_benchmark's RNG stream to recover (s, sigma) per pair
    from codebench.data import TaskDifficulty
    rng = random.Random(seed); out = {}
    diffs = list(TaskDifficulty)
    for i in range(n_tasks):
        rng.choice(diffs)
        for a in AG:
            s = rng.uniform(0.30,1.0); sig = rng.uniform(0.05,0.25)*(1-s*0.5)
            out[(f"task-{i:03d}",a)] = (s,sig)
    return out
def truth(s,sig,k):
    q = 1-NormalDist(s,sig).cdf(0.95); return q, 1-(1-q)**k
def run(n_tasks, n_roll, seed):
    h = make_rollout_benchmark(n_tasks=n_tasks, agents=AG, n_rollouts=n_roll, seed=seed)
    lat = latent(n_tasks, seed)
    # sanity: replayed latent must reproduce observed success frequencies roughly
    g = {}
    for r in h.results: g.setdefault((r.task_id,r.agent_name),[]).append(r)
    subs = {}
    for s_ in h.submissions: subs.setdefault((s_.task_id,s_.agent_name),[]).append(s_)
    rows = []
    for key, rs in g.items():
        s,sig = lat[key]; q,t5 = truth(s,sig,K)
        n=len(rs); c=sum(r.execution_success for r in rs)
        rows.append(dict(key=key, q=q, true5=t5,
            broken5=_estimate_pass_at_k(rs,K), rel5=pass_at_k(n,c,min(K,n)), rel1=c/n,
            proxy=single_rollout_proxy(rs[0], subs[key][0]), mean_pr=sum(r.pass_rate for r in rs)/n))
    return rows
res = {}
for (nt, nr) in [(10,8),(200,20)]:
    rows = run(nt, nr, 42)
    agg = {}
    for a in AG:
        ra=[r for r in rows if r["key"][1]==a]
        m=lambda f: sum(r[f] for r in ra)/len(ra)
        agg[a]=dict(true5=m("true5"), broken5=m("broken5"), rel5=m("rel5"), q=m("q"), rel1=m("rel1"))
    bias_b = sum(r["broken5"]-r["true5"] for r in rows)/len(rows)
    bias_r = sum(r["rel5"]-r["true5"] for r in rows)/len(rows)
    mae_b = sum(abs(r["broken5"]-r["true5"]) for r in rows)/len(rows)
    mae_r = sum(abs(r["rel5"]-r["true5"]) for r in rows)/len(rows)
    # H3 on the pair level, vs ground truth, with bootstrap CI
    rng=random.Random(1); B=2000; px=[r["proxy"] for r in rows]; tt=[r["true5"] for r in rows]; ee=[r["rel5"] for r in rows]
    rho_true=spearmanr(px,tt).statistic; rho_est=spearmanr(px,ee).statistic
    bs=[]
    for _ in range(B):
        idx=[rng.randrange(len(rows)) for _ in rows]
        v=spearmanr([px[i] for i in idx],[tt[i] for i in idx]).statistic
        if v==v: bs.append(v)
    bs.sort(); ci=(bs[int(.025*len(bs))], bs[int(.975*len(bs))])
    zero=sum(1 for r in rows if r["true5"]<0.01)/len(rows); zero_est=sum(1 for r in rows if r["rel5"]==0)/len(rows)
    print(f"\n### n_tasks={nt}, rollouts={nr}, pairs={len(rows)}")
    for a,v in agg.items(): print(f"  {a:12} true rel@5={v['true5']:.3f}  broken pass@5={v['broken5']:.3f}  rollout rel@5={v['rel5']:.3f}  true p={v['q']:.3f}  est p={v['rel1']:.3f}")
    print(f"  bias vs truth: broken={bias_b:+.3f} (MAE {mae_b:.3f}); rollout={bias_r:+.3f} (MAE {mae_r:.3f})")
    rk=lambda f: sorted(AG,key=lambda a:-agg[a][f])
    print("  ranking truth:",rk("true5")," broken:",rk("broken5")," rollout:",rk("rel5"))
    print(f"  H3: rho(proxy,true rel@5)={rho_true:.3f} 95%CI[{ci[0]:.3f},{ci[1]:.3f}]  rho(proxy,est rel@5)={rho_est:.3f}  pairs true<0.01: {zero:.2f}, est==0: {zero_est:.2f}")
    res[f"{nt}x{nr}"]=dict(agg=agg,bias_broken=bias_b,bias_rollout=bias_r,mae_broken=mae_b,mae_rollout=mae_r,rho_true=rho_true,rho_ci=ci,rho_est=rho_est,frac_zero_true=zero,frac_zero_est=zero_est,pairs=len(rows))
json.dump(res, open(sys.argv[2],"w"), indent=1)
