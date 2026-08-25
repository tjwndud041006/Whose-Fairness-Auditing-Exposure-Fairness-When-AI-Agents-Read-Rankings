"""Round-21: P-MMF (Xu et al., WWW'23) as an external baseline on our Steam workload, audited post hoc.
Faithful port of the authors' CPU algorithm (ext/P-MMF/P-MMF.py): sigmoid-normalised scores, provider
quota rho=(1+1/P)*providerLen/sum, budget B_t=T*K*rho with exhausted-provider masking, adjusted-score
top-K selection, within-list order by raw score, dual mirror descent with cvxpy prefix projection.
Differences from their setup: T = our 200-user canonical workload (same users as loaders.steam_data, seed 11),
candidate pool = each user's MF-scored items (ranking_scores.npz, ~185 per user), K=10.
Evaluation: provider exposure under each curve v of our set, U_inf over our top-15 providers with our
merit targets (same ctx as every Steam number in the paper), Umax(a) at a in {0,.25,1}; NDCG@10 vs score sort.
Run: python pmmf_steam.py
"""
import json, sys, numpy as np, scipy.sparse as sp, cvxpy as cp, pandas as pd
from loaders import LOADERS, K, NQ
from policy_structure4 import build_canonical
from curveset import load_set

from config import DATA
BASE = f"{DATA}/steam"
d = np.load(f"{BASE}/ranking_scores.npz")
S = sp.coo_matrix((d["data"], (d["row"], d["col"])), shape=tuple(d["shape"])).tocsr()
iid2pid = {int(k): int(v) for k, v in json.load(open(f"{BASE}/iid2pid.json")).items()}
rng = np.random.default_rng(11)
users = [u for u in range(S.shape[0]) if S[u].nnz >= K]; rng.shuffle(users); users = users[:NQ]
data = LOADERS["Steam"]()
# sanity: same instance as the paper
row = S[users[0]]; sc = np.sort(row.data)[::-1][:K]
assert np.allclose(np.maximum(sc,1e-3), data[0][0]), "user selection mismatch"
ctx = build_canonical(data, ntop=15); vH = ctx["vH"]; top = ctx["top"]; tgt = ctx["tgt"]; bidx = ctx["bidx"]
agents = load_set(10, "main")                      # truncated-renormalised main set (paper canonical)
names = list(agents.keys()); V = [agents[n] for n in names]

# ---- provider quota rho from training interactions (their providerLen) ----
tr = pd.read_csv(f"{BASE}/steam.train", sep="\t")
pids = sorted(set(iid2pid.values())); P = len(pids); pidx = {p:i for i,p in enumerate(pids)}
plen = tr["publisher:token"].value_counts()
providerLen = np.array([plen.get(p, 0) for p in pids], float)
if providerLen.sum() == 0: providerLen = np.ones(P)
rho = (1+1/P)*providerLen/providerLen.sum()
item_num = S.shape[1]
A = np.zeros((item_num, P))
for i in range(item_num):
    if i in iid2pid: A[i, pidx[iid2pid[i]]] = 1

def sigmoid(x): return 1/(1+np.exp(-x))
def CPU_layer(ordered_tilde_dual, rho_o, lambd):
    m = len(rho_o); ans = cp.Variable(m)
    obj = cp.Minimize(cp.sum_squares(cp.multiply(rho_o, ans) - cp.multiply(rho_o, ordered_tilde_dual)))
    cons = [cp.sum(cp.multiply(rho_o[:i], ans[:i])) >= -lambd for i in range(1, m+1)]
    cp.Problem(obj, cons).solve(solver=cp.CLARABEL)
    return ans.value
def compute_next_dual(eta, rho, dual, gradient, lambd):
    tilde = dual - eta*gradient/rho/rho
    order = np.argsort(tilde*rho)
    nxt = CPU_layer(tilde[order], rho[order], lambd)
    return nxt[order.argsort()]

def pmmf(lambd, eta0=1e-3, alpha=0.1):
    T = NQ
    UI = np.full((T, item_num), -1e9)
    for t, u in enumerate(users):
        r = S[u]; UI[t, r.indices] = sigmoid(r.data)
    mu = np.zeros(P); B = T*K*rho; eta = eta0/np.sqrt(T); gcs = np.zeros(P); lists = []
    for t in range(T):
        x_title = UI[t] - A @ mu
        mask = (1.0 - (A @ (B > 0).astype(float))) * -10000.0
        x = np.argsort(x_title + mask)[::-1][:K]
        x = x[np.argsort(UI[t, x])[::-1]]             # within-list order by raw score (as in their code)
        lists.append(x)
        B = B - A[x].sum(0)
        g = -A[x].mean(0) + B/(T*K)
        g = alpha*g + (1-alpha)*gcs; gcs = g
        mu = compute_next_dual(eta, rho, mu, g, lambd)
    return lists, UI

def evaluate(lists, UI):
    # provider exposure under curve v; U_inf over our top-15 with our targets
    def E_of(v):
        E = np.zeros(len(ctx["brands"]))
        for t, x in enumerate(lists):
            for k, it in enumerate(x):
                b = f"pub{iid2pid.get(int(it), -1)}"
                if b in bidx: E[bidx[b]] += v[k]
        return E
    def U(v): E = E_of(v); g = np.array(top); return float(np.max(np.abs(E[g]/(NQ*v.sum()*tgt[g]) - 1)))
    def Umax(a): return max(U((1-a)*vH + a*v) for v in V) if a > 0 else U(vH)
    wt = [(U(v), n) for v, n in zip(V, names)]
    # NDCG@10 vs score-sorted ideal, and their MMF metric
    ndcg = []; expo = np.zeros(P)
    for t, x in enumerate(lists):
        s = UI[t]; ideal = np.sort(s)[::-1][:K]
        dcg = sum(s[it]/np.log2(k+2) for k, it in enumerate(x)); idcg = sum(ideal[k]/np.log2(k+2) for k in range(K))
        ndcg.append(dcg/idcg)
        for it in x:
            if int(it) in iid2pid: expo[pidx[iid2pid[int(it)]]] += 1
    MMF = float(np.min(expo/(rho*NQ*K)))
    def MMF_v(v):   # position-weighted version of their metric: provider share of total weighted exposure vs quota rho
        e = np.zeros(P)
        for t, x in enumerate(lists):
            for k, it in enumerate(x):
                if int(it) in iid2pid: e[pidx[iid2pid[int(it)]]] += v[k]
        return float(np.min(e/(rho*NQ*v.sum())))
    mmf_types = {n: MMF_v(v) for n, v in zip(names, V)}
    mmf_vH = MMF_v(vH); mmf_worst = min(mmf_types.values()); mmf_a25 = min(MMF_v((1-.25)*vH+.25*v) for v in V)
    return dict(U_a0=U(vH), U_a25=Umax(0.25), worst_type=max(wt)[0], worst_name=max(wt)[1],
                MMF_vH=mmf_vH, MMF_a25=mmf_a25, MMF_worst=mmf_worst, MMF_worst_name=min(mmf_types, key=mmf_types.get), MMF_types=mmf_types,
                per_type={n: u for u, n in wt}, ndcg=float(np.mean(ndcg)), MMF=MMF,
                in_top10_share=float(np.mean([len(set(x) & set(S[u].indices[np.argsort(-S[u].data)[:K]]))/K for x, u in zip(lists, users)])))

out = {}
# relevance sort (lambda -> 0 limit / their base model) for reference
T = NQ; UI0 = np.full((T, item_num), -1e9)
for t, u in enumerate(users):
    r = S[u]; UI0[t, r.indices] = sigmoid(r.data)
rel = [np.argsort(UI0[t])[::-1][:K] for t in range(T)]
out["relevance_sort"] = evaluate(rel, UI0); print("relevance", {k: (round(v,3) if isinstance(v,float) else v) for k, v in out["relevance_sort"].items() if k not in ("per_type","MMF_types")}, flush=True)
for lam, eta0 in [(0.1, 1e-3), (1.0, 1e-3), (0.1, 1e-1), (1.0, 1e-1), (0.1, 1.0)]:
    lists, UI = pmmf(lam, eta0=eta0)
    key = f"pmmf_lambda_{lam}_eta_{eta0}"
    out[key] = evaluate(lists, UI)
    print("P-MMF", key, {k: (round(v,3) if isinstance(v,float) else v) for k, v in out[key].items() if k not in ("per_type","MMF_types")}, flush=True)
json.dump(dict(results=out, names=names, rho_source="steam.train interactions", users=users), open("pmmf_steam.json", "w"), indent=1)
print("saved pmmf_steam.json")
