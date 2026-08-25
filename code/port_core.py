"""Shared core for dataset ports: joint LP, U_of, lambda_break sweep, sign-indeterminacy.
A dataset port supplies `data` = list of (merit K-array, provider K-array) per query.
run_all(data, K, ntop) reproduces the four core numbers and returns a dict.
"""
import json, glob
from config import CURVES
import numpy as np
from scipy import sparse
from scipy.optimize import linprog

def load_curves(K, which="main"):
    """Round-9 rebase: curves come from the frozen manifest in curveset.py
    (which='main' -> 8 representative agents; 'ext'; 'full').
    Pre-rebase behaviour (lcurve_* glob) available via which='legacy'."""
    vH = 1/np.log2(np.arange(2, K+2)); vH /= vH.sum()
    if which == "legacy":
        agent = {}
        for fn in glob.glob(f"{CURVES}/lcurve_*.json"):
            d = json.load(open(fn)); c = np.array(d["results"]["pick"]["curve"], dtype=float)[:K]
            if c.sum() > 0: agent[d["model"].split("/")[-1]] = c/c.sum()
        return vH, agent
    from curveset import load_set
    return vH, load_set(K, which)

def build(data, K, ntop=15, vH_override=None):
    """Solver context without running the full analysis: returns dict with
    solve(cert_curves)->(P,U), U_of(P,v), vH, agent_curves, brands/top/tgt."""
    NQ = len(data)
    vH, agent_curves = load_curves(K)
    if vH_override is not None:
        vH = np.asarray(vH_override, dtype=float); vH = vH/vH.sum()
    brands = sorted(set(b for _, bs in data for b in bs)); bidx = {b: i for i, b in enumerate(brands)}
    M = np.zeros(len(brands))
    for m, bs in data:
        for mi, b in zip(m, bs): M[bidx[b]] += mi
    top = list(np.argsort(-M)[:min(ntop, len(brands))]); tgt = M/M.sum()
    NV = K*K; NVAR = NQ*NV+1; T = NQ*NV

    def solve(cert_curves):
        rows, cols, vals, b_eq = [], [], [], []; r = 0
        for q in range(NQ):
            for i in range(K):
                for k in range(K): rows.append(r); cols.append(q*NV+i*K+k); vals.append(1.0)
                b_eq.append(1.0); r += 1
            for k in range(K):
                for i in range(K): rows.append(r); cols.append(q*NV+i*K+k); vals.append(1.0)
                b_eq.append(1.0); r += 1
        A_eq = sparse.coo_matrix((vals, (rows, cols)), shape=(r, NVAR)).tocsr()
        urows, ucols, uvals, b_ub = [], [], [], []; ur = 0
        for v in cert_curves:
            tot = NQ*v.sum()
            for b in top:
                for sign in (1.0, -1.0):
                    for q, (m, bs) in enumerate(data):
                        for i, bb in enumerate(bs):
                            if bidx[bb] == b:
                                for k in range(K): urows.append(ur); ucols.append(q*NV+i*K+k); uvals.append(sign*v[k])
                    urows.append(ur); ucols.append(T); uvals.append(-tot*tgt[b])
                    b_ub.append(sign*tot*tgt[b]); ur += 1
        A_ub = sparse.coo_matrix((uvals, (urows, ucols)), shape=(ur, NVAR)).tocsr()
        c_obj = np.zeros(NVAR); c_obj[T] = 1
        res = linprog(c_obj, A_eq=A_eq, b_eq=np.array(b_eq), A_ub=A_ub, b_ub=np.array(b_ub),
                      bounds=[(0, 1)]*(NQ*NV)+[(0, None)], method="highs")
        assert res.status == 0, res.message
        return res.x[:NQ*NV].reshape(NQ, K, K), res.x[T]

    def U_of(Ps, v):
        E = np.zeros(len(brands))
        for q, (m, bs) in enumerate(data):
            e = Ps[q] @ v
            np.add.at(E, [bidx[b] for b in bs], e)
        tot = NQ*v.sum()
        ti = np.array(top)
        return np.max(np.abs(E[ti]/(tot*tgt[ti])-1))

    return {"solve": solve, "U_of": U_of, "vH": vH, "agent_curves": agent_curves,
            "brands": brands, "bidx": bidx, "top": top, "tgt": tgt}


def run_all(data, K, ntop=15, sweep_hi=0.6, sweep_n=61, vH_override=None):
    NQ = len(data)
    vH, agent_curves = load_curves(K)
    if vH_override is not None:
        vH = np.asarray(vH_override, dtype=float); vH = vH/vH.sum()
    vs_agents = list(agent_curves.values())
    brands = sorted(set(b for _, bs in data for b in bs)); bidx = {b: i for i, b in enumerate(brands)}
    M = np.zeros(len(brands))
    for m, bs in data:
        for mi, b in zip(m, bs): M[bidx[b]] += mi
    top = list(np.argsort(-M)[:min(ntop, len(brands))]); tgt = M/M.sum()
    NV = K*K; NVAR = NQ*NV+1; T = NQ*NV

    def solve(cert_curves):
        rows, cols, vals, b_eq = [], [], [], []; r = 0
        for q in range(NQ):
            for i in range(K):
                for k in range(K): rows.append(r); cols.append(q*NV+i*K+k); vals.append(1.0)
                b_eq.append(1.0); r += 1
            for k in range(K):
                for i in range(K): rows.append(r); cols.append(q*NV+i*K+k); vals.append(1.0)
                b_eq.append(1.0); r += 1
        A_eq = sparse.coo_matrix((vals, (rows, cols)), shape=(r, NVAR)).tocsr()
        urows, ucols, uvals, b_ub = [], [], [], []; ur = 0
        for v in cert_curves:
            tot = NQ*v.sum()
            for b in top:
                for sign in (1.0, -1.0):
                    for q, (m, bs) in enumerate(data):
                        for i, bb in enumerate(bs):
                            if bidx[bb] == b:
                                for k in range(K): urows.append(ur); ucols.append(q*NV+i*K+k); uvals.append(sign*v[k])
                    urows.append(ur); ucols.append(T); uvals.append(-tot*tgt[b])
                    b_ub.append(sign*tot*tgt[b]); ur += 1
        A_ub = sparse.coo_matrix((uvals, (urows, ucols)), shape=(ur, NVAR)).tocsr()
        c_obj = np.zeros(NVAR); c_obj[T] = 1
        res = linprog(c_obj, A_eq=A_eq, b_eq=np.array(b_eq), A_ub=A_ub, b_ub=np.array(b_ub),
                      bounds=[(0, 1)]*(NQ*NV)+[(0, None)], method="highs")
        assert res.status == 0, res.message
        return res.x[:NQ*NV].reshape(NQ, K, K), res.x[T]

    def U_of(Ps, v):
        E = np.zeros(len(brands))
        for q, (m, bs) in enumerate(data):
            e = Ps[q] @ v
            np.add.at(E, [bidx[b] for b in bs], e)
        tot = NQ*v.sum()
        ti = np.array(top)
        return np.max(np.abs(E[ti]/(tot*tgt[ti])-1))

    P_H, U_H = solve([vH])
    P_R, U_R = solve([vH]+vs_agents)

    def at_share(Ps, a):
        if a == 0: return U_of(Ps, vH)
        return max(U_of(Ps, (1-a)*vH + a*vA) for vA in vs_agents)

    lam_break = None; prev = None; sweep = []
    for a in np.linspace(0, sweep_hi, sweep_n):
        uh, urr = at_share(P_H, a), at_share(P_R, a)
        sweep.append((float(a), float(uh), float(urr)))
        if prev is not None and prev < 0 and uh-urr >= 0 and lam_break is None:
            lam_break = float(a)
        prev = uh-urr

    vs_all = [vH]+vs_agents
    sign_indet = {}
    for label, gset in (("top", top), ("all", [b for b in range(len(brands)) if M[b] > 0])):
        nind = 0; widths = []
        for b in gset:
            ds = []
            for v in vs_all:
                E = 0.0
                for q, (m, bs) in enumerate(data):
                    e = P_H[q] @ v
                    for i, bb in enumerate(bs):
                        if bidx[bb] == b: E += e[i]
                ds.append(E/(NQ*v.sum()*tgt[b])-1)
            widths.append(max(ds)-min(ds))
            if min(ds) < 0 < max(ds): nind += 1
        sign_indet[label] = (nind, len(gset), float(np.mean(widths)))

    out = {"U_H": float(U_H), "U_rob": float(U_R), "premium": float(U_R-U_H),
           "lambda_break": lam_break, "sweep": sweep, "sign_indet": sign_indet,
           "n_groups": len(brands), "top_groups": [str(brands[b]) for b in top],
           "tgt_top": [float(tgt[b]) for b in top]}
    print(f"U*_H={U_H:.4f}  U*_rob={U_R:.4f}  premium={U_R-U_H:.4f}  lambda_break={lam_break}")
    for label, (ni, n, w) in sign_indet.items():
        print(f"sign-indeterminate [{label}]: {ni}/{n} ({100*ni/n:.0f}%)  mean width={w:.3f}")
    return out
