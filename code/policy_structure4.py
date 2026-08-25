"""Two things at once.
(1) LP degeneracy fix: canonical (lexicographic) solve — stage 1 minimizes worst-case
    U as before; stage 2 fixes U<=U*+tol and minimizes the mean absolute deviation of
    the top providers under vH. Recompute sign-ind / interval widths under the
    canonical P_H and P_R for all four domains (Table 2 sensitivity check).
(2) Amortization-slack hypothesis: providers appearing in fewer queries have less
    cross-query slack, hence less width reduction under the robust policy.
    Per-provider (n_queries, width_H, width_R) over ALL providers; Spearman rho
    between n_queries and relative width reduction.
Run: python policy_structure4.py
"""
import json
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from scipy.stats import spearmanr
from port_core import load_curves
from loaders import LOADERS, K

def build_canonical(data, ntop=15, vH_override=None):
    NQ = len(data)
    vH, agent_curves = load_curves(K)
    if vH_override is not None:
        vH = np.asarray(vH_override, dtype=float); vH = vH/vH.sum()
    brands = sorted(set(b for _, bs in data for b in bs)); bidx = {b: i for i, b in enumerate(brands)}
    M = np.zeros(len(brands))
    for m, bs in data:
        for mi, b in zip(m, bs): M[bidx[b]] += mi
    top = list(np.argsort(-M)[:min(ntop, len(brands))]); tgt = M/M.sum()
    NV = K*K; T = NQ*NV

    def eq_block(ncols):
        rows, cols, vals, b_eq = [], [], [], []; r = 0
        for q in range(NQ):
            for i in range(K):
                for k in range(K): rows.append(r); cols.append(q*NV+i*K+k); vals.append(1.0)
                b_eq.append(1.0); r += 1
            for k in range(K):
                for i in range(K): rows.append(r); cols.append(q*NV+i*K+k); vals.append(1.0)
                b_eq.append(1.0); r += 1
        return sparse.coo_matrix((vals, (rows, cols)), shape=(r, ncols)).tocsr(), np.array(b_eq)

    def dev_rows(v, bvar_col, ncols, start_r):
        """sign*(E_b) - tot*tgt_b * x[bvar_col(b)] <= sign*tot*tgt_b for b in top"""
        rows, cols, vals, b_ub = [], [], [], []; r = start_r
        tot = NQ*v.sum()
        for jb, b in enumerate(top):
            for sign in (1.0, -1.0):
                for q, (m, bs) in enumerate(data):
                    for i, bb in enumerate(bs):
                        if bidx[bb] == b:
                            for k in range(K): rows.append(r); cols.append(q*NV+i*K+k); vals.append(sign*v[k])
                rows.append(r); cols.append(bvar_col(jb)); vals.append(-tot*tgt[b])
                b_ub.append(sign*tot*tgt[b]); r += 1
        return rows, cols, vals, b_ub, r

    def solve(cert_curves):
        # stage 1: min t
        ncols = T+1
        A_eq, b_eq = eq_block(ncols)
        rows, cols, vals, b_ub = [], [], [], []; r = 0
        for v in cert_curves:
            rr, cc, vv, bb, r = dev_rows(v, lambda jb: T, ncols, r)
            rows += rr; cols += cc; vals += vv; b_ub += bb
        A_ub = sparse.coo_matrix((vals, (rows, cols)), shape=(r, ncols)).tocsr()
        c1 = np.zeros(ncols); c1[T] = 1
        res = linprog(c1, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=np.array(b_ub),
                      bounds=[(0, 1)]*T+[(0, None)], method="highs")
        assert res.status == 0, res.message
        Ustar = res.x[T]
        # stage 2: fix t <= U*+tol, minimize sum of |dev under vH| via aux d_b
        nd = len(top); ncols2 = T+1+nd
        A_eq2, b_eq2 = eq_block(ncols2)
        rows, cols, vals, b_ub2 = [], [], [], []; r = 0
        for v in cert_curves:
            rr, cc, vv, bb, r = dev_rows(v, lambda jb: T, ncols2, r)
            rows += rr; cols += cc; vals += vv; b_ub2 += bb
        rr, cc, vv, bb, r = dev_rows(vH, lambda jb: T+1+jb, ncols2, r)
        rows += rr; cols += cc; vals += vv; b_ub2 += bb
        rows.append(r); cols.append(T); vals.append(1.0); b_ub2.append(Ustar+1e-8); r += 1
        A_ub2 = sparse.coo_matrix((vals, (rows, cols)), shape=(r, ncols2)).tocsr()
        c2 = np.zeros(ncols2); c2[T+1:] = 1.0
        res2 = linprog(c2, A_eq=A_eq2, b_eq=b_eq2, A_ub=A_ub2, b_ub=np.array(b_ub2),
                       bounds=[(0, 1)]*T+[(0, None)]*(1+nd), method="highs")
        assert res2.status == 0, res2.message
        return res2.x[:T].reshape(NQ, K, K), float(Ustar)

    return {"solve": solve, "vH": vH, "agents": list(agent_curves.values()),
            "brands": brands, "bidx": bidx, "top": top, "tgt": tgt, "M": M}

def brand_exposures(Ps, data, ctx, v):
    E = np.zeros(len(ctx["brands"]))
    for q, (m, bs) in enumerate(data):
        e = Ps[q] @ v
        np.add.at(E, [ctx["bidx"][b] for b in bs], e)
    return E

def widths_all(Ps, data, ctx, curves, group):
    NQ = len(data)
    devs = []
    for v in curves:
        E = brand_exposures(Ps, data, ctx, v)
        devs.append(E/(NQ*v.sum()*ctx["tgt"])-1)
    devs = np.array(devs)   # ncurves x nbrands
    lo = devs.min(axis=0)[group]; hi = devs.max(axis=0)[group]
    return lo, hi

if __name__ == "__main__":
  out = {}
  for name, loader in LOADERS.items():
      data = loader()
      ctx = build_canonical(data, ntop=15)
      vH = ctx["vH"]; agents = ctx["agents"]; curves = [vH]+agents
      P_H, U_H = ctx["solve"]([vH])
      P_R, U_R = ctx["solve"](curves)
      top = np.array(ctx["top"])
      loH, hiH = widths_all(P_H, data, ctx, curves, top)
      loR, hiR = widths_all(P_R, data, ctx, curves, top)
      si_H = int(np.sum((loH < 0) & (hiH > 0))); si_R = int(np.sum((loR < 0) & (hiR > 0)))
      wH = hiH-loH; wR = hiR-loR
      # slack hypothesis over ALL providers with positive merit
      allg = np.array([b for b in range(len(ctx["brands"])) if ctx["M"][b] > 0])
      loHa, hiHa = widths_all(P_H, data, ctx, curves, allg)
      loRa, hiRa = widths_all(P_R, data, ctx, curves, allg)
      wHa = hiHa-loHa; wRa = hiRa-loRa
      nq = np.zeros(len(ctx["brands"]))
      for q, (m, bs) in enumerate(data):
          for b in set(bs): nq[ctx["bidx"][b]] += 1
      nqa = nq[allg]
      mask = wHa > 1e-9
      red_rel = (wHa[mask]-wRa[mask])/wHa[mask]
      rho_rel, p_rel = spearmanr(nqa[mask], red_rel)
      rho_abs, p_abs = spearmanr(nqa[mask], wHa[mask]-wRa[mask])
      out[name] = {"U_H": U_H, "U_R": U_R,
                   "canon_sign_ind_H": si_H, "canon_sign_ind_R": si_R,
                   "canon_width_H": float(wH.mean()), "canon_width_R": float(wR.mean()),
                   "n_providers_all": int(len(allg)), "n_with_width": int(mask.sum()),
                   "spearman_nq_vs_rel_reduction": [float(rho_rel), float(p_rel)],
                   "spearman_nq_vs_abs_reduction": [float(rho_abs), float(p_abs)]}
      print(f"{name:8s} U_H={U_H:.4f} U_R={U_R:.4f} | canon sign-ind H={si_H}/15 R={si_R}/15"
            f" width H={wH.mean():.3f} R={wR.mean():.3f}")
      print(f"         slack: {len(allg)} providers ({int(mask.sum())} with width>0), "
            f"rho(nq, rel-reduction)={rho_rel:+.3f} (p={p_rel:.1e}) abs={rho_abs:+.3f} (p={p_abs:.1e})")
  json.dump(out, open("policy_structure4.json", "w"), indent=1)
  print("saved policy_structure4.json")
