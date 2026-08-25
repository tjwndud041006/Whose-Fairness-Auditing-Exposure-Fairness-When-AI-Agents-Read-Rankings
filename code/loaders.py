"""Domain loaders for the four testbeds. Each returns a list of
(merit K-array, provider K-array) per query, with the EXACT seeds/protocols of the
original port scripts (esci: rng(0) e3c protocol; trec: rng(7); steam: rng(11);
lastfm: rng(21)) so all numbers reproduce.

This release version reads the compact derived files in data/derived (built from the
raw public datasets by prepare_data.py). The selection logic is unchanged; only the
file reads differ. prepare_data.py verifies that the derived-file loaders return
exactly the same workloads as the raw-data loaders.
"""
import json
import numpy as np
from config import DATA

K = 10
NQ = 200

def esci_table():
    """ESCI (US, small version, E/S labels) merged with product brand; row order preserved from the raw build."""
    import pandas as pd
    return pd.read_parquet(f"{DATA}/esci_es_us_small.parquet")

def esci_data():
    import pandas as pd
    rng = np.random.default_rng(0)
    GAIN = {"E": 1.0, "S": 0.5}
    ex = esci_table()
    cnt = ex.groupby("query_id").size()
    all_q = cnt[cnt>=K].index.to_numpy().copy()
    rng.shuffle(all_q); qs = all_q[:NQ]
    data = []
    for qi in qs:
        g = ex[ex.query_id==qi]
        sub = g.sample(K, random_state=int(qi)%2**31) if len(g)>K else g
        data.append((np.array([GAIN[l] for l in sub.esci_label]), sub.product_brand.to_numpy()))
    return data

def trec_data():
    import pyarrow.parquet as pq
    rng = np.random.default_rng(7)
    topics = [json.loads(l) for l in open(f"{DATA}/trec_topics_ge30.jsonl")]
    topics = [t for t in topics if len(t["rel_docs"]) >= 3*K]
    draws = []
    for _ in range(NQ):
        t = topics[rng.integers(len(topics))]
        draws.append(rng.choice(t["rel_docs"], size=3*K, replace=False))
    needed = np.unique(np.concatenate(draws))
    tbl = pq.read_table(f"{DATA}/trec_articles_subset.parquet",
                        columns=["page_id", "pred_qual", "page_subcont_regions"])
    ids = tbl["page_id"].to_numpy()
    order = np.argsort(ids); ids_sorted = ids[order]
    pos = np.searchsorted(ids_sorted, needed)
    ok = (pos < len(ids_sorted))
    ok[ok] &= (ids_sorted[pos[ok]] == needed[ok])
    lookup = {int(i): int(order[p]) for i, p, o in zip(needed, pos, ok) if o}
    qual = tbl["pred_qual"].to_numpy()
    regions_col = tbl["page_subcont_regions"]
    def region_of(row):
        lst = regions_col[row].as_py()
        return lst[0] if lst else "UNK"
    data = []
    for d in draws:
        rows = [lookup[int(x)] for x in d if int(x) in lookup][:K]
        m = np.array([max(qual[r], 0.01) for r in rows])
        gs = np.array([region_of(r) for r in rows])
        data.append((m, gs))
    return data

def steam_data():
    import scipy.sparse as sp
    rng = np.random.default_rng(11)
    BASE = f"{DATA}/steam"
    d = np.load(f"{BASE}/ranking_scores.npz")
    S = sp.coo_matrix((d["data"], (d["row"], d["col"])), shape=tuple(d["shape"])).tocsr()
    iid2pid = {int(k): int(v) for k, v in
               json.load(open(f"{BASE}/iid2pid.json")).items()}
    users = [u for u in range(S.shape[0]) if S[u].nnz >= K]
    rng.shuffle(users)
    data = []
    for u in users[:NQ]:
        row = S[u]
        items = row.indices; scores = row.data
        topk = items[np.argsort(-scores)[:K]]
        sc = np.sort(scores)[::-1][:K]
        m = np.maximum(sc, 1e-3)
        gs = np.array([f"pub{iid2pid.get(int(i), -1)}" for i in topk])
        data.append((m, gs))
    return data

def lastfm_data():
    import pandas as pd
    rng = np.random.default_rng(21)
    D = f"{DATA}/lastfm"
    gender = {}
    for line in open(f"{D}/artists_subset.tsv", encoding="utf-8", errors="replace"):
        p = line.rstrip("\n").split("\t")
        if len(p) < 3: continue
        try: aid = int(p[0])
        except ValueError: continue
        c = [int(x) for x in p[2].split("/")]
        if c[1] > 0 or c[2] > 0:
            gender[aid] = "male" if c[1] >= c[2] else "female"
    counts = pd.read_parquet(f"{D}/user_counts.parquet").set_index("userid")["n"]
    users = counts[counts >= 2*K].index.to_numpy().copy()
    rng.shuffle(users)
    sel = set(users[:NQ*2].tolist())
    df = pd.read_parquet(f"{D}/plays_subset.parquet")
    df = df[df.userid.isin(sel)]
    names = {}
    for line in open(f"{D}/artists_subset.tsv", encoding="utf-8", errors="replace"):
        p = line.rstrip("\n").split("\t")
        try: names[int(p[0])] = p[1]
        except (ValueError, IndexError): continue
    data = []
    for u in users[:NQ*2]:
        if len(data) >= NQ: break
        g = df[df.userid == u].nlargest(K, "playcount")
        if len(g) < K: continue
        m = np.log1p(g.playcount.to_numpy(dtype=float))
        aids = g.artistid.to_numpy()
        data.append((m, np.array([names.get(int(a), f"a{a}") for a in aids])))
    return data

LOADERS = {"ESCI": esci_data, "TREC22": trec_data, "Steam": steam_data, "LastFM": lastfm_data}
