"""Build the compact derived dataset files in data/derived from the raw public datasets,
and verify that the derived-file loaders reproduce the raw-data workloads exactly
(canonical seeds and every bootstrap seed used in the paper).

Raw inputs (set RAW_DIR; defaults below are the layout used by the authors):
  ESCI   : esci_examples.parquet, esci_products.parquet        (amazon-science/esci-data, shopping_queries_dataset_*.parquet)
  TREC22 : trec2022/trec_2022_train_reldocs.jsonl, trec2022/trec_2022_articles_discrete.parquet   (TREC Fair Ranking 2022)
  Steam  : steam/.../recommendation/log/steam-base-mf/ranking_scores.npz, processed_dataset/steam/{iid2pid.json,steam.train}
           (FairDiverse toolkit, P-MMF Steam preprocessing; copied verbatim into data/derived/steam by the release build)
  LastFM : lastfm/LastFM360k-Le75.txt, lastfm/LastFM360k-MB-artists.txt   (LastFM-360K with MusicBrainz artist gender)

Derived outputs (data/derived):
  esci_es_us_small.parquet      ESCI rows (US, small_version==1, label in {E,S}) joined with brand, raw row order
  trec_topics_ge30.jsonl        TREC topics with >= 30 relevant docs (original order)
  trec_articles_subset.parquet  page_id / pred_qual / page_subcont_regions for every page the paper's draws can touch
  lastfm/user_counts.parquet    listening-event count per user (all users; needed to reproduce the user shuffle)
  lastfm/plays_subset.parquet   play rows of every user the paper's draws can select, raw row order
  lastfm/artists_subset.tsv     raw artist lines (id, name, gender counts) for artists of those users

Run: python prepare_data.py [--verify]      (verify compares derived vs raw loaders for all seeds; needs the raw files)
"""
import json, os, sys
import numpy as np
import pandas as pd
from config import DATA

RAW = os.environ.get("RAW_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw"))
K, NQ = 10, 200
SEEDS_BOOT = [1000 + b for b in range(20)]          # boot_domains.py (TREC22 / LastFM bootstrap workloads)

def build_esci():
    ex = pd.read_parquet(f"{RAW}/esci_examples.parquet",
                         columns=["query_id","product_id","esci_label","product_locale","small_version"])
    ex = ex[(ex.product_locale=="us")&(ex.small_version==1)&(ex.esci_label.isin(["E","S"]))]
    pr = pd.read_parquet(f"{RAW}/esci_products.parquet",
                         columns=["product_id","product_locale","product_brand"])
    pr = pr[pr.product_locale=="us"][["product_id","product_brand"]]
    ex = ex.merge(pr, on="product_id", how="left"); ex["product_brand"] = ex["product_brand"].fillna("UNKNOWN")
    ex[["query_id","product_id","esci_label","product_brand"]].to_parquet(f"{DATA}/esci_es_us_small.parquet", index=False)
    print("esci rows", len(ex))

def trec_draw_ids(topics, seed):
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(NQ):
        t = topics[rng.integers(len(topics))]
        draws.append(rng.choice(t["rel_docs"], size=3*K, replace=False))
    return np.unique(np.concatenate(draws))

def build_trec():
    import pyarrow.parquet as pq, pyarrow as pa
    topics = [json.loads(l) for l in open(f"{RAW}/trec2022/trec_2022_train_reldocs.jsonl")]
    topics = [t for t in topics if len(t["rel_docs"]) >= 3*K]
    with open(f"{DATA}/trec_topics_ge30.jsonl", "w") as f:
        for t in topics: f.write(json.dumps(t)+"\n")
    needed = np.unique(np.concatenate([trec_draw_ids(topics, s) for s in [7]+SEEDS_BOOT]))
    tbl = pq.read_table(f"{RAW}/trec2022/trec_2022_articles_discrete.parquet",
                        columns=["page_id", "pred_qual", "page_subcont_regions"])
    ids = tbl["page_id"].to_numpy()
    mask = np.isin(ids, needed)
    sub = tbl.filter(pa.array(mask))
    pq.write_table(sub, f"{DATA}/trec_articles_subset.parquet")
    print("trec topics", len(topics), "needed pages", len(needed), "kept rows", sub.num_rows)

def lastfm_users(counts, seed):
    rng = np.random.default_rng(seed)
    users = counts[counts >= 2*K].index.to_numpy().copy()
    rng.shuffle(users)
    return users[:NQ*2]

def build_lastfm():
    os.makedirs(f"{DATA}/lastfm", exist_ok=True)
    df = pd.read_csv(f"{RAW}/lastfm/LastFM360k-Le75.txt")
    counts = df.groupby("userid").size()
    counts.rename("n").reset_index().to_parquet(f"{DATA}/lastfm/user_counts.parquet", index=False)
    sel = set()
    for s in [21]+SEEDS_BOOT: sel |= set(lastfm_users(counts, s).tolist())
    sub = df[df.userid.isin(sel)]
    sub.to_parquet(f"{DATA}/lastfm/plays_subset.parquet", index=False)
    aids = set(sub.artistid.unique().tolist())
    with open(f"{DATA}/lastfm/artists_subset.tsv", "w", encoding="utf-8") as out:
        for line in open(f"{RAW}/lastfm/LastFM360k-MB-artists.txt", encoding="utf-8", errors="replace"):
            p = line.rstrip("\n").split("\t")
            try: aid = int(p[0])
            except ValueError: continue
            if aid in aids: out.write(line if line.endswith("\n") else line+"\n")
    print("lastfm users kept", len(sel), "rows", len(sub), "artists", len(aids))

def verify():
    """Compare derived-file loaders (this release) with raw-data loaders (original code) for every seed."""
    import loaders
    src = open("loaders.py").read()
    raw_src = open("loaders_raw_reference.py").read()
    def make(module_src, fn_name, seed_default, seed, ns_base):
        a = module_src.index(f"def {fn_name}(")
        lines = module_src[a:].split("\n"); body_lines = [lines[0]]
        for l in lines[1:]:
            if l and not l[0].isspace(): break          # next top-level statement ends the function
            body_lines.append(l)
        body = "\n".join(body_lines).replace(f"np.random.default_rng({seed_default})", f"np.random.default_rng({seed})")
        ns = dict(ns_base); exec(body, ns); return ns[fn_name]
    raw_ns = {"json": json, "np": np, "K": K, "NQ": NQ, "RAW": RAW}
    def same(d1, d2):
        return len(d1)==len(d2) and all(np.allclose(a[0], b[0]) and list(a[1])==list(b[1]) for a, b in zip(d1, d2))
    checks = [("esci_data", 0, [0]), ("trec_data", 7, [7]+SEEDS_BOOT), ("steam_data", 11, [11]), ("lastfm_data", 21, [21]+SEEDS_BOOT)]
    for fn, sd, seeds in checks:
        for s in seeds:
            d_new = make(src, fn, sd, s, dict(vars(loaders)))()
            d_raw = make(raw_src, fn, sd, s, raw_ns)()
            assert same(d_new, d_raw), (fn, s)
        print(f"{fn}: identical for seeds {seeds[:1]}{'+20 bootstrap' if len(seeds)>1 else ''}")
    # ESCI table used by the bootstrap scripts (main_frontier / triad_window / main_fig4_all / mixcal_merit)
    ex = loaders.esci_table()
    exr = pd.read_parquet(f"{RAW}/esci_examples.parquet", columns=["query_id","product_id","esci_label","product_locale","small_version"])
    exr = exr[(exr.product_locale=="us")&(exr.small_version==1)&(exr.esci_label.isin(["E","S"]))]
    pr = pd.read_parquet(f"{RAW}/esci_products.parquet", columns=["product_id","product_locale","product_brand"])
    pr = pr[pr.product_locale=="us"][["product_id","product_brand"]]
    exr = exr.merge(pr, on="product_id", how="left"); exr["product_brand"] = exr["product_brand"].fillna("UNKNOWN")
    assert (ex[["query_id","product_id","esci_label","product_brand"]].to_numpy() == exr[["query_id","product_id","esci_label","product_brand"]].to_numpy()).all()
    print("esci_table: identical")

if __name__ == "__main__":
    if "--verify" not in sys.argv:
        os.makedirs(DATA, exist_ok=True)
        build_esci(); build_trec(); build_lastfm()
        print("derived files written to", DATA)
    else:
        verify()
