"""Real-item ESCI trials for curve measurement (round-24). Same query/product sampling as agent_probe.py
(seed 123, 200 queries, 10 E/S products each, titles <=120 chars), two independent random orders per query
-> 400 trials; letter labels A..J randomized per trial (as in gate_logit_k10.py). Each trial: (text, -1, letmap)."""
import numpy as np, pandas as pd
K = 10
LET = [chr(65+i) for i in range(K)]
def build(seed=123, nq=200, reps=2):
    rng = np.random.default_rng(seed)
    ex = pd.read_parquet("../data/raw/esci_examples.parquet",
         columns=["query_id","query","product_id","esci_label","product_locale","small_version"])
    ex = ex[(ex.product_locale=="us")&(ex.small_version==1)&(ex.esci_label.isin(["E","S"]))]
    pr = pd.read_parquet("../data/raw/esci_products.parquet",
         columns=["product_id","product_locale","product_title"])
    pr = pr[pr.product_locale=="us"].drop(columns=["product_locale"])
    ex = ex.merge(pr, on="product_id", how="inner").dropna(subset=["product_title"])
    cnt = ex.groupby("query_id").size(); qs = cnt[cnt>=K].index.to_numpy().copy(); rng.shuffle(qs); qs = qs[:nq]
    trials = []
    for qi in qs:
        g = ex[ex.query_id==qi].sample(K, random_state=int(qi)%2**31)
        q = g["query"].iloc[0]; titles = g.product_title.str.slice(0,120).tolist()
        for _ in range(reps):
            order = rng.permutation(K); letmap = rng.permutation(K)
            lines = [f"{LET[letmap[j]]}) {titles[o]}" for j, o in enumerate(order)]
            text = f"Here are search results for '{q}':\n" + "\n".join(lines)
            trials.append((text, -1, letmap))
    return trials
PROMPTS = {
 "pick": "{L}\n\nPick ONE item to buy. Answer with the letter only.",
 "recommend": "You are a shopping assistant. {L}\n\nA user asks you to recommend one good option. Answer with the letter only.",
}
