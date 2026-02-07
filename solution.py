"""
BookRec Hackathon — Full Solution
==================================
Optimizes:  Score = 0.7 * mean(NDCG@20) + 0.3 * mean(Diversity@20)

Pipeline:
  1. SVD collaborative filtering (k=64)
  2. Rich feature engineering (~30 features)
  3. HistGradientBoostingRegressor for ranking
  4. Genre-diversity-aware MMR reranking
"""

import os, time, warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── paths ──────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(BASE, "materials")
OUT    = os.path.join(BASE, "submission.csv")

# ── hyper-parameters ───────────────────────────────────────────────────────────
SVD_K        = 64          # latent factors
NEG_RATIO    = 5           # negatives per positive for training
MMR_LAMBDA   = 0.60        # relevance weight in MMR (rest → diversity)
MMR_POOL     = 80          # top-N candidates fed into MMR greedy selector
TOP_K        = 20          # items to recommend

_EMPTY = frozenset()

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
def load_data():
    t0 = time.time()
    print("[1/8] Loading data …")
    inter   = pd.read_csv(os.path.join(DATA, "interactions.csv"))
    users   = pd.read_csv(os.path.join(DATA, "users.csv"))
    eds     = pd.read_csv(os.path.join(DATA, "editions.csv"))
    auth    = pd.read_csv(os.path.join(DATA, "authors.csv"))
    genres  = pd.read_csv(os.path.join(DATA, "genres.csv"))
    bg      = pd.read_csv(os.path.join(DATA, "book_genres.csv"))
    targets = pd.read_csv(os.path.join(DATA, "targets.csv"))
    cands   = pd.read_csv(os.path.join(DATA, "candidates.csv"))

    inter["event_ts"] = pd.to_datetime(inter["event_ts"])
    print(f"   interactions {len(inter):,}  |  targets {len(targets):,}  |  "
          f"candidates {len(cands):,}  ({time.time()-t0:.1f}s)")
    return inter, users, eds, auth, genres, bg, targets, cands

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  MAPPINGS
# ═══════════════════════════════════════════════════════════════════════════════
def build_maps(eds, bg):
    print("[2/8] Building mappings …")
    e2b  = eds.set_index("edition_id")["book_id"].to_dict()
    e2a  = eds.set_index("edition_id")["author_id"].to_dict()
    bgm  = bg.groupby("book_id")["genre_id"].apply(frozenset).to_dict()
    egm  = {eid: bgm.get(bid, _EMPTY) for eid, bid in e2b.items()}
    egs  = {eid: len(gs) for eid, gs in egm.items()}       # |G(edition)|
    print(f"   editions {len(e2b):,}  books-w-genres {len(bgm):,}")
    return e2b, e2a, bgm, egm, egs

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  SVD COLLABORATIVE FILTERING
# ═══════════════════════════════════════════════════════════════════════════════
def compute_svd(inter, targets, cands, k=SVD_K):
    t0 = time.time()
    print(f"[3/8] SVD (k={k}) …")

    tmp = inter.copy()
    tmp["rel"] = tmp["event_type"].map({1: 1.0, 2: 3.0})
    ui  = tmp.groupby(["user_id", "edition_id"])["rel"].max().reset_index()

    # --- recency weighting ---
    max_ts = inter["event_ts"].max()
    ts_map = inter.groupby(["user_id","edition_id"])["event_ts"].max().reset_index()
    ui = ui.merge(ts_map, on=["user_id","edition_id"], how="left")
    days_ago = (max_ts - ui["event_ts"]).dt.days.values.astype(np.float32)
    recency_w = np.exp(-days_ago / 180.0)          # half-life ≈ 6 months
    ui["val"] = np.log1p(ui["rel"].values) * recency_w

    all_u = sorted(set(targets["user_id"]) | set(inter["user_id"]))
    all_e = sorted(set(inter["edition_id"]) | set(cands["edition_id"]))
    u_enc = {u: i for i, u in enumerate(all_u)}
    e_enc = {e: i for i, e in enumerate(all_e)}

    mask = ui["user_id"].isin(u_enc) & ui["edition_id"].isin(e_enc)
    ui_v = ui[mask]
    rows = ui_v["user_id"].map(u_enc).values
    cols = ui_v["edition_id"].map(e_enc).values
    vals = ui_v["val"].values.astype(np.float32)

    R = csr_matrix((vals, (rows, cols)), shape=(len(u_enc), len(e_enc)))
    actual_k = min(k, min(R.shape) - 1)
    U, s, Vt = svds(R, k=actual_k)
    idx = np.argsort(-s); U = U[:, idx]; s = s[idx]; Vt = Vt[idx, :]

    uf = U * s[np.newaxis, :]        # user factors  (n_u, k)
    itf = Vt.T                        # item factors  (n_e, k)

    print(f"   matrix {R.shape}, nnz {R.nnz:,}  ({time.time()-t0:.1f}s)")
    return uf, itf, u_enc, e_enc

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════
def make_user_stats(inter, users, eds, bgm):
    """Aggregate user-level features."""
    max_ts = inter["event_ts"].max()
    uf = inter.groupby("user_id").agg(
        u_n=("event_type", "count"),
        u_reads=("event_type", lambda x: (x==2).sum()),
        u_wish=("event_type", lambda x: (x==1).sum()),
        u_avgr=("rating", "mean"),
        u_maxr=("rating", "max"),
        u_stdr=("rating", "std"),
        u_ned=("edition_id", "nunique"),
        u_last=("event_ts", "max"),
        u_first=("event_ts", "min"),
    ).reset_index()
    uf["u_rr"] = uf["u_reads"] / uf["u_n"].clip(1)
    uf["u_rec"] = (max_ts - uf["u_last"]).dt.days.astype(float)
    uf["u_span"] = (uf["u_last"] - uf["u_first"]).dt.days.clip(1).astype(float)
    uf["u_rate"] = uf["u_n"] / uf["u_span"] * 30.0
    uf.drop(columns=["u_last","u_first"], inplace=True)

    im = inter.merge(eds[["edition_id","book_id","author_id"]], on="edition_id", how="left")
    uf = uf.merge(im.groupby("user_id")["book_id"].nunique().reset_index(name="u_nb"), on="user_id", how="left")
    uf = uf.merge(im.groupby("user_id")["author_id"].nunique().reset_index(name="u_na"), on="user_id", how="left")
    uf = uf.merge(users[["user_id","gender","age"]], on="user_id", how="left")

    # genre profile
    ugp = {}   # user → {genre: weight}
    ugs = {}   # user → set(genres)
    for uid, grp in im.groupby("user_id"):
        gc = defaultdict(float)
        for et, bid in zip(grp["event_type"], grp["book_id"]):
            if pd.notna(bid):
                for g in bgm.get(int(bid), _EMPTY):
                    gc[g] += (3.0 if et == 2 else 1.0)
        ugp[uid] = dict(gc)
        ugs[uid] = set(gc.keys())

    rows = []
    for uid in uf["user_id"]:
        gc = ugp.get(uid, {})
        tot = sum(gc.values()) if gc else 0
        rows.append({
            "user_id": uid,
            "u_ng": len(gc),
            "u_tgs": max(gc.values())/max(tot,1) if gc else 0,
            "u_gent": -sum((v/tot)*np.log2(v/tot+1e-12) for v in gc.values()) if tot else 0,
        })
    uf = uf.merge(pd.DataFrame(rows), on="user_id", how="left")
    return uf, ugp, ugs, im

def make_item_stats(inter, eds, bgm, egm):
    """Aggregate item-level features."""
    max_ts = inter["event_ts"].max()
    itf = inter.groupby("edition_id").agg(
        i_pop=("event_type","count"),
        i_nu=("user_id","nunique"),
        i_reads=("event_type", lambda x:(x==2).sum()),
        i_wish=("event_type", lambda x:(x==1).sum()),
        i_avgr=("rating","mean"),
        i_stdr=("rating","std"),
        i_last=("event_ts","max"),
    ).reset_index()
    itf["i_rr"] = itf["i_reads"]/itf["i_pop"].clip(1)
    itf["i_rec"] = (max_ts - itf["i_last"]).dt.days.astype(float)
    itf.drop(columns=["i_last"], inplace=True)

    im = inter.merge(eds[["edition_id","book_id","author_id"]], on="edition_id", how="left")
    bp = im.groupby("book_id")["event_type"].count().reset_index(name="i_bpop")
    ap = im.groupby("author_id")["event_type"].count().reset_index(name="i_apop")

    em = eds[["edition_id","book_id","author_id","publication_year",
              "age_restriction","language_id"]].copy()
    em = em.merge(bp, on="book_id", how="left").fillna({"i_bpop":0})
    em = em.merge(ap, on="author_id", how="left").fillna({"i_apop":0})
    em["i_ng"] = em["edition_id"].map(lambda x: len(egm.get(x, _EMPTY)))
    return itf, em

def compute_pair_features(pairs, uf, itf, em,
                          ua_counts, ugs, ugp, ubs,
                          egm, egs, e2a,
                          user_f, item_f, u_enc, e_enc):
    """Compute features for (user_id, edition_id) pairs DataFrame."""
    df = pairs.copy()

    # --- SVD score -------------------------------------------------------
    svd = np.zeros(len(df), dtype=np.float32)
    uids = df["user_id"].values
    eids = df["edition_id"].values
    for i in range(len(df)):
        ui = u_enc.get(uids[i])
        ei = e_enc.get(eids[i])
        if ui is not None and ei is not None:
            svd[i] = np.dot(user_f[ui], item_f[ei])
    df["svd"] = svd

    # --- merge user / item stats -----------------------------------------
    df = df.merge(uf,  on="user_id",    how="left")
    df = df.merge(itf, on="edition_id", how="left")
    df = df.merge(em,  on="edition_id", how="left", suffixes=("","_em"))

    # --- author affinity --------------------------------------------------
    aid_col = df["edition_id"].map(e2a)
    keys = list(zip(df["user_id"], aid_col))
    df["ua_cnt"]  = [ua_counts.get(k, (0,0,0))[0] for k in keys]
    df["ua_rd"]   = [ua_counts.get(k, (0,0,0))[1] for k in keys]
    df["ua_avgr"] = [ua_counts.get(k, (0,0,np.nan))[2] for k in keys]

    # --- same book --------------------------------------------------------
    sb = np.zeros(len(df), dtype=np.int8)
    for i in range(len(df)):
        bid_map = em.set_index("edition_id")["book_id"] if False else None
        # faster: use e2b dict
        pass
    # rewritten efficiently below
    from functools import lru_cache
    e2b_local = em.set_index("edition_id")["book_id"].to_dict()
    for i in range(len(df)):
        uid = uids[i]
        bid = e2b_local.get(eids[i])
        if bid is not None and bid in ubs.get(uid, set()):
            sb[i] = 1
    df["ui_sb"] = sb

    # --- genre features ---------------------------------------------------
    jac  = np.zeros(len(df), dtype=np.float32)
    wov  = np.zeros(len(df), dtype=np.float32)
    newg = np.zeros(len(df), dtype=np.float32)
    for i in range(len(df)):
        uid = uids[i]; eid = eids[i]
        ug = ugs.get(uid)
        ig = egm.get(eid, _EMPTY)
        ig_sz = egs.get(eid, 0)
        if not ug or ig_sz == 0:
            newg[i] = 1.0 if ig_sz else 0.0
            continue
        inter_sz = len(ug & ig)
        union_sz = len(ug) + ig_sz - inter_sz
        jac[i]  = inter_sz / union_sz if union_sz else 0.0
        gp = ugp.get(uid, {})
        w  = sum(gp.get(g, 0) for g in ig)
        tw = sum(gp.values())
        wov[i]  = w / tw if tw else 0.0
        newg[i] = (ig_sz - inter_sz) / ig_sz
    df["ui_jac"]  = jac
    df["ui_wov"]  = wov
    df["ui_newg"] = newg

    df.fillna(0, inplace=True)
    return df


FEATURE_COLS = [
    "svd",
    # user
    "u_n","u_reads","u_wish","u_avgr","u_maxr","u_stdr","u_ned",
    "u_rr","u_rec","u_span","u_rate","u_nb","u_na","gender","age",
    "u_ng","u_tgs","u_gent",
    # item
    "i_pop","i_nu","i_reads","i_wish","i_avgr","i_stdr","i_rr","i_rec",
    "i_bpop","i_apop","i_ng","publication_year","age_restriction","language_id",
    # user–item
    "ua_cnt","ua_rd","ua_avgr","ui_sb","ui_jac","ui_wov","ui_newg",
]

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  TRAIN MODEL
# ═══════════════════════════════════════════════════════════════════════════════
def prepare_training_data(inter, eds, egm, egs, e2a, bgm,
                          user_f, item_f, u_enc, e_enc, users):
    """Create labelled training set from interactions."""
    t0 = time.time()
    print("[5/8] Preparing training data …")

    # --- relevance labels ---
    tmp = inter.copy()
    tmp["rel"] = tmp["event_type"].map({1: 1.0, 2: 3.0})
    pos = tmp.groupby(["user_id","edition_id"])["rel"].max().reset_index()
    pos.rename(columns={"rel":"label"}, inplace=True)

    # --- negative sampling ---
    all_eids = list(set(inter["edition_id"]))
    u_inter  = inter.groupby("user_id")["edition_id"].apply(set).to_dict()
    negs = []
    for uid, eset in u_inter.items():
        n = min(len(eset) * NEG_RATIO, 500)   # cap per user
        pool = [e for e in all_eids if e not in eset]
        if len(pool) > n:
            chosen = np.random.choice(pool, size=n, replace=False)
        else:
            chosen = pool
        for eid in chosen:
            negs.append({"user_id": uid, "edition_id": eid, "label": 0.0})
    neg_df = pd.DataFrame(negs)
    train = pd.concat([pos, neg_df], ignore_index=True).sample(frac=1, random_state=42)

    print(f"   pos {len(pos):,}  neg {len(neg_df):,}  total {len(train):,}")

    # --- features ---
    uf, ugp, ugs, im = make_user_stats(inter, users, eds, bgm)
    itf, em = make_item_stats(inter, eds, bgm, egm)

    # author affinity (user, author) → (count, reads, avg_rating)
    ua = im.groupby(["user_id","author_id"]).agg(
        c=("event_type","count"),
        r=("event_type", lambda x:(x==2).sum()),
        ar=("rating","mean"),
    ).reset_index()
    ua_dict = {(row.user_id, row.author_id): (row.c, row.r, row.ar) for row in ua.itertuples()}

    # user → set of book_ids
    ubs = im.groupby("user_id")["book_id"].apply(set).to_dict()

    print("   Computing features for training pairs …")
    train_feat = compute_pair_features(
        train[["user_id","edition_id"]],
        uf, itf, em, ua_dict, ugs, ugp, ubs,
        egm, egs, e2a,
        user_f, item_f, u_enc, e_enc,
    )
    train_feat["label"] = train["label"].values

    print(f"   Training data ready ({time.time()-t0:.1f}s)")
    return train_feat, uf, itf, em, ua_dict, ugs, ugp, ubs


def train_model(train_feat):
    t0 = time.time()
    print("[6/8] Training HistGradientBoosting …")
    X = train_feat[FEATURE_COLS].values.astype(np.float32)
    y = train_feat["label"].values.astype(np.float32)

    model = HistGradientBoostingRegressor(
        max_iter=500,
        max_depth=7,
        learning_rate=0.05,
        min_samples_leaf=30,
        l2_regularization=1.0,
        max_bins=255,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
    )
    model.fit(X, y)
    print(f"   n_iter={model.n_iter_}  ({time.time()-t0:.1f}s)")
    return model

# ═══════════════════════════════════════════════════════════════════════════════
# 6.  SCORE CANDIDATES
# ═══════════════════════════════════════════════════════════════════════════════
def score_candidates(cands, model,
                     uf, itf, em, ua_dict, ugs, ugp, ubs,
                     egm, egs, e2a,
                     user_f, item_f, u_enc, e_enc):
    t0 = time.time()
    print("[7/8] Scoring candidates …")
    cf = compute_pair_features(
        cands[["user_id","edition_id"]],
        uf, itf, em, ua_dict, ugs, ugp, ubs,
        egm, egs, e2a,
        user_f, item_f, u_enc, e_enc,
    )

    X = cf[FEATURE_COLS].values.astype(np.float32)
    cf["score"] = model.predict(X)
    print(f"   done ({time.time()-t0:.1f}s)")
    return cf[["user_id","edition_id","score"]]

# ═══════════════════════════════════════════════════════════════════════════════
# 7.  DIVERSITY-AWARE  MMR  RERANKING
# ═══════════════════════════════════════════════════════════════════════════════
def mmr_rerank(scored, egm, lam=MMR_LAMBDA, pool=MMR_POOL, top_k=TOP_K):
    """
    Greedy reranking that balances relevance and genre diversity.

    For each user:
      1.  Take the top-`pool` candidates by score.
      2.  Iteratively pick the item maximising:
              λ · norm_score  +  (1-λ) · diversity_gain
          where diversity_gain = 0.5·coverage_gain + 0.5·avg_jaccard_dist
    """
    t0 = time.time()
    print(f"[8/8] MMR reranking (λ={lam}, pool={pool}) …")
    results = []

    for uid, grp in scored.groupby("user_id"):
        grp_sorted = grp.nlargest(pool, "score")
        eids   = grp_sorted["edition_id"].values
        scores = grp_sorted["score"].values.astype(np.float64)

        # normalise scores to [0,1]
        smin, smax = scores.min(), scores.max()
        if smax > smin:
            snorm = (scores - smin) / (smax - smin)
        else:
            snorm = np.ones_like(scores)

        # genre sets for this user's pool
        gsets = [egm.get(int(e), _EMPTY) for e in eids]

        selected_idx = []
        covered = set()
        remaining = list(range(len(eids)))

        for _k in range(top_k):
            best_i   = -1
            best_val = -1e18

            for idx in remaining:
                rel = snorm[idx]

                ig = gsets[idx]
                ig_sz = len(ig)

                # coverage gain
                if ig_sz > 0:
                    cov_gain = len(ig - covered) / ig_sz
                else:
                    cov_gain = 0.0

                # ILD: avg Jaccard distance to already selected
                if selected_idx:
                    dist_sum = 0.0
                    for si in selected_idx:
                        sg = gsets[si]
                        union_sz = len(ig | sg)
                        if union_sz:
                            dist_sum += 1.0 - len(ig & sg) / union_sz
                        else:
                            dist_sum += 0.0
                    avg_dist = dist_sum / len(selected_idx)
                else:
                    avg_dist = 1.0

                div = 0.5 * cov_gain + 0.5 * avg_dist
                val = lam * rel + (1.0 - lam) * div

                if val > best_val:
                    best_val = val
                    best_i   = idx

            selected_idx.append(best_i)
            covered |= gsets[best_i]
            remaining.remove(best_i)

        for rank, idx in enumerate(selected_idx, 1):
            results.append((uid, int(eids[idx]), rank))

    out = pd.DataFrame(results, columns=["user_id","edition_id","rank"])
    print(f"   {len(out):,} rows  ({time.time()-t0:.1f}s)")
    return out

# ═══════════════════════════════════════════════════════════════════════════════
# 8.  VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════
def validate(sub, cands, targets):
    """Basic format checks."""
    t_users = set(targets["user_id"])
    s_users = set(sub["user_id"])
    assert s_users == t_users, f"User mismatch: {len(s_users)} vs {len(t_users)}"

    cand_map = cands.groupby("user_id")["edition_id"].apply(set).to_dict()
    for uid, grp in sub.groupby("user_id"):
        assert len(grp) == TOP_K, f"User {uid}: {len(grp)} rows (need {TOP_K})"
        assert set(grp["rank"]) == set(range(1, TOP_K+1)), f"User {uid}: bad ranks"
        assert grp["edition_id"].nunique() == TOP_K, f"User {uid}: duplicate editions"
        assert set(grp["edition_id"]).issubset(cand_map[uid]), \
            f"User {uid}: edition not in candidates"
    print("   ✓ Submission valid!")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    wall = time.time()
    print("=" * 62)
    print("  BookRec Hackathon — Solution")
    print("=" * 62)

    # 1) load
    inter, users, eds, auth, genres, bg, targets, cands = load_data()

    # 2) mappings
    e2b, e2a, bgm, egm, egs = build_maps(eds, bg)

    # 3) SVD
    user_f, item_f, u_enc, e_enc = compute_svd(inter, targets, cands)

    # 4-5) features + training
    train_feat, uf, itf, em, ua_dict, ugs, ugp, ubs = \
        prepare_training_data(inter, eds, egm, egs, e2a, bgm,
                              user_f, item_f, u_enc, e_enc, users)

    # 6) model
    model = train_model(train_feat)

    # 7) score candidates
    scored = score_candidates(cands, model,
                              uf, itf, em, ua_dict, ugs, ugp, ubs,
                              egm, egs, e2a,
                              user_f, item_f, u_enc, e_enc)

    # 8) rerank + save
    sub = mmr_rerank(scored, egm)
    sub.to_csv(OUT, index=False)
    print(f"\n   Saved → {OUT}")

    # 9) validate
    validate(sub, cands, targets)

    print(f"\n   Total wall time: {time.time()-wall:.0f}s")

if __name__ == "__main__":
    main()
