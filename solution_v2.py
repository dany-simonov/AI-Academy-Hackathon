"""
BookRec Hackathon — Solution v2 (temporal training + proper evaluation)
======================================================================
Score = 0.7 * mean(NDCG@20) + 0.3 * mean(Diversity@20)

Key improvements over v1:
  - Temporal training: features from T1, labels from T2
  - Proper local evaluation with custom validation candidates
  - Multiple SVD signals (base + recency-weighted)
  - Trending-item features
  - Fine-tuned MMR reranking
"""

import os, sys, time, math, warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore")
np.random.seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "materials")
OUT  = os.path.join(BASE, "submission.csv")

TOP_K      = 20
SVD_K      = 64
NEG_RATIO  = 5
MMR_POOL   = 80
_EMPTY     = frozenset()

# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    t0 = time.time()
    inter   = pd.read_csv(os.path.join(DATA, "interactions.csv"))
    users   = pd.read_csv(os.path.join(DATA, "users.csv"))
    eds     = pd.read_csv(os.path.join(DATA, "editions.csv"))
    bg      = pd.read_csv(os.path.join(DATA, "book_genres.csv"))
    targets = pd.read_csv(os.path.join(DATA, "targets.csv"))
    cands   = pd.read_csv(os.path.join(DATA, "candidates.csv"))
    inter["event_ts"] = pd.to_datetime(inter["event_ts"])
    print(f"[load] {len(inter):,} inter  "
          f"{len(targets):,} users  {len(cands):,} cands  ({time.time()-t0:.1f}s)")
    return inter, users, eds, bg, targets, cands

# ─────────────────────────────────────────────────────────────────────────────
#  MAPPINGS
# ─────────────────────────────────────────────────────────────────────────────
def build_maps(eds, bg):
    e2b = eds.set_index("edition_id")["book_id"].to_dict()
    e2a = eds.set_index("edition_id")["author_id"].to_dict()
    bgm = bg.groupby("book_id")["genre_id"].apply(frozenset).to_dict()
    egm = {eid: bgm.get(bid, _EMPTY) for eid, bid in e2b.items()}
    egs = {eid: len(g) for eid, g in egm.items()}
    return e2b, e2a, bgm, egm, egs

# ─────────────────────────────────────────────────────────────────────────────
#  SVD  (recency-weighted)
# ─────────────────────────────────────────────────────────────────────────────
def do_svd(inter, all_users, all_editions, k=SVD_K, half_life=180):
    """SVD on interaction matrix with recency weighting."""
    tmp = inter.copy()
    tmp["rel"] = tmp["event_type"].map({1: 1.0, 2: 3.0})
    ui = tmp.groupby(["user_id","edition_id"]).agg(
        rel=("rel","max"), ts=("event_ts","max")).reset_index()
    ref = inter["event_ts"].max()
    days = (ref - ui["ts"]).dt.days.values.astype(np.float32)
    ui["val"] = np.log1p(ui["rel"].values) * np.exp(-days / half_life)

    u_enc = {u: i for i, u in enumerate(all_users)}
    e_enc = {e: i for i, e in enumerate(all_editions)}
    m = ui["user_id"].isin(u_enc) & ui["edition_id"].isin(e_enc)
    uiv = ui[m]
    R = csr_matrix(
        (uiv["val"].values.astype(np.float32),
         (uiv["user_id"].map(u_enc).values, uiv["edition_id"].map(e_enc).values)),
        shape=(len(u_enc), len(e_enc)))
    ak = min(k, min(R.shape) - 1)
    U, s, Vt = svds(R, k=ak)
    idx = np.argsort(-s); U = U[:, idx]; s = s[idx]; Vt = Vt[idx, :]
    print(f"   SVD {R.shape} nnz={R.nnz:,} k={ak}")
    return U * s[np.newaxis, :], Vt.T, u_enc, e_enc

# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE BUILDER  (reusable for any interaction set)
# ─────────────────────────────────────────────────────────────────────────────
class FeatureBuilder:
    """Precomputes user / item / content stats from a given interaction set."""

    COLS = [
        "svd",
        # user
        "u_n","u_reads","u_wish","u_avgr","u_maxr","u_stdr","u_ned",
        "u_rr","u_rec","u_span","u_rate","u_nb","u_na","gender","age",
        "u_ng","u_tgs","u_gent",
        # item
        "i_pop","i_nu","i_reads","i_wish","i_avgr","i_stdr","i_rr","i_rec",
        "i_bpop","i_apop","i_ng","publication_year","age_restriction","language_id",
        # item trending
        "i_trend",
        # user–item
        "ua_cnt","ua_rd","ua_avgr_feat","ui_sb","ui_jac","ui_wov","ui_newg",
    ]

    def __init__(self, inter, users, eds, e2b, e2a, bgm, egm, egs,
                 user_f, item_f, u_enc, e_enc):
        self.egm = egm; self.egs = egs; self.e2a = e2a
        self.user_f = user_f; self.item_f = item_f
        self.u_enc = u_enc; self.e_enc = e_enc
        max_ts = inter["event_ts"].max()

        # ---- user stats ----
        uf = inter.groupby("user_id").agg(
            u_n=("event_type","count"),
            u_reads=("event_type", lambda x:(x==2).sum()),
            u_wish=("event_type", lambda x:(x==1).sum()),
            u_avgr=("rating","mean"), u_maxr=("rating","max"), u_stdr=("rating","std"),
            u_ned=("edition_id","nunique"),
            u_last=("event_ts","max"), u_first=("event_ts","min"),
        ).reset_index()
        uf["u_rr"]   = uf["u_reads"]/uf["u_n"].clip(1)
        uf["u_rec"]  = (max_ts-uf["u_last"]).dt.days.astype(float)
        uf["u_span"] = (uf["u_last"]-uf["u_first"]).dt.days.clip(1).astype(float)
        uf["u_rate"] = uf["u_n"]/uf["u_span"]*30
        uf.drop(columns=["u_last","u_first"], inplace=True)

        im = inter.merge(eds[["edition_id","book_id","author_id"]],
                         on="edition_id", how="left")
        uf = uf.merge(im.groupby("user_id")["book_id"].nunique()
                      .reset_index(name="u_nb"), on="user_id", how="left")
        uf = uf.merge(im.groupby("user_id")["author_id"].nunique()
                      .reset_index(name="u_na"), on="user_id", how="left")
        uf = uf.merge(users[["user_id","gender","age"]], on="user_id", how="left")

        # genre profile
        ugp = {}; ugs = {}
        for uid, grp in im.groupby("user_id"):
            gc = defaultdict(float)
            for et, bid in zip(grp["event_type"], grp["book_id"]):
                if pd.notna(bid):
                    for g in bgm.get(int(bid), _EMPTY):
                        gc[g] += (3 if et == 2 else 1)
            ugp[uid] = dict(gc); ugs[uid] = set(gc.keys())
        rows = []
        for uid in uf["user_id"]:
            gc = ugp.get(uid, {}); tot = sum(gc.values()) if gc else 0
            rows.append({"user_id": uid, "u_ng": len(gc),
                "u_tgs": max(gc.values())/max(tot,1) if gc else 0,
                "u_gent": -sum((v/tot)*np.log2(v/tot+1e-12)
                               for v in gc.values()) if tot else 0})
        uf = uf.merge(pd.DataFrame(rows), on="user_id", how="left")
        self.uf = uf; self.ugp = ugp; self.ugs = ugs

        # author affinity
        ua = im.groupby(["user_id","author_id"]).agg(
            c=("event_type","count"),
            r=("event_type", lambda x:(x==2).sum()),
            ar=("rating","mean")).reset_index()
        self.ua_dict = {(row.user_id, row.author_id): (row.c, row.r, row.ar)
                        for row in ua.itertuples()}

        # user books
        self.ubs = im.groupby("user_id")["book_id"].apply(set).to_dict()

        # ---- item stats ----
        itf = inter.groupby("edition_id").agg(
            i_pop=("event_type","count"), i_nu=("user_id","nunique"),
            i_reads=("event_type", lambda x:(x==2).sum()),
            i_wish=("event_type", lambda x:(x==1).sum()),
            i_avgr=("rating","mean"), i_stdr=("rating","std"),
            i_last=("event_ts","max"),
        ).reset_index()
        itf["i_rr"]  = itf["i_reads"]/itf["i_pop"].clip(1)
        itf["i_rec"] = (max_ts - itf["i_last"]).dt.days.astype(float)
        itf.drop(columns=["i_last"], inplace=True)

        # trending: interactions in last 7 days / last 30 days
        recent7  = inter[inter["event_ts"] >= max_ts - pd.Timedelta(days=7)]
        recent30 = inter[inter["event_ts"] >= max_ts - pd.Timedelta(days=30)]
        cnt7  = recent7.groupby("edition_id")["event_type"].count().reset_index(name="c7")
        cnt30 = recent30.groupby("edition_id")["event_type"].count().reset_index(name="c30")
        trend = cnt7.merge(cnt30, on="edition_id", how="outer").fillna(0)
        trend["i_trend"] = trend["c7"] / trend["c30"].clip(1)
        itf = itf.merge(trend[["edition_id","i_trend"]], on="edition_id", how="left")
        itf["i_trend"] = itf["i_trend"].fillna(0)

        bp = im.groupby("book_id")["event_type"].count().reset_index(name="i_bpop")
        ap = im.groupby("author_id")["event_type"].count().reset_index(name="i_apop")
        em = eds[["edition_id","book_id","author_id",
                  "publication_year","age_restriction","language_id"]].copy()
        em = em.merge(bp, on="book_id", how="left").fillna({"i_bpop": 0})
        em = em.merge(ap, on="author_id", how="left").fillna({"i_apop": 0})
        em["i_ng"] = em["edition_id"].map(lambda x: len(egm.get(x, _EMPTY)))
        self.itf = itf; self.em = em
        self.e2b_local = em.set_index("edition_id")["book_id"].to_dict()

    # ────────────────────── featurize pairs ──────────────────────
    def featurize(self, pairs_df):
        """Add features to a (user_id, edition_id) DataFrame."""
        df = pairs_df.copy()
        uids = df["user_id"].values; eids = df["edition_id"].values
        N = len(df)

        # SVD score
        svd = np.zeros(N, np.float32)
        for i in range(N):
            ui, ei = self.u_enc.get(uids[i]), self.e_enc.get(eids[i])
            if ui is not None and ei is not None:
                svd[i] = np.dot(self.user_f[ui], self.item_f[ei])
        df["svd"] = svd

        # merge stats
        df = df.merge(self.uf,  on="user_id",    how="left")
        df = df.merge(self.itf, on="edition_id", how="left")
        df = df.merge(self.em,  on="edition_id", how="left", suffixes=("","_em"))

        # author affinity
        aids = [self.e2a.get(e) for e in eids]
        keys = list(zip(uids, aids))
        df["ua_cnt"]       = [self.ua_dict.get(k, (0,0,0))[0] for k in keys]
        df["ua_rd"]        = [self.ua_dict.get(k, (0,0,0))[1] for k in keys]
        df["ua_avgr_feat"] = [self.ua_dict.get(k, (0,0,np.nan))[2] for k in keys]

        # same book
        sb = np.zeros(N, np.int8)
        for i in range(N):
            bid = self.e2b_local.get(eids[i])
            if bid is not None and bid in self.ubs.get(uids[i], set()):
                sb[i] = 1
        df["ui_sb"] = sb

        # genre features
        jac  = np.zeros(N, np.float32)
        wov  = np.zeros(N, np.float32)
        newg = np.zeros(N, np.float32)
        for i in range(N):
            ug = self.ugs.get(uids[i])
            ig = self.egm.get(eids[i], _EMPTY); ig_sz = self.egs.get(eids[i], 0)
            if not ug or ig_sz == 0:
                newg[i] = 1.0 if ig_sz else 0.0; continue
            isz = len(ug & ig); usz = len(ug) + ig_sz - isz
            jac[i]  = isz / usz if usz else 0
            gp = self.ugp.get(uids[i], {})
            w  = sum(gp.get(g, 0) for g in ig)
            tw = sum(gp.values())
            wov[i]  = w / tw if tw else 0
            newg[i] = (ig_sz - isz) / ig_sz
        df["ui_jac"] = jac; df["ui_wov"] = wov; df["ui_newg"] = newg

        df.fillna(0, inplace=True)
        return df

# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING DATA CREATION  (temporal: features from T1, labels from T2)
# ─────────────────────────────────────────────────────────────────────────────
def make_training_data(T1, T2, fb: FeatureBuilder):
    """
    Positive: (user, edition) pairs from T2 with relevance labels.
    Negative: random editions not in T1∪T2 for each user.
    Features: computed from T1 via FeatureBuilder.
    """
    t0 = time.time()
    T2c = T2.copy()
    T2c["rel"] = T2c["event_type"].map({1: 1.0, 2: 3.0})
    pos = T2c.groupby(["user_id","edition_id"])["rel"].max().reset_index()
    pos.rename(columns={"rel": "label"}, inplace=True)

    all_eids = list(set(T1["edition_id"]))
    u_all = defaultdict(set)
    for uid, eid in zip(T1["user_id"], T1["edition_id"]):
        u_all[uid].add(eid)
    for uid, eid in zip(T2["user_id"], T2["edition_id"]):
        u_all[uid].add(eid)

    negs = []
    for uid in pos["user_id"].unique():
        n_pos = len(pos[pos["user_id"] == uid])
        n_neg = min(n_pos * NEG_RATIO, 300)
        excl  = u_all[uid]
        pool  = [e for e in all_eids if e not in excl]
        if len(pool) > n_neg:
            chosen = np.random.choice(pool, n_neg, replace=False)
        else:
            chosen = pool
        for eid in chosen:
            negs.append({"user_id": uid, "edition_id": eid, "label": 0.0})

    neg_df = pd.DataFrame(negs)
    train  = pd.concat([pos, neg_df], ignore_index=True).sample(frac=1, random_state=42)
    print(f"   training: {len(pos):,} pos + {len(neg_df):,} neg = {len(train):,}")

    feat = fb.featurize(train[["user_id","edition_id"]])
    feat["label"] = train["label"].values
    print(f"   featurized ({time.time()-t0:.1f}s)")
    return feat

# ─────────────────────────────────────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────────────────────────────────────
def train_gbm(feat):
    t0 = time.time()
    X = feat[FeatureBuilder.COLS].values.astype(np.float32)
    y = feat["label"].values.astype(np.float32)
    mdl = HistGradientBoostingRegressor(
        max_iter=500, max_depth=7, learning_rate=0.05,
        min_samples_leaf=30, l2_regularization=1.0, max_bins=255,
        random_state=42, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=30)
    mdl.fit(X, y)
    print(f"   GBM n_iter={mdl.n_iter_} ({time.time()-t0:.1f}s)")
    return mdl

# ─────────────────────────────────────────────────────────────────────────────
#  MMR RERANKING
# ─────────────────────────────────────────────────────────────────────────────
def mmr_rerank(scored_df, egm, lam=0.60, pool=MMR_POOL, top_k=TOP_K):
    t0 = time.time()
    results = []
    for uid, grp in scored_df.groupby("user_id"):
        gs = grp.nlargest(pool, "score")
        eids = gs["edition_id"].values; scores = gs["score"].values.astype(np.float64)
        smin, smax = scores.min(), scores.max()
        sn = (scores - smin) / (smax - smin) if smax > smin else np.ones_like(scores)
        gsets = [egm.get(int(e), _EMPTY) for e in eids]

        sel = []; cov = set(); rem = list(range(len(eids)))
        for _ in range(top_k):
            bi = -1; bv = -1e18
            for idx in rem:
                ig = gsets[idx]; ig_sz = len(ig)
                cg = len(ig - cov) / ig_sz if ig_sz else 0
                if sel:
                    ds = 0.0
                    for si in sel:
                        sg = gsets[si]; u = len(ig | sg)
                        ds += (1 - len(ig & sg) / u) if u else 0
                    ad = ds / len(sel)
                else:
                    ad = 1.0
                v = lam * sn[idx] + (1 - lam) * (0.5 * cg + 0.5 * ad)
                if v > bv:
                    bv = v; bi = idx
            sel.append(bi); cov |= gsets[bi]; rem.remove(bi)

        for rank, idx in enumerate(sel, 1):
            results.append((uid, int(eids[idx]), rank))

    out = pd.DataFrame(results, columns=["user_id","edition_id","rank"])
    print(f"   MMR {len(out):,} rows ({time.time()-t0:.1f}s)")
    return out

# ─────────────────────────────────────────────────────────────────────────────
#  METRICS  (exact formulas from rulls.txt)
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(sub, gt_dict, egm, target_users):
    ndcgs, divs = [], []
    for uid in target_users:
        ugrp = sub[sub["user_id"] == uid]
        if len(ugrp) == 0:
            continue
        ranked = ugrp.sort_values("rank")["edition_id"].values[:TOP_K]
        rels = [gt_dict.get((uid, int(e)), 0) for e in ranked]

        # NDCG
        dcg = sum(r / math.log2(k+2) for k, r in enumerate(rels))
        all_r = sorted([v for (u, _), v in gt_dict.items() if u == uid], reverse=True)
        ideal = (all_r + [0]*TOP_K)[:TOP_K]
        idcg = sum(r / math.log2(k+2) for k, r in enumerate(ideal))
        ndcg = dcg / idcg if idcg > 0 else 0.0

        # Diversity
        trel = [1 if r > 0 else 0 for r in rels]
        gsets = [egm.get(int(e), _EMPTY) for e in ranked]
        w = [1 / math.log2(k+2) for k in range(TOP_K)]
        ws = sum(w)
        cov_val = 0.0; covered = set()
        for k in range(TOP_K):
            if trel[k] and gsets[k]:
                cov_val += w[k] * len(gsets[k] - covered) / len(gsets[k])
                covered |= gsets[k]
        coverage = cov_val / ws if ws else 0

        rel_idx = [k for k in range(TOP_K) if trel[k]]
        if len(rel_idx) < 2:
            ild = 0.0
        else:
            ds = 0.0; cnt = 0
            for i in range(len(rel_idx)):
                for j in range(i+1, len(rel_idx)):
                    gi, gj = gsets[rel_idx[i]], gsets[rel_idx[j]]
                    u = len(gi | gj)
                    ds += (1 - len(gi & gj) / u) if u else 0
                    cnt += 1
            ild = ds / cnt if cnt else 0

        div = 0.5 * coverage + 0.5 * ild
        ndcgs.append(ndcg); divs.append(div)

    mn = np.mean(ndcgs) if ndcgs else 0
    md = np.mean(divs)  if divs  else 0
    return mn, md, 0.7 * mn + 0.3 * md

# ─────────────────────────────────────────────────────────────────────────────
#  VALIDATION CANDIDATE GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def make_val_candidates(T1, T2, n_cands=200):
    """Create per-user candidate lists with T2 positives + random negatives."""
    T2_ue = T2.groupby("user_id")["edition_id"].apply(set).to_dict()
    T1_ue = T1.groupby("user_id")["edition_id"].apply(set).to_dict()
    all_eids = list(set(T1["edition_id"]))
    rows = []
    for uid, pos_set in T2_ue.items():
        # must have T1 history too (otherwise features are empty)
        if uid not in T1_ue:
            continue
        positives = list(pos_set)
        n_neg = max(n_cands - len(positives), 100)
        excl = T1_ue.get(uid, set()) | pos_set
        pool = [e for e in all_eids if e not in excl]
        if len(pool) > n_neg:
            negs = list(np.random.choice(pool, n_neg, replace=False))
        else:
            negs = pool
        for eid in positives + negs:
            rows.append({"user_id": uid, "edition_id": eid})
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
#  SUBMISSION VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
def validate_submission(sub, cands, targets):
    t_u = set(targets["user_id"]); s_u = set(sub["user_id"])
    assert s_u == t_u, f"User mismatch {len(s_u)} vs {len(t_u)}"
    cm = cands.groupby("user_id")["edition_id"].apply(set).to_dict()
    for uid, g in sub.groupby("user_id"):
        assert len(g) == TOP_K
        assert set(g["rank"]) == set(range(1, TOP_K+1))
        assert g["edition_id"].nunique() == TOP_K
        assert set(g["edition_id"]).issubset(cm[uid])
    print("   ✓ Submission valid!")

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    wall = time.time()
    print("="*60)
    print("  BookRec Hackathon — v2 (temporal)")
    print("="*60)

    # ── load ──
    inter, users, eds, bg, targets, cands = load_data()
    e2b, e2a, bgm, egm, egs = build_maps(eds, bg)

    all_uids = sorted(set(targets["user_id"]) | set(inter["user_id"]))
    all_eids = sorted(set(inter["edition_id"]) | set(cands["edition_id"]))

    # ── time split ──
    max_ts = inter["event_ts"].max()
    val_cutoff = max_ts - pd.Timedelta(days=30)
    T1 = inter[inter["event_ts"] < val_cutoff].copy()
    T2 = inter[inter["event_ts"] >= val_cutoff].copy()
    print(f"\n[split] T1 {len(T1):,}  T2 {len(T2):,}  cutoff {val_cutoff.date()}")

    # ── T2 GT ──
    T2c = T2.copy(); T2c["rel"] = T2c["event_type"].map({1:1, 2:3})
    gt = T2c.groupby(["user_id","edition_id"])["rel"].max().reset_index()
    gt_dict = {(r.user_id, r.edition_id): r.rel for r in gt.itertuples()}

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASE A: LOCAL EVALUATION  (features from T1, labels from T2)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n── Phase A: local evaluation ──")
    print("[svd-T1]")
    uf_t1, if_t1, ue_t1, ee_t1 = do_svd(T1, all_uids, all_eids)

    print("[features-T1]")
    fb_t1 = FeatureBuilder(T1, users, eds, e2b, e2a, bgm, egm, egs,
                           uf_t1, if_t1, ue_t1, ee_t1)

    print("[training data T1→T2]")
    train_feat = make_training_data(T1, T2, fb_t1)
    model_t = train_gbm(train_feat)

    print("[val candidates]")
    val_cands_df = make_val_candidates(T1, T2)
    print(f"   {len(val_cands_df):,} val candidate pairs")

    print("[score val candidates]")
    vc_feat = fb_t1.featurize(val_cands_df[["user_id","edition_id"]])
    vc_feat["score"] = model_t.predict(
        vc_feat[FeatureBuilder.COLS].values.astype(np.float32))

    # MMR on val candidates — try several λ values
    val_users = set(val_cands_df["user_id"]) & set(gt["user_id"])
    best_score = -1; best_lam = 0.60
    for lam in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]:
        val_sub = mmr_rerank(vc_feat, egm, lam=lam, pool=MMR_POOL)
        mn, md, sc = compute_metrics(val_sub, gt_dict, egm, val_users)
        tag = " ◄ best" if sc > best_score else ""
        if sc > best_score:
            best_score = sc; best_lam = lam
        print(f"   λ={lam:.2f}  NDCG={mn:.5f}  Div={md:.5f}  Score={sc:.5f}{tag}")

    print(f"\n   Best λ = {best_lam:.2f}  Score = {best_score:.5f}")

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASE B: FINAL SUBMISSION  (features from ALL data, model from T1→T2)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n── Phase B: final submission ──")
    print("[svd-full]")
    uf_all, if_all, ue_all, ee_all = do_svd(inter, all_uids, all_eids)

    print("[features-full]")
    fb_all = FeatureBuilder(inter, users, eds, e2b, e2a, bgm, egm, egs,
                            uf_all, if_all, ue_all, ee_all)

    print("[score competition candidates]")
    cf = fb_all.featurize(cands[["user_id","edition_id"]])
    cf["score"] = model_t.predict(
        cf[FeatureBuilder.COLS].values.astype(np.float32))

    print(f"[mmr λ={best_lam}]")
    sub = mmr_rerank(cf, egm, lam=best_lam, pool=MMR_POOL)
    sub.to_csv(OUT, index=False)
    print(f"   saved → {OUT}")

    validate_submission(sub, cands, targets)
    print(f"\n   Total: {time.time()-wall:.0f}s")

if __name__ == "__main__":
    main()
