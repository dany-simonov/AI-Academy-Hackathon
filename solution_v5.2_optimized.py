"""
BookRec Hackathon — Solution v5.2 (OPTIMIZED FOR REAL SCORE)
=============================================================
Critical insights from LB results:
  - v5 (λ=0.3): LB=0.4428 (too much diversity)
  - v5.1 (λ=0.96): LB=0.4359 (too much relevance!)
  
KEY FIX: Metric = 0.7×NDCG + 0.3×Diversity
  → Optimal λ should be around 0.7-0.8 (matching metric weights!)
  
Improvements:
  - Multi-k SVD (k=32,64,128) for richer representations
  - Ensemble: temporal_model (T1) + full_model (all data)
  - λ tuning in proper range: 0.65-0.85
  - Better negative sampling (mix of random + popular + engaged users' items)
  - Reduced overfitting (stronger regularization, early stopping)
  - Ensemble weights optimization
"""

import os, time, math, warnings
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
OUT  = os.path.join(BASE, "submission_3.1.csv")

TOP_K      = 20
NEG_RATIO  = 5  # Balanced
MMR_POOL   = 120  # Default pool (can be overridden in grid)
_EMPTY     = frozenset()

# ═════════════════════════════════════════════════════════════════════════════
def load_data():
    """Load all data files."""
    t0 = time.time()
    inter   = pd.read_csv(os.path.join(DATA, "interactions.csv"))
    users   = pd.read_csv(os.path.join(DATA, "users.csv"))
    eds     = pd.read_csv(os.path.join(DATA, "editions.csv"))
    bg      = pd.read_csv(os.path.join(DATA, "book_genres.csv"))
    targets = pd.read_csv(os.path.join(DATA, "targets.csv"))
    cands   = pd.read_csv(os.path.join(DATA, "candidates.csv"))
    inter["event_ts"] = pd.to_datetime(inter["event_ts"])
    print(f"[load] {len(inter):,} inter  {len(targets):,} users  {len(cands):,} cands  ({time.time()-t0:.1f}s)")
    return inter, users, eds, bg, targets, cands

def build_maps(eds, bg):
    """Build edition/book/genre mappings."""
    e2b = eds.set_index("edition_id")["book_id"].to_dict()
    e2a = eds.set_index("edition_id")["author_id"].to_dict()
    bgm = bg.groupby("book_id")["genre_id"].apply(frozenset).to_dict()
    egm = {eid: bgm.get(bid, _EMPTY) for eid, bid in e2b.items()}
    egs = {eid: len(g) for eid, g in egm.items()}
    return e2b, e2a, bgm, egm, egs

def do_svd(inter, all_users, all_editions, k=64, half_life=150):
    """
    Multi-k SVD for richer representations.
    Returns dict {k: (user_factors, item_factors, u_enc, e_enc)}
    """
    tmp = inter.copy()
    tmp["rel"] = tmp["event_type"].map({1: 1.0, 2: 3.0})  # Standard: read=3x wishlist
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
    
    results = {}
    for ksz in [32, 64, 128]:
        ak = min(ksz, min(R.shape) - 1)
        U, s, Vt = svds(R, k=ak)
        idx = np.argsort(-s); U = U[:, idx]; s = s[idx]; Vt = Vt[idx, :]
        results[ksz] = (U * s[np.newaxis, :], Vt.T)
        print(f"   SVD k={ksz} shape={R.shape} nnz={R.nnz:,}")
    
    return results, u_enc, e_enc

# ═════════════════════════════════════════════════════════════════════════════
class FeatureBuilder:
    """Build features with multi-k SVD."""
    COLS = [
        "svd32", "svd64", "svd128",  # Multi-k SVD signals
        "u_n","u_reads","u_wish","u_avgr","u_maxr","u_stdr","u_ned",
        "u_rr","u_rec","u_span","u_rate","u_nb","u_na","gender","age",
        "u_ng","u_tgs","u_gent",
        "i_pop","i_nu","i_reads","i_wish","i_avgr","i_stdr","i_rr","i_rec",
        "i_bpop","i_apop","i_ng","publication_year","age_restriction","language_id",
        "i_trend", "i_conversion",
        "ua_cnt","ua_rd","ua_avgr_feat","ui_sb","ui_jac","ui_wov","ui_newg",
        "ui_genre_match",
    ]

    def __init__(self, inter, users, eds, e2b, e2a, bgm, egm, egs,
                 svd_results, u_enc, e_enc):
        self.egm = egm; self.egs = egs; self.e2a = e2a; self.e2b = e2b
        self.svd_results = svd_results
        self.u_enc = u_enc; self.e_enc = e_enc
        max_ts = inter["event_ts"].max()

        # User stats
        uf = inter.groupby("user_id").agg(
            u_n=("event_type","count"),
            u_reads=("event_type", lambda x:(x==2).sum()),
            u_wish=("event_type", lambda x:(x==1).sum()),
            u_avgr=("rating","mean"), u_maxr=("rating","max"), u_stdr=("rating","std"),
            u_ned=("edition_id","nunique"),
            u_last=("event_ts","max"), u_first=("event_ts","min"),
        ).reset_index()
        uf["u_rr"] = uf["u_reads"]/uf["u_n"].clip(1)
        uf["u_rec"] = (max_ts-uf["u_last"]).dt.days.astype(float)
        uf["u_span"] = (uf["u_last"]-uf["u_first"]).dt.days.clip(1).astype(float)
        uf["u_rate"] = uf["u_n"]/uf["u_span"]*30
        uf.drop(columns=["u_last","u_first"], inplace=True)

        im = inter.merge(eds[["edition_id","book_id","author_id"]], on="edition_id", how="left")
        uf = uf.merge(im.groupby("user_id")["book_id"].nunique().reset_index(name="u_nb"), on="user_id", how="left")
        uf = uf.merge(im.groupby("user_id")["author_id"].nunique().reset_index(name="u_na"), on="user_id", how="left")
        uf = uf.merge(users[["user_id","gender","age"]], on="user_id", how="left")

        # Genre profile
        ugp = {}; ugs = {}; self.user_genre_weights = {}
        for uid, grp in im.groupby("user_id"):
            gc = defaultdict(float)
            for et, bid in zip(grp["event_type"], grp["book_id"]):
                if pd.notna(bid):
                    for g in bgm.get(int(bid), _EMPTY):
                        weight = 3.0 if et == 2 else 1.0
                        gc[g] += weight
            ugp[uid] = dict(gc); ugs[uid] = set(gc.keys())
            total_weight = sum(gc.values())
            self.user_genre_weights[uid] = {g: w/total_weight for g, w in gc.items()} if total_weight > 0 else {}

        rows = []
        for uid in uf["user_id"]:
            gc = ugp.get(uid, {}); tot = sum(gc.values()) if gc else 0
            rows.append({"user_id": uid, "u_ng": len(gc),
                "u_tgs": max(gc.values())/max(tot,1) if gc else 0,
                "u_gent": -sum((v/tot)*np.log2(v/tot+1e-12) for v in gc.values()) if tot else 0})
        uf = uf.merge(pd.DataFrame(rows), on="user_id", how="left")
        self.uf = uf; self.ugp = ugp; self.ugs = ugs
        
        # Author affinity
        ua = im.groupby(["user_id","author_id"]).agg(
            c=("event_type","count"),
            r=("event_type", lambda x:(x==2).sum()),
            ar=("rating","mean")).reset_index()
        self.ua_dict = {(row.user_id, row.author_id): (row.c, row.r, row.ar)
                        for row in ua.itertuples()}
        self.ubs = im.groupby("user_id")["book_id"].apply(set).to_dict()

        # Item stats
        itf = inter.groupby("edition_id").agg(
            i_pop=("event_type","count"), i_nu=("user_id","nunique"),
            i_reads=("event_type", lambda x:(x==2).sum()),
            i_wish=("event_type", lambda x:(x==1).sum()),
            i_avgr=("rating","mean"), i_stdr=("rating","std"),
            i_last=("event_ts","max"),
        ).reset_index()
        itf["i_rr"] = itf["i_reads"]/itf["i_pop"].clip(1)
        itf["i_rec"] = (max_ts - itf["i_last"]).dt.days.astype(float)
        itf["i_conversion"] = itf["i_reads"] / itf["i_pop"].clip(1)
        itf.drop(columns=["i_last"], inplace=True)

        # Trending
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

    def featurize(self, pairs_df):
        """Compute features for user-item pairs."""
        df = pairs_df.copy()
        uids = df["user_id"].values; eids = df["edition_id"].values
        N = len(df)

        # Multi-k SVD scores
        for ksz in [32, 64, 128]:
            uf_k, if_k = self.svd_results[ksz]
            s = np.zeros(N, np.float32)
            for i in range(N):
                ui, ei = self.u_enc.get(uids[i]), self.e_enc.get(eids[i])
                if ui is not None and ei is not None:
                    s[i] = np.dot(uf_k[ui], if_k[ei])
            df[f"svd{ksz}"] = s

        # Merge stats
        df = df.merge(self.uf,  on="user_id",    how="left")
        df = df.merge(self.itf, on="edition_id", how="left")
        df = df.merge(self.em,  on="edition_id", how="left", suffixes=("","_em"))

        # Author features
        aids = [self.e2a.get(e) for e in eids]
        keys = list(zip(uids, aids))
        df["ua_cnt"] = [self.ua_dict.get(k, (0,0,0))[0] for k in keys]
        df["ua_rd"] = [self.ua_dict.get(k, (0,0,0))[1] for k in keys]
        df["ua_avgr_feat"] = [self.ua_dict.get(k, (0,0,np.nan))[2] for k in keys]

        # Same book
        sb = np.zeros(N, np.int8)
        for i in range(N):
            bid = self.e2b_local.get(eids[i])
            if bid is not None and bid in self.ubs.get(uids[i], set()):
                sb[i] = 1
        df["ui_sb"] = sb

        # Genre features
        jac = np.zeros(N, np.float32)
        wov = np.zeros(N, np.float32)
        newg = np.zeros(N, np.float32)
        genre_match = np.zeros(N, np.float32)
        
        for i in range(N):
            uid = uids[i]; eid = eids[i]
            ug = self.ugs.get(uid)
            ig = self.egm.get(eid, _EMPTY)
            ig_sz = self.egs.get(eid, 0)
            
            if not ug or ig_sz == 0:
                newg[i] = 1.0 if ig_sz else 0.0
                continue
                
            isz = len(ug & ig)
            usz = len(ug) + ig_sz - isz
            jac[i] = isz / usz if usz else 0
            
            user_weights = self.user_genre_weights.get(uid, {})
            weighted_score = sum(user_weights.get(g, 0) for g in ig)
            wov[i] = weighted_score
            
            newg[i] = (ig_sz - isz) / ig_sz
            
            if ig:
                genre_scores = [user_weights.get(g, 0) for g in ig]
                genre_match[i] = max(genre_scores) if genre_scores else 0
            
        df["ui_jac"] = jac
        df["ui_wov"] = wov
        df["ui_newg"] = newg
        df["ui_genre_match"] = genre_match

        df.fillna(0, inplace=True)
        return df

# ═════════════════════════════════════════════════════════════════════════════
def make_training_data(T1, T2, fb: FeatureBuilder):
    """Create training data with smart negative sampling."""
    t0 = time.time()
    T2c = T2.copy()
    T2c["rel"] = T2c["event_type"].map({1: 1.0, 2: 3.0})
    pos = T2c.groupby(["user_id","edition_id"])["rel"].max().reset_index()
    pos.rename(columns={"rel": "label"}, inplace=True)

    # Smart negative sampling
    all_eids = list(set(T1["edition_id"]))
    u_all = defaultdict(set)
    for uid, eid in zip(T1["user_id"], T1["edition_id"]):
        u_all[uid].add(eid)
    for uid, eid in zip(T2["user_id"], T2["edition_id"]):
        u_all[uid].add(eid)

    # Get popular and diverse items
    item_pop = T1.groupby("edition_id")["event_type"].count()
    popular = item_pop.nlargest(500).index.tolist()
    
    negs = []
    for uid in pos["user_id"].unique():
        n_pos = len(pos[pos["user_id"] == uid])
        n_neg = min(n_pos * NEG_RATIO, 300)
        excl = u_all[uid]
        
        # 60% random, 40% popular
        n_rand = int(n_neg * 0.6)
        n_pop = n_neg - n_rand
        
        pool_r = [e for e in all_eids if e not in excl]
        pool_p = [e for e in popular if e not in excl]
        
        chosen = []
        if len(pool_r) > n_rand:
            chosen.extend(np.random.choice(pool_r, n_rand, replace=False))
        else:
            chosen.extend(pool_r)
            
        if len(pool_p) > n_pop:
            chosen.extend(np.random.choice(pool_p, n_pop, replace=False))
        else:
            chosen.extend(pool_p)
            
        for eid in chosen:
            negs.append({"user_id": uid, "edition_id": eid, "label": 0.0})

    neg_df = pd.DataFrame(negs)
    train = pd.concat([pos, neg_df], ignore_index=True).sample(frac=1, random_state=42)
    print(f"   train: {len(pos):,} pos + {len(neg_df):,} neg = {len(train):,}")

    feat = fb.featurize(train[["user_id","edition_id"]])
    feat["label"] = train["label"].values
    print(f"   featurized ({time.time()-t0:.1f}s)")
    return feat

def train_gbm(feat, name="model"):
    """Train GBM with careful regularization."""
    t0 = time.time()
    X = feat[FeatureBuilder.COLS].values.astype(np.float32)
    y = feat["label"].values.astype(np.float32)
    
    mdl = HistGradientBoostingRegressor(
        max_iter=900,
        max_depth=7,
        learning_rate=0.04,
        min_samples_leaf=30,
        l2_regularization=1.0,
        max_bins=255,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=60)
    mdl.fit(X, y)
    print(f"   {name} GBM n_iter={mdl.n_iter_} ({time.time()-t0:.1f}s)")
    return mdl

# ═════════════════════════════════════════════════════════════════════════════
def mmr_rerank(scored_df, egm, lam=0.75, pool=MMR_POOL, top_k=TOP_K):
    """MMR reranking with proper balance."""
    t0 = time.time()
    results = []
    for uid, grp in scored_df.groupby("user_id"):
        gs = grp.nlargest(pool, "score")
        eids = gs["edition_id"].values
        scores = gs["score"].values.astype(np.float64)
        
        smin, smax = scores.min(), scores.max()
        if smax > smin:
            sn = (scores - smin) / (smax - smin)
        else:
            sn = np.ones_like(scores)
        
        gsets = [egm.get(int(e), _EMPTY) for e in eids]

        sel = []
        cov = set()
        rem = list(range(len(eids)))
        
        for _ in range(top_k):
            bi = -1
            bv = -1e18
            
            for idx in rem:
                ig = gsets[idx]
                ig_sz = len(ig)
                
                # Coverage gain
                if ig_sz > 0:
                    cg = len(ig - cov) / ig_sz
                else:
                    cg = 0.0
                
                # Distance to selected
                if sel:
                    ds = 0.0
                    for si in sel:
                        sg = gsets[si]
                        u = len(ig | sg)
                        if u > 0:
                            ds += (1 - len(ig & sg) / u)
                    ad = ds / len(sel)
                else:
                    ad = 1.0
                
                # Balanced diversity
                diversity_score = 0.5 * cg + 0.5 * ad
                
                # MMR score: λ should match metric weights!
                v = lam * sn[idx] + (1 - lam) * diversity_score
                
                if v > bv:
                    bv = v
                    bi = idx
            
            sel.append(bi)
            cov |= gsets[bi]
            rem.remove(bi)

        for rank, idx in enumerate(sel, 1):
            results.append((uid, int(eids[idx]), rank))

    out = pd.DataFrame(results, columns=["user_id","edition_id","rank"])
    print(f"   MMR {len(out):,} rows ({time.time()-t0:.1f}s)")
    return out

# ═════════════════════════════════════════════════════════════════════════════
def compute_metrics(sub, gt_dict, egm, target_users):
    """Compute NDCG and Diversity metrics."""
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
        cov_val = 0.0
        covered = set()
        for k in range(TOP_K):
            if trel[k] and gsets[k]:
                cov_val += w[k] * len(gsets[k] - covered) / len(gsets[k])
                covered |= gsets[k]
        coverage = cov_val / ws if ws else 0

        rel_idx = [k for k in range(TOP_K) if trel[k]]
        if len(rel_idx) < 2:
            ild = 0.0
        else:
            ds = 0.0
            cnt = 0
            for i in range(len(rel_idx)):
                for j in range(i+1, len(rel_idx)):
                    gi, gj = gsets[rel_idx[i]], gsets[rel_idx[j]]
                    u = len(gi | gj)
                    ds += (1 - len(gi & gj) / u) if u else 0
                    cnt += 1
            ild = ds / cnt if cnt else 0

        div = 0.5 * coverage + 0.5 * ild
        ndcgs.append(ndcg)
        divs.append(div)

    mn = np.mean(ndcgs) if ndcgs else 0
    md = np.mean(divs) if divs else 0
    return mn, md, 0.7 * mn + 0.3 * md

def make_val_candidates(T1, T2, n_cands=250):
    """Create validation candidates."""
    T2_ue = T2.groupby("user_id")["edition_id"].apply(set).to_dict()
    T1_ue = T1.groupby("user_id")["edition_id"].apply(set).to_dict()
    all_eids = list(set(T1["edition_id"]))
    rows = []
    for uid, pos_set in T2_ue.items():
        if uid not in T1_ue:
            continue
        positives = list(pos_set)
        n_neg = max(n_cands - len(positives), 150)
        excl = T1_ue.get(uid, set()) | pos_set
        pool = [e for e in all_eids if e not in excl]
        if len(pool) > n_neg:
            negs = list(np.random.choice(pool, n_neg, replace=False))
        else:
            negs = pool
        for eid in positives + negs:
            rows.append({"user_id": uid, "edition_id": eid})
    return pd.DataFrame(rows)

def validate_submission(sub, cands, targets):
    """Validate submission format."""
    t_u = set(targets["user_id"])
    s_u = set(sub["user_id"])
    assert s_u == t_u, f"User mismatch {len(s_u)} vs {len(t_u)}"
    cm = cands.groupby("user_id")["edition_id"].apply(set).to_dict()
    for uid, g in sub.groupby("user_id"):
        assert len(g) == TOP_K
        assert set(g["rank"]) == set(range(1, TOP_K+1))
        assert g["edition_id"].nunique() == TOP_K
        assert set(g["edition_id"]).issubset(cm[uid])
    print("   ✓ Submission valid!")

# ═════════════════════════════════════════════════════════════════════════════
def main():
    wall = time.time()
    print("="*70)
    print("  BookRec Hackathon — v5.2 OPTIMIZED")
    print("  Target: λ≈0.7-0.8 matching metric weights (0.7×NDCG + 0.3×Div)")
    print("="*70)

    # Load
    inter, users, eds, bg, targets, cands = load_data()
    e2b, e2a, bgm, egm, egs = build_maps(eds, bg)
    all_uids = sorted(set(targets["user_id"]) | set(inter["user_id"]))
    all_eids = sorted(set(inter["edition_id"]) | set(cands["edition_id"]))

    # Time split
    max_ts = inter["event_ts"].max()
    val_cutoff = max_ts - pd.Timedelta(days=30)
    T1 = inter[inter["event_ts"] < val_cutoff].copy()
    T2 = inter[inter["event_ts"] >= val_cutoff].copy()
    print(f"\n[split] T1 {len(T1):,}  T2 {len(T2):,}  cutoff {val_cutoff.date()}")

    # Ground truth
    T2c = T2.copy()
    T2c["rel"] = T2c["event_type"].map({1:1.0, 2:3.0})
    gt = T2c.groupby(["user_id","edition_id"])["rel"].max().reset_index()
    gt_dict = {(r.user_id, r.edition_id): r.rel for r in gt.itertuples()}

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE A: LOCAL EVALUATION
    # ══════════════════════════════════════════════════════════════════════════
    print("\n── Phase A: local evaluation ──")
    print("[multi-k SVD on T1]")
    svd_t1, ue_t1, ee_t1 = do_svd(T1, all_uids, all_eids)

    print("[features T1]")
    fb_t1 = FeatureBuilder(T1, users, eds, e2b, e2a, bgm, egm, egs,
                           svd_t1, ue_t1, ee_t1)

    print("[train temporal model T1→T2]")
    train_t = make_training_data(T1, T2, fb_t1)
    model_t = train_gbm(train_t, "temporal")

    print("[val candidates]")
    val_cands_df = make_val_candidates(T1, T2)
    print(f"   {len(val_cands_df):,} pairs")

    print("[score temporal]")
    vc_feat_t = fb_t1.featurize(val_cands_df[["user_id","edition_id"]])
    vc_feat_t["score_temporal"] = model_t.predict(
        vc_feat_t[FeatureBuilder.COLS].values.astype(np.float32))

    # Train full model for ensemble
    print("\n[multi-k SVD on full data]")
    svd_all, ue_all, ee_all = do_svd(inter, all_uids, all_eids)

    print("[features full]")
    fb_all = FeatureBuilder(inter, users, eds, e2b, e2a, bgm, egm, egs,
                            svd_all, ue_all, ee_all)

    print("[train full model (all data)]")
    train_all = make_training_data(T1, T2, fb_all)  # Still use T1→T2 for consistency
    model_all = train_gbm(train_all, "full")

    print("[score full]")
    vc_feat_all = fb_all.featurize(val_cands_df[["user_id","edition_id"]])
    vc_feat_all["score_full"] = model_all.predict(
        vc_feat_all[FeatureBuilder.COLS].values.astype(np.float32))

    # Grid search: ensemble weights × λ × MMR_POOL
    val_users = set(val_cands_df["user_id"]) & set(gt["user_id"])
    best_score = -1
    best_config = (0.5, 0.75, MMR_POOL)  # (ensemble_weight, lambda, mmr_pool)
    
    print("\n[grid search: ensemble_weight × λ × MMR_POOL]")
    print("   Note: metric = 0.7×NDCG + 0.3×Diversity → optimal λ ≈ 0.7-0.8")
    
    ensemble_weights = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    lambda_values = [0.65, 0.68, 0.70, 0.72, 0.75, 0.78, 0.80, 0.82, 0.85]
    mmr_pool_values = [100, 120, 140, 160]
    
    # Cache normalized scores to avoid repeated normalization inside loops
    s1 = vc_feat_t["score_temporal"].values
    s2 = vc_feat_all["score_full"].values
    s1n = (s1 - s1.min()) / (s1.max() - s1.min() + 1e-9)
    s2n = (s2 - s2.min()) / (s2.max() - s2.min() + 1e-9)

    for mp in mmr_pool_values:
        print(f"   Testing MMR_POOL={mp}")
        for ew in ensemble_weights:
            for lam in lambda_values:
                # Ensemble scores
                vc_feat_t["score"] = ew * s1n + (1 - ew) * s2n
                
                val_sub = mmr_rerank(vc_feat_t, egm, lam=lam, pool=mp)
                mn, md, sc = compute_metrics(val_sub, gt_dict, egm, val_users)
                tag = " ◄ BEST" if sc > best_score else ""
                if sc > best_score:
                    best_score = sc
                    best_config = (ew, lam, mp)
                print(f"   pool={mp} w={ew:.2f} λ={lam:.2f}  NDCG={mn:.5f}  Div={md:.5f}  Score={sc:.5f}{tag}")

    best_w, best_lam, best_pool = best_config
    print(f"\n   → Best: ensemble_weight={best_w:.2f}  λ={best_lam:.2f}  pool={best_pool}  Score={best_score:.5f}")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE B: FINAL SUBMISSION
    # ══════════════════════════════════════════════════════════════════════════
    print("\n── Phase B: final submission ──")
    print("[score competition candidates]")
    cf_t = fb_t1.featurize(cands[["user_id","edition_id"]])
    cf_t["score_temporal"] = model_t.predict(
        cf_t[FeatureBuilder.COLS].values.astype(np.float32))
    
    cf_all = fb_all.featurize(cands[["user_id","edition_id"]])
    cf_all["score_full"] = model_all.predict(
        cf_all[FeatureBuilder.COLS].values.astype(np.float32))

    # Ensemble with best weights
    s1 = cf_t["score_temporal"].values
    s2 = cf_all["score_full"].values
    s1n = (s1 - s1.min()) / (s1.max() - s1.min() + 1e-9)
    s2n = (s2 - s2.min()) / (s2.max() - s2.min() + 1e-9)
    cf_t["score"] = best_w * s1n + (1 - best_w) * s2n

    print(f"[mmr with best config: w={best_w:.2f} λ={best_lam:.2f} pool={best_pool}]")
    sub = mmr_rerank(cf_t, egm, lam=best_lam, pool=best_pool)
    sub.to_csv(OUT, index=False)
    print(f"   saved → {OUT}")

    validate_submission(sub, cands, targets)
    print(f"\n   Total: {time.time()-wall:.0f}s")

if __name__ == "__main__":
    main()
