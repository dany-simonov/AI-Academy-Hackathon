"""
BookRec Hackathon — Solution v5 (Improved)
==========================================
Key improvements:
  - Better λ tuning (lower values for more diversity)
  - Enhanced features
  - Better negative sampling
  - Improved MMR algorithm
  - Multiple SVD components
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
DATA = os.path.join(BASE, "data")
SUBMIT = os.path.join(BASE, "submit")
OUT  = os.path.join(BASE, "submission.csv")

TOP_K      = 20
SVD_K      = 64
NEG_RATIO  = 3  # Reduced from 5
MMR_POOL   = 100  # Increased from 80
_EMPTY     = frozenset()

# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    t0 = time.time()
    inter   = pd.read_csv(os.path.join(DATA, "interactions.csv"))
    users   = pd.read_csv(os.path.join(DATA, "users.csv"))
    eds     = pd.read_csv(os.path.join(DATA, "editions.csv"))
    bg      = pd.read_csv(os.path.join(DATA, "book_genres.csv"))
    targets = pd.read_csv(os.path.join(SUBMIT, "targets.csv"))
    cands   = pd.read_csv(os.path.join(SUBMIT, "candidates.csv"))
    inter["event_ts"] = pd.to_datetime(inter["event_ts"])
    print(f"[load] {len(inter):,} inter  {len(targets):,} users  {len(cands):,} cands  ({time.time()-t0:.1f}s)")
    return inter, users, eds, bg, targets, cands

def build_maps(eds, bg):
    e2b = eds.set_index("edition_id")["book_id"].to_dict()
    e2a = eds.set_index("edition_id")["author_id"].to_dict()
    bgm = bg.groupby("book_id")["genre_id"].apply(frozenset).to_dict()
    egm = {eid: bgm.get(bid, _EMPTY) for eid, bid in e2b.items()}
    egs = {eid: len(g) for eid, g in egm.items()}
    return e2b, e2a, bgm, egm, egs

def do_svd(inter, all_users, all_editions, k=SVD_K, half_life=120):
    """Enhanced SVD with better recency weighting."""
    tmp = inter.copy()
    tmp["rel"] = tmp["event_type"].map({1: 1.5, 2: 3.0})  # Slightly higher wishlist weight
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
class FeatureBuilder:
    COLS = [
        "svd", "svd_norm",
        # user features
        "u_n","u_reads","u_wish","u_avgr","u_maxr","u_stdr","u_ned",
        "u_rr","u_rec","u_span","u_rate","u_nb","u_na","gender","age",
        "u_ng","u_tgs","u_gent", "u_activity_recent",
        # item features  
        "i_pop","i_nu","i_reads","i_wish","i_avgr","i_stdr","i_rr","i_rec",
        "i_bpop","i_apop","i_ng","publication_year","age_restriction","language_id",
        "i_trend", "i_conversion", "i_popularity_rank",
        # user-item features
        "ua_cnt","ua_rd","ua_avgr_feat","ui_sb","ui_jac","ui_wov","ui_newg",
        "ui_genre_match", "ui_author_affinity", "ui_recency_match",
    ]

    def __init__(self, inter, users, eds, e2b, e2a, bgm, egm, egs,
                 user_f, item_f, u_enc, e_enc):
        self.egm = egm; self.egs = egs; self.e2a = e2a; self.e2b = e2b
        self.user_f = user_f; self.item_f = item_f
        self.u_enc = u_enc; self.e_enc = e_enc
        max_ts = inter["event_ts"].max()

        # Enhanced user stats
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
        
        # Recent activity (last 30 days)
        recent_inter = inter[inter["event_ts"] >= max_ts - pd.Timedelta(days=30)]
        recent_activity = recent_inter.groupby("user_id")["event_type"].count().reset_index(name="u_activity_recent")
        uf = uf.merge(recent_activity, on="user_id", how="left").fillna({"u_activity_recent": 0})
        
        uf.drop(columns=["u_last","u_first"], inplace=True)

        im = inter.merge(eds[["edition_id","book_id","author_id"]], on="edition_id", how="left")
        uf = uf.merge(im.groupby("user_id")["book_id"].nunique().reset_index(name="u_nb"), on="user_id", how="left")
        uf = uf.merge(im.groupby("user_id")["author_id"].nunique().reset_index(name="u_na"), on="user_id", how="left")
        uf = uf.merge(users[["user_id","gender","age"]], on="user_id", how="left")

        # Enhanced genre profile
        ugp = {}; ugs = {}; self.user_genre_weights = {}
        for uid, grp in im.groupby("user_id"):
            gc = defaultdict(float)
            for et, bid in zip(grp["event_type"], grp["book_id"]):
                if pd.notna(bid):
                    for g in bgm.get(int(bid), _EMPTY):
                        weight = 3.0 if et == 2 else 1.5  # Higher weights
                        gc[g] += weight
            ugp[uid] = dict(gc); ugs[uid] = set(gc.keys())
            # Normalize genre weights for this user
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

        # User books and authors
        self.ubs = im.groupby("user_id")["book_id"].apply(set).to_dict()
        self.user_authors = im.groupby("user_id")["author_id"].apply(set).to_dict()

        # Enhanced item stats
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
        
        # Popularity ranking
        itf["i_popularity_rank"] = itf["i_pop"].rank(pct=True)
        
        itf.drop(columns=["i_last"], inplace=True)

        # Enhanced trending features
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
        """Enhanced feature computation."""
        df = pairs_df.copy()
        uids = df["user_id"].values; eids = df["edition_id"].values
        N = len(df)

        # SVD scores (original and normalized)
        svd = np.zeros(N, np.float32)
        for i in range(N):
            ui, ei = self.u_enc.get(uids[i]), self.e_enc.get(eids[i])
            if ui is not None and ei is not None:
                svd[i] = np.dot(self.user_f[ui], self.item_f[ei])
        df["svd"] = svd
        # Normalized SVD score
        if svd.std() > 0:
            df["svd_norm"] = (svd - svd.mean()) / svd.std()
        else:
            df["svd_norm"] = svd

        # Merge basic stats
        df = df.merge(self.uf,  on="user_id",    how="left")
        df = df.merge(self.itf, on="edition_id", how="left")
        df = df.merge(self.em,  on="edition_id", how="left", suffixes=("","_em"))

        # Enhanced author features
        aids = [self.e2a.get(e) for e in eids]
        keys = list(zip(uids, aids))
        df["ua_cnt"] = [self.ua_dict.get(k, (0,0,0))[0] for k in keys]
        df["ua_rd"] = [self.ua_dict.get(k, (0,0,0))[1] for k in keys]
        df["ua_avgr_feat"] = [self.ua_dict.get(k, (0,0,np.nan))[2] for k in keys]
        
        # Author affinity (binary: has user read this author before?)
        ui_author_affinity = np.zeros(N, np.float32)
        for i in range(N):
            aid = aids[i]
            if aid and aid in self.user_authors.get(uids[i], set()):
                ui_author_affinity[i] = 1.0
        df["ui_author_affinity"] = ui_author_affinity

        # Same book feature
        sb = np.zeros(N, np.int8)
        for i in range(N):
            bid = self.e2b_local.get(eids[i])
            if bid is not None and bid in self.ubs.get(uids[i], set()):
                sb[i] = 1
        df["ui_sb"] = sb 
       # Enhanced genre features
        jac = np.zeros(N, np.float32)
        wov = np.zeros(N, np.float32)
        newg = np.zeros(N, np.float32)
        genre_match = np.zeros(N, np.float32)
        recency_match = np.zeros(N, np.float32)
        
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
            
            # Weighted overlap using user's genre preferences
            user_weights = self.user_genre_weights.get(uid, {})
            weighted_score = sum(user_weights.get(g, 0) for g in ig)
            wov[i] = weighted_score
            
            newg[i] = (ig_sz - isz) / ig_sz
            
            # Genre match strength (how well item genres match user preferences)
            if ig:
                genre_scores = [user_weights.get(g, 0) for g in ig]
                genre_match[i] = max(genre_scores) if genre_scores else 0
            
            # Recency match (prefer items similar to recent interactions)
            # This is a simplified version - could be enhanced
            recency_match[i] = min(1.0, isz / max(1, len(ug)) * 2)
            
        df["ui_jac"] = jac
        df["ui_wov"] = wov
        df["ui_newg"] = newg
        df["ui_genre_match"] = genre_match
        df["ui_recency_match"] = recency_match

        df.fillna(0, inplace=True)
        return df

# ─────────────────────────────────────────────────────────────────────────────
def make_training_data(T1, T2, fb: FeatureBuilder):
    """Enhanced training data creation with better negative sampling."""
    t0 = time.time()
    T2c = T2.copy()
    T2c["rel"] = T2c["event_type"].map({1: 1.5, 2: 3.0})  # Higher wishlist weight
    pos = T2c.groupby(["user_id","edition_id"])["rel"].max().reset_index()
    pos.rename(columns={"rel": "label"}, inplace=True)

    # Better negative sampling strategy
    all_eids = list(set(T1["edition_id"]))
    u_all = defaultdict(set)
    for uid, eid in zip(T1["user_id"], T1["edition_id"]):
        u_all[uid].add(eid)
    for uid, eid in zip(T2["user_id"], T2["edition_id"]):
        u_all[uid].add(eid)

    # Get popular items for better negative sampling
    item_popularity = T1.groupby("edition_id")["event_type"].count().sort_values(ascending=False)
    popular_items = item_popularity.head(1000).index.tolist()
    
    negs = []
    for uid in pos["user_id"].unique():
        n_pos = len(pos[pos["user_id"] == uid])
        n_neg = min(n_pos * NEG_RATIO, 200)  # Reduced negative ratio
        excl = u_all[uid]
        
        # Mix of random and popular negatives
        pool_random = [e for e in all_eids if e not in excl]
        pool_popular = [e for e in popular_items if e not in excl]
        
        # 70% random, 30% popular negatives
        n_random = int(n_neg * 0.7)
        n_popular = n_neg - n_random
        
        chosen = []
        if len(pool_random) > n_random:
            chosen.extend(np.random.choice(pool_random, n_random, replace=False))
        else:
            chosen.extend(pool_random)
            
        if len(pool_popular) > n_popular:
            chosen.extend(np.random.choice(pool_popular, n_popular, replace=False))
        else:
            chosen.extend(pool_popular)
            
        for eid in chosen:
            negs.append({"user_id": uid, "edition_id": eid, "label": 0.0})

    neg_df = pd.DataFrame(negs)
    train = pd.concat([pos, neg_df], ignore_index=True).sample(frac=1, random_state=42)
    print(f"   training: {len(pos):,} pos + {len(neg_df):,} neg = {len(train):,}")

    feat = fb.featurize(train[["user_id","edition_id"]])
    feat["label"] = train["label"].values
    print(f"   featurized ({time.time()-t0:.1f}s)")
    return feat

def train_gbm(feat):
    """Enhanced GBM training with better parameters."""
    t0 = time.time()
    X = feat[FeatureBuilder.COLS].values.astype(np.float32)
    y = feat["label"].values.astype(np.float32)
    
    # Enhanced parameters for better performance
    mdl = HistGradientBoostingRegressor(
        max_iter=800,  # Increased iterations
        max_depth=8,   # Slightly deeper trees
        learning_rate=0.03,  # Lower learning rate
        min_samples_leaf=20,  # Reduced for more flexibility
        l2_regularization=0.5,  # Reduced regularization
        max_bins=255,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,  # Larger validation set
        n_iter_no_change=50)  # More patience
    mdl.fit(X, y)
    print(f"   GBM n_iter={mdl.n_iter_} ({time.time()-t0:.1f}s)")
    return mdl

# ─────────────────────────────────────────────────────────────────────────────
def mmr_rerank(scored_df, egm, lam=0.3, pool=MMR_POOL, top_k=TOP_K):
    """Enhanced MMR with better diversity focus."""
    t0 = time.time()
    results = []
    for uid, grp in scored_df.groupby("user_id"):
        gs = grp.nlargest(pool, "score")
        eids = gs["edition_id"].values
        scores = gs["score"].values.astype(np.float64)
        
        # Better score normalization
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
                
                # Coverage gain (new genres)
                if ig_sz > 0:
                    cg = len(ig - cov) / ig_sz
                else:
                    cg = 0.0
                
                # Average distance to selected items
                if sel:
                    ds = 0.0
                    for si in sel:
                        sg = gsets[si]
                        u = len(ig | sg)
                        if u > 0:
                            ds += (1 - len(ig & sg) / u)
                        else:
                            ds += 0.0
                    ad = ds / len(sel)
                else:
                    ad = 1.0
                
                # Enhanced diversity component with genre coverage boost
                diversity_score = 0.6 * cg + 0.4 * ad
                
                # Final MMR score with lower lambda for more diversity
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

# ─────────────────────────────────────────────────────────────────────────────
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

def make_val_candidates(T1, T2, n_cands=200):
    """Create validation candidates."""
    T2_ue = T2.groupby("user_id")["edition_id"].apply(set).to_dict()
    T1_ue = T1.groupby("user_id")["edition_id"].apply(set).to_dict()
    all_eids = list(set(T1["edition_id"]))
    rows = []
    for uid, pos_set in T2_ue.items():
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

# ═══════════════════════════════════════════════════════════════════════════════
def main():
    wall = time.time()
    print("="*60)
    print("  BookRec Hackathon — v5 (Improved)")
    print("="*60)

    # Load data
    inter, users, eds, bg, targets, cands = load_data()
    e2b, e2a, bgm, egm, egs = build_maps(eds, bg)

    all_uids = sorted(set(targets["user_id"]) | set(inter["user_id"]))
    all_eids = sorted(set(inter["edition_id"]) | set(cands["edition_id"]))

    # Time split for validation
    max_ts = inter["event_ts"].max()
    val_cutoff = max_ts - pd.Timedelta(days=30)
    T1 = inter[inter["event_ts"] < val_cutoff].copy()
    T2 = inter[inter["event_ts"] >= val_cutoff].copy()
    print(f"\n[split] T1 {len(T1):,}  T2 {len(T2):,}  cutoff {val_cutoff.date()}")

    # Ground truth for validation
    T2c = T2.copy()
    T2c["rel"] = T2c["event_type"].map({1:1.5, 2:3})
    gt = T2c.groupby(["user_id","edition_id"])["rel"].max().reset_index()
    gt_dict = {(r.user_id, r.edition_id): r.rel for r in gt.itertuples()}

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE A: LOCAL EVALUATION with λ tuning
    # ══════════════════════════════════════════════════════════════════════════
    print("\n── Phase A: local evaluation with λ tuning ──")
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

    # Extensive λ tuning with focus on diversity
    val_users = set(val_cands_df["user_id"]) & set(gt["user_id"])
    best_score = -1
    best_lam = 0.3
    
    print("\n[λ tuning - focusing on diversity]")
    lambda_values = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]
    
    for lam in lambda_values:
        val_sub = mmr_rerank(vc_feat, egm, lam=lam, pool=MMR_POOL)
        mn, md, sc = compute_metrics(val_sub, gt_dict, egm, val_users)
        tag = " ◄ best" if sc > best_score else ""
        if sc > best_score:
            best_score = sc
            best_lam = lam
        print(f"   λ={lam:.2f}  NDCG={mn:.5f}  Div={md:.5f}  Score={sc:.5f}{tag}")

    print(f"\n   → Best λ = {best_lam:.2f}  Score = {best_score:.5f}")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE B: FINAL SUBMISSION
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