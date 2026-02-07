"""
BookRec Hackathon — Solution v5 (grid search for best ensemble/MMR)
Produces `submission_1.csv` with best-found config on time-split validation.
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
OUT1 = os.path.join(BASE, "submission_1.csv")

TOP_K = 20
_EMPTY = frozenset()

# Reuse multi-k SVD and feature builder logic (simplified copy of v4)

def do_svd(inter, all_u, all_e, k=64, half_life=180):
    tmp = inter.copy()
    tmp["rel"] = tmp["event_type"].map({1:1.0,2:3.0})
    ui = tmp.groupby(["user_id","edition_id"]).agg(rel=("rel","max"), ts=("event_ts","max")).reset_index()
    ref = inter["event_ts"].max()
    days = (ref - ui["ts"]).dt.days.values.astype(np.float32)
    ui["val"] = np.log1p(ui["rel"].values) * np.exp(-days / half_life)
    u_enc = {u:i for i,u in enumerate(all_u)}
    e_enc = {e:i for i,e in enumerate(all_e)}
    mask = ui["user_id"].isin(u_enc) & ui["edition_id"].isin(e_enc)
    uiv = ui[mask]
    R = csr_matrix((uiv["val"].values.astype(np.float32), (uiv["user_id"].map(u_enc).values, uiv["edition_id"].map(e_enc).values)), shape=(len(u_enc), len(e_enc)))
    ak = min(k, min(R.shape)-1)
    U, s, Vt = svds(R, k=ak)
    idx = np.argsort(-s); U = U[:, idx]; s = s[idx]; Vt = Vt[idx, :]
    return U * s[np.newaxis, :], Vt.T, u_enc, e_enc

class FeatureBuilderV5:
    COLS = [
        "svd32","svd64","svd128",
        "u_n","u_reads","u_wish","u_avgr","u_maxr","u_stdr","u_ned",
        "u_rr","u_rec","u_span","u_rate","u_nb","u_na","gender","age",
        "u_ng","u_tgs","u_gent",
        "i_pop","i_nu","i_reads","i_wish","i_avgr","i_stdr","i_rr","i_rec",
        "i_bpop","i_apop","i_ng","publication_year","age_restriction","language_id",
        "i_trend","i_conversion",
        "ua_cnt","ua_rd","ua_avgr_feat","ui_sb","ui_jac","ui_wov","ui_newg",
        "ui_genre_read_rate",
    ]

    def __init__(self, inter, users, eds, e2b, e2a, bgm, egm, egs, svd_results, u_enc, e_enc):
        self.egm = egm; self.egs = egs; self.e2a = e2a
        self.svd_results = svd_results; self.u_enc = u_enc; self.e_enc = e_enc
        max_ts = inter["event_ts"].max()
        # user stats
        uf = inter.groupby("user_id").agg(u_n=("event_type","count"), u_reads=("event_type", lambda x:(x==2).sum()), u_wish=("event_type", lambda x:(x==1).sum()), u_avgr=("rating","mean"), u_maxr=("rating","max"), u_stdr=("rating","std"), u_ned=("edition_id","nunique"), u_last=("event_ts","max"), u_first=("event_ts","min")).reset_index()
        uf["u_rr"] = uf["u_reads"]/uf["u_n"].clip(1)
        uf["u_rec"] = (max_ts - uf["u_last"]).dt.days.astype(float)
        uf["u_span"] = (uf["u_last"] - uf["u_first"]).dt.days.clip(1).astype(float)
        uf["u_rate"] = uf["u_n"] / uf["u_span"] * 30
        uf.drop(columns=["u_last","u_first"], inplace=True)
        im = inter.merge(eds[["edition_id","book_id","author_id"]], on="edition_id", how="left")
        uf = uf.merge(im.groupby("user_id")["book_id"].nunique().reset_index(name="u_nb"), on="user_id", how="left")
        uf = uf.merge(im.groupby("user_id")["author_id"].nunique().reset_index(name="u_na"), on="user_id", how="left")
        uf = uf.merge(users[["user_id","gender","age"]], on="user_id", how="left")
        # genre profile
        ugp = {}; ugs = {}
        self.u_genre_read_rate = {}
        for uid, grp in im.groupby("user_id"):
            gc = defaultdict(float); total = defaultdict(int); reads = defaultdict(int)
            for et, bid in zip(grp["event_type"], grp["book_id"]):
                if pd.notna(bid):
                    for g in bgm.get(int(bid), _EMPTY):
                        gc[g] += (3 if et == 2 else 1)
                        total[g] += 1
                        if et == 2: reads[g] += 1
            ugp[uid] = dict(gc); ugs[uid] = set(gc.keys())
            for g in total:
                self.u_genre_read_rate[(uid,g)] = reads[g] / total[g]
        rows = []
        for uid in uf["user_id"]:
            gc = ugp.get(uid, {}); tot = sum(gc.values()) if gc else 0
            rows.append({"user_id": uid, "u_ng": len(gc), "u_tgs": max(gc.values())/max(tot,1) if gc else 0, "u_gent": -sum((v/tot)*np.log2(v/tot+1e-12) for v in gc.values()) if tot else 0})
        uf = uf.merge(pd.DataFrame(rows), on="user_id", how="left")
        self.uf = uf; self.ugp = ugp; self.ugs = ugs
        ua = im.groupby(["user_id","author_id"]).agg(c=("event_type","count"), r=("event_type", lambda x:(x==2).sum()), ar=("rating","mean")).reset_index()
        self.ua_dict = {(row.user_id, row.author_id): (row.c, row.r, row.ar) for row in ua.itertuples()}
        self.ubs = im.groupby("user_id")["book_id"].apply(set).to_dict()
        # item stats
        itf = inter.groupby("edition_id").agg(i_pop=("event_type","count"), i_nu=("user_id","nunique"), i_reads=("event_type", lambda x:(x==2).sum()), i_wish=("event_type", lambda x:(x==1).sum()), i_avgr=("rating","mean"), i_stdr=("rating","std"), i_last=("event_ts","max")).reset_index()
        itf["i_rr"] = itf["i_reads"]/itf["i_pop"].clip(1)
        itf["i_rec"] = (max_ts - itf["i_last"]).dt.days.astype(float)
        itf.drop(columns=["i_last"], inplace=True)
        recent7 = inter[inter["event_ts"] >= max_ts - pd.Timedelta(days=7)]
        recent30 = inter[inter["event_ts"] >= max_ts - pd.Timedelta(days=30)]
        c7 = recent7.groupby("edition_id")["event_type"].count().reset_index(name="c7")
        c30 = recent30.groupby("edition_id")["event_type"].count().reset_index(name="c30")
        trend = c7.merge(c30, on="edition_id", how="outer").fillna(0)
        trend["i_trend"] = trend["c7"] / trend["c30"].clip(1)
        itf = itf.merge(trend[["edition_id","i_trend"]], on="edition_id", how="left")
        itf["i_trend"] = itf["i_trend"].fillna(0)
        itf["i_conversion"] = itf["i_reads"] / itf["i_pop"].clip(1)
        bp = im.groupby("book_id")["event_type"].count().reset_index(name="i_bpop")
        ap = im.groupby("author_id")["event_type"].count().reset_index(name="i_apop")
        em = eds[["edition_id","book_id","author_id","publication_year","age_restriction","language_id"]].copy()
        em = em.merge(bp, on="book_id", how="left").fillna({"i_bpop":0})
        em = em.merge(ap, on="author_id", how="left").fillna({"i_apop":0})
        em["i_ng"] = em["edition_id"].map(lambda x: len(egm.get(x, _EMPTY)))
        self.itf = itf; self.em = em; self.e2b_local = em.set_index("edition_id")["book_id"].to_dict()

    def featurize(self, pairs_df):
        df = pairs_df.copy(); uids = df["user_id"].values; eids = df["edition_id"].values; N = len(df)
        # multi-svd
        for k in [32,64,128]:
            uf_k, if_k = self.svd_results[k]
            s = np.zeros(N, np.float32)
            for i in range(N):
                ui = self.u_enc.get(uids[i]); ei = self.e_enc.get(eids[i])
                if ui is not None and ei is not None:
                    s[i] = np.dot(uf_k[ui], if_k[ei])
            df[f"svd{k}"] = s
        df = df.merge(self.uf, on="user_id", how="left")
        df = df.merge(self.itf, on="edition_id", how="left")
        df = df.merge(self.em, on="edition_id", how="left", suffixes=("","_em"))
        aids = [self.e2a.get(e) for e in eids]
        keys = list(zip(uids, aids))
        df["ua_cnt"] = [self.ua_dict.get(k, (0,0,0))[0] for k in keys]
        df["ua_rd"] = [self.ua_dict.get(k, (0,0,0))[1] for k in keys]
        df["ua_avgr_feat"] = [self.ua_dict.get(k, (0,0,np.nan))[2] for k in keys]
        sb = np.zeros(N, np.int8)
        for i in range(N):
            bid = self.e2b_local.get(eids[i])
            if bid is not None and bid in self.ubs.get(uids[i], set()): sb[i] = 1
        df["ui_sb"] = sb
        jac = np.zeros(N, np.float32); wov = np.zeros(N, np.float32); newg = np.zeros(N, np.float32); grr = np.zeros(N, np.float32)
        for i in range(N):
            uid = uids[i]; eid = eids[i]
            ug = self.ugs.get(uid); ig = self.egm.get(eid, _EMPTY); ig_sz = self.egs.get(eid, 0)
            if not ug or ig_sz == 0:
                newg[i] = 1.0 if ig_sz else 0.0; continue
            isz = len(ug & ig); usz = len(ug) + ig_sz - isz
            jac[i] = isz / usz if usz else 0
            gp = self.ugp.get(uid, {}); w = sum(gp.get(g, 0) for g in ig)
            tw = sum(gp.values()); wov[i] = w / tw if tw else 0
            newg[i] = (ig_sz - isz) / ig_sz
            rates = [self.u_genre_read_rate.get((uid, g), 0) for g in ig]
            grr[i] = np.mean(rates) if rates else 0
        df["ui_jac"] = jac; df["ui_wov"] = wov; df["ui_newg"] = newg; df["ui_genre_read_rate"] = grr
        df.fillna(0, inplace=True)
        return df

# metrics and mmr

def mmr_rerank_df(scored_df, egm, lam, pool=80, top_k=20):
    results = []
    for uid, grp in scored_df.groupby("user_id"):
        gs = grp.nlargest(pool, "score")
        eids = gs["edition_id"].values; scores = gs["score"].values.astype(np.float64)
        smin, smax = scores.min(), scores.max()
        snorm = (scores - smin) / (smax - smin) if smax > smin else np.ones_like(scores)
        gsets = [egm.get(int(e), _EMPTY) for e in eids]
        sel = []; cov = set(); rem = list(range(len(eids)))
        for _ in range(top_k):
            best_i = -1; best_v = -1e18
            for idx in rem:
                ig = gsets[idx]; ig_sz = len(ig)
                cov_gain = len(ig - cov) / ig_sz if ig_sz else 0
                if sel:
                    dist_sum = 0.0
                    for si in sel:
                        sg = gsets[si]; union = len(ig | sg)
                        if union: dist_sum += 1.0 - len(ig & sg) / union
                    avg_dist = dist_sum / len(sel)
                else:
                    avg_dist = 1.0
                div = 0.5 * cov_gain + 0.5 * avg_dist
                val = lam * snorm[idx] + (1.0 - lam) * div
                if val > best_v:
                    best_v = val; best_i = idx
            sel.append(best_i); cov |= gsets[best_i]; rem.remove(best_i)
        for rank, idx in enumerate(sel, 1):
            results.append((uid, int(eids[idx]), rank))
    return pd.DataFrame(results, columns=["user_id","edition_id","rank"]) 


def compute_score(sub, gt_dict, egm, users):
    ndcgs = []; divs = []
    for uid in users:
        grp = sub[sub["user_id"] == uid]
        if grp.empty: continue
        ranked = grp.sort_values("rank")["edition_id"].values[:TOP_K]
        rels = [gt_dict.get((uid, int(e)), 0) for e in ranked]
        dcg = sum(r / math.log2(k+2) for k, r in enumerate(rels))
        all_r = sorted([v for (u,e), v in gt_dict.items() if u == uid], reverse=True)
        ideal = (all_r + [0]*TOP_K)[:TOP_K]
        idcg = sum(r / math.log2(k+2) for k, r in enumerate(ideal))
        ndcg = dcg / idcg if idcg > 0 else 0
        trel = [1 if r > 0 else 0 for r in rels]
        gsets = [egm.get(int(e), _EMPTY) for e in ranked]
        w = [1.0 / math.log2(k+2) for k in range(TOP_K)]; wsum = sum(w)
        covered = set(); cov = 0.0
        for k in range(TOP_K):
            if trel[k] and gsets[k]:
                cov += w[k] * len(gsets[k] - covered) / len(gsets[k])
                covered |= gsets[k]
        coverage = cov / wsum if wsum else 0
        L = [k for k in range(TOP_K) if trel[k]]
        if len(L) < 2: ild = 0.0
        else:
            dsum = 0.0; cnt = 0
            for i in range(len(L)):
                for j in range(i+1, len(L)):
                    gi = gsets[L[i]]; gj = gsets[L[j]]; u = len(gi | gj)
                    if u: dsum += 1.0 - len(gi & gj) / u
                    cnt += 1
            ild = dsum / cnt if cnt else 0
        ndcgs.append(ndcg); divs.append(0.5 * coverage + 0.5 * ild)
    mn = np.mean(ndcgs) if ndcgs else 0; md = np.mean(divs) if divs else 0
    return mn, md, 0.7 * mn + 0.3 * md

# MAIN
if __name__ == '__main__':
    t0 = time.time()
    inter = pd.read_csv(os.path.join(DATA, 'interactions.csv'))
    users = pd.read_csv(os.path.join(DATA, 'users.csv'))
    eds = pd.read_csv(os.path.join(DATA, 'editions.csv'))
    bg = pd.read_csv(os.path.join(DATA, 'book_genres.csv'))
    targets = pd.read_csv(os.path.join(DATA, 'targets.csv'))
    cands = pd.read_csv(os.path.join(DATA, 'candidates.csv'))
    inter['event_ts'] = pd.to_datetime(inter['event_ts'])

    e2b = eds.set_index('edition_id')['book_id'].to_dict()
    e2a = eds.set_index('edition_id')['author_id'].to_dict()
    bgm = bg.groupby('book_id')['genre_id'].apply(frozenset).to_dict()
    egm = {eid: bgm.get(bid, _EMPTY) for eid, bid in e2b.items()}
    egs = {eid: len(gs) for eid, gs in egm.items()}

    all_u = sorted(set(targets['user_id']) | set(inter['user_id']))
    all_e = sorted(set(inter['edition_id']) | set(cands['edition_id']))

    # time split
    max_ts = inter['event_ts'].max(); val_cut = max_ts - pd.Timedelta(days=30)
    T1 = inter[inter['event_ts'] < val_cut].copy(); T2 = inter[inter['event_ts'] >= val_cut].copy()

    # GT
    T2c = T2.copy(); T2c['rel'] = T2c['event_type'].map({1:1,2:3})
    gt = T2c.groupby(['user_id','edition_id'])['rel'].max().reset_index()
    gt_dict = {(r.user_id, r.edition_id): r.rel for r in gt.itertuples()}

    # SVD multi-k for T1 and full
    svd_t1 = {}
    for k in [32,64,128]:
        uf, itf, uenc, eenc = do_svd(T1, all_u, all_e, k=k)
        svd_t1[k] = (uf, itf)
    svd_all = {}
    for k in [32,64,128]:
        uf, itf, uenc2, eenc2 = do_svd(inter, all_u, all_e, k=k)
        svd_all[k] = (uf, itf)

    # Feature builders
    fb_t1 = FeatureBuilderV5(T1, users, eds, e2b, e2a, bgm, egm, egs, svd_t1, uenc, eenc)
    fb_all = FeatureBuilderV5(inter, users, eds, e2b, e2a, bgm, egm, egs, svd_all, uenc2, eenc2)

    # Train temporal and full models
    # temporal
    tmp = T2.copy(); tmp['rel'] = tmp['event_type'].map({1:1.0,2:3.0})
    pos = tmp.groupby(['user_id','edition_id'])['rel'].max().reset_index().rename(columns={'rel':'label'})
    all_eids = list(set(T1['edition_id']))
    u_all = defaultdict(set)
    for uid,eid in zip(T1['user_id'], T1['edition_id']): u_all[uid].add(eid)
    for uid,eid in zip(T2['user_id'], T2['edition_id']): u_all[uid].add(eid)
    negs = []
    for uid in pos['user_id'].unique():
        npos = len(pos[pos['user_id'] == uid]); nneg = min(npos * 5, 300)
        excl = u_all[uid]
        pool = [e for e in all_eids if e not in excl]
        if len(pool) > nneg: chosen = np.random.choice(pool, nneg, replace=False)
        else: chosen = pool
        for eid in chosen: negs.append({'user_id': uid, 'edition_id': eid, 'label': 0.0})
    neg_df = pd.DataFrame(negs)
    train_df = pd.concat([pos, neg_df], ignore_index=True).sample(frac=1, random_state=42)
    feat_train = fb_t1.featurize(train_df[['user_id','edition_id']])
    feat_train['label'] = train_df['label'].values
    X = feat_train[FeatureBuilderV5.COLS].values.astype(np.float32); y = feat_train['label'].values.astype(np.float32)
    model_temporal = HistGradientBoostingRegressor(max_iter=1500, max_depth=7, learning_rate=0.05, min_samples_leaf=30, l2_regularization=1.0, max_bins=255, random_state=42, early_stopping=True, validation_fraction=0.1, n_iter_no_change=50)
    model_temporal.fit(X,y)

    # full model
    tmp2 = inter.copy(); tmp2['rel'] = tmp2['event_type'].map({1:1.0,2:3.0})
    pos2 = tmp2.groupby(['user_id','edition_id'])['rel'].max().reset_index().rename(columns={'rel':'label'})
    u_inter_all = inter.groupby('user_id')['edition_id'].apply(set).to_dict()
    ae_full = list(set(inter['edition_id']))
    negs2 = []
    for uid, eset in u_inter_all.items():
        nn = min(len(eset) * 5, 500)
        pool = [e for e in ae_full if e not in eset]
        if len(pool) > nn: ch = np.random.choice(pool, nn, replace=False)
        else: ch = pool
        for eid in ch: negs2.append({'user_id': uid, 'edition_id': eid, 'label': 0.0})
    neg2 = pd.DataFrame(negs2)
    train_full = pd.concat([pos2, neg2], ignore_index=True).sample(frac=1, random_state=42)
    feat_full = fb_all.featurize(train_full[['user_id','edition_id']])
    feat_full['label'] = train_full['label'].values
    Xf = feat_full[FeatureBuilderV5.COLS].values.astype(np.float32); yf = feat_full['label'].values.astype(np.float32)
    model_full = HistGradientBoostingRegressor(max_iter=1500, max_depth=7, learning_rate=0.05, min_samples_leaf=30, l2_regularization=1.0, max_bins=255, random_state=42, early_stopping=True, validation_fraction=0.1, n_iter_no_change=50)
    model_full.fit(Xf, yf)

    # prepare val candidates
    T2_ue = T2.groupby('user_id')['edition_id'].apply(set).to_dict()
    T1_ue = T1.groupby('user_id')['edition_id'].apply(set).to_dict()
    ae = list(set(T1['edition_id']))
    rows = []
    for uid, ps in T2_ue.items():
        if uid not in T1_ue: continue
        nn = max(200 - len(ps), 100)
        excl = T1_ue.get(uid, set()) | ps
        pool = [e for e in ae if e not in excl]
        if len(pool) > nn: ns = list(np.random.choice(pool, nn, replace=False))
        else: ns = pool
        for eid in list(ps) + ns: rows.append({'user_id': uid, 'edition_id': eid})
    val_cands = pd.DataFrame(rows)
    vc_feat = fb_t1.featurize(val_cands[['user_id','edition_id']])
    vc_feat['score_temporal'] = model_temporal.predict(vc_feat[FeatureBuilderV5.COLS].values.astype(np.float32))
    vc_feat['score_full'] = model_full.predict(fb_all.featurize(val_cands[['user_id','edition_id']])[FeatureBuilderV5.COLS].values.astype(np.float32))

    # grid
    ensemble_weights = [0.5, 0.6, 0.7]
    lambdas = [0.94, 0.95, 0.96]
    pools = [60, 80]
    best = None
    val_users = set(val_cands['user_id']) & set(gt['user_id'])
    print('Starting grid search')
    for w in ensemble_weights:
        for lam in lambdas:
            for pool in pools:
                s1 = vc_feat['score_temporal'].values; s2 = vc_feat['score_full'].values
                s1n = (s1 - s1.min()) / (s1.max() - s1.min() + 1e-12)
                s2n = (s2 - s2.min()) / (s2.max() - s2.min() + 1e-12)
                vc_feat['score'] = w * s1n + (1.0 - w) * s2n
                sub = mmr_rerank_df(vc_feat[['user_id','edition_id','score']], egm, lam=lam, pool=pool, top_k=TOP_K)
                mn, md, sc = compute_score(sub, gt_dict, egm, val_users)
                print(f'w={w:.2f} lam={lam:.2f} pool={pool} -> NDCG={mn:.5f} Div={md:.5f} Score={sc:.5f}')
                if best is None or sc > best[0]:
                    best = (sc, w, lam, pool)
    print('Best on val:', best)

    # score competition candidates with best params
    s1c = fb_all.featurize(cands[['user_id','edition_id']])
    s1c['score_temporal'] = model_temporal.predict(fb_all.featurize(cands[['user_id','edition_id']])[FeatureBuilderV5.COLS].values.astype(np.float32))
    s1c['score_full'] = model_full.predict(fb_all.featurize(cands[['user_id','edition_id']])[FeatureBuilderV5.COLS].values.astype(np.float32))
    w = best[1]; lam = best[2]; pool = best[3]
    s1n = (s1c['score_temporal'].values - s1c['score_temporal'].values.min()) / (s1c['score_temporal'].values.max() - s1c['score_temporal'].values.min() + 1e-12)
    s2n = (s1c['score_full'].values - s1c['score_full'].values.min()) / (s1c['score_full'].values.max() - s1c['score_full'].values.min() + 1e-12)
    s1c['score'] = w * s1n + (1.0 - w) * s2n
    final_sub = mmr_rerank_df(s1c[['user_id','edition_id','score']], egm, lam=lam, pool=pool, top_k=TOP_K)
    final_sub.to_csv(OUT1, index=False)
    import hashlib
    md5 = hashlib.md5(open(OUT1,'rb').read()).hexdigest()
    print('Saved', OUT1, 'md5', md5)
    print('Total time', time.time() - t0)
