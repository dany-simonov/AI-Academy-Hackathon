"""
Local evaluation on a time-split of the training interactions.
Computes: Score = 0.7*mean(NDCG@20) + 0.3*mean(Diversity@20)
exactly as described in rulls.txt.
"""

import os, time, warnings, math
import numpy as np
import pandas as pd
from collections import defaultdict

warnings.filterwarnings("ignore")
np.random.seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "materials")

# ═══════════════════════════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading data …")
inter   = pd.read_csv(os.path.join(DATA, "interactions.csv"))
eds     = pd.read_csv(os.path.join(DATA, "editions.csv"))
bg      = pd.read_csv(os.path.join(DATA, "book_genres.csv"))
users   = pd.read_csv(os.path.join(DATA, "users.csv"))
targets = pd.read_csv(os.path.join(DATA, "targets.csv"))
cands   = pd.read_csv(os.path.join(DATA, "candidates.csv"))

inter["event_ts"] = pd.to_datetime(inter["event_ts"])
max_ts = inter["event_ts"].max()

_EMPTY = frozenset()

# Mappings
e2b = eds.set_index("edition_id")["book_id"].to_dict()
bgm = bg.groupby("book_id")["genre_id"].apply(frozenset).to_dict()
egm = {eid: bgm.get(bid, _EMPTY) for eid, bid in e2b.items()}

# ═══════════════════════════════════════════════════════════════════════════════
# Time split: last 30 days → validation "future"
# ═══════════════════════════════════════════════════════════════════════════════
val_cutoff = max_ts - pd.Timedelta(days=30)
train_inter = inter[inter["event_ts"] < val_cutoff]
val_inter   = inter[inter["event_ts"] >= val_cutoff]

print(f"Train: {len(train_inter):,}  Val: {len(val_inter):,}")
print(f"Date split: {val_cutoff}")

# Ground truth: best relevance per (user, edition) in val period
val_inter = val_inter.copy()
val_inter["rel"] = val_inter["event_type"].map({1: 1, 2: 3})
gt = val_inter.groupby(["user_id","edition_id"])["rel"].max().reset_index()
gt_dict = {(r.user_id, r.edition_id): r.rel for r in gt.itertuples()}

# Users that exist in both periods AND are target users
val_users = set(gt["user_id"]) & set(targets["user_id"])
print(f"Validation users (in target & have val interactions): {len(val_users)}")

# ═══════════════════════════════════════════════════════════════════════════════
# Build a simplified model on train-only, generate recommendations
# We reuse the full solution's pipeline but on train_inter
# ═══════════════════════════════════════════════════════════════════════════════
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.ensemble import HistGradientBoostingRegressor

SVD_K = 64

# -- SVD on train_inter --
tmp = train_inter.copy()
tmp["rel_w"] = tmp["event_type"].map({1:1.0, 2:3.0})
ui = tmp.groupby(["user_id","edition_id"])["rel_w"].max().reset_index()
ts_map = tmp.groupby(["user_id","edition_id"])["event_ts"].max().reset_index()
ui = ui.merge(ts_map, on=["user_id","edition_id"], how="left")
days_ago = (val_cutoff - ui["event_ts"]).dt.days.values.astype(np.float32)
recency_w = np.exp(-days_ago / 180.0)
ui["val"] = np.log1p(ui["rel_w"].values) * recency_w

all_u = sorted(set(targets["user_id"]) | set(train_inter["user_id"]))
all_e = sorted(set(train_inter["edition_id"]) | set(cands["edition_id"]))
u_enc = {u: i for i, u in enumerate(all_u)}
e_enc = {e: i for i, e in enumerate(all_e)}

mask = ui["user_id"].isin(u_enc) & ui["edition_id"].isin(e_enc)
ui_v = ui[mask]
R = csr_matrix(
    (ui_v["val"].values.astype(np.float32),
     (ui_v["user_id"].map(u_enc).values, ui_v["edition_id"].map(e_enc).values)),
    shape=(len(u_enc), len(e_enc))
)
actual_k = min(SVD_K, min(R.shape)-1)
U, s, Vt = svds(R, k=actual_k)
idx = np.argsort(-s); U=U[:,idx]; s=s[idx]; Vt=Vt[idx,:]
user_f = U * s[np.newaxis, :]
item_f = Vt.T
print(f"SVD done: {R.shape}, nnz={R.nnz:,}")

# -- Features & model (simplified: a quick version) --
e2a = eds.set_index("edition_id")["author_id"].to_dict()
egs = {eid: len(gs) for eid, gs in egm.items()}

# user stats from train
max_train_ts = train_inter["event_ts"].max()
uf = train_inter.groupby("user_id").agg(
    u_n=("event_type","count"),
    u_reads=("event_type", lambda x:(x==2).sum()),
    u_wish=("event_type", lambda x:(x==1).sum()),
    u_avgr=("rating","mean"), u_maxr=("rating","max"), u_stdr=("rating","std"),
    u_ned=("edition_id","nunique"),
    u_last=("event_ts","max"), u_first=("event_ts","min"),
).reset_index()
uf["u_rr"]   = uf["u_reads"]/uf["u_n"].clip(1)
uf["u_rec"]  = (max_train_ts - uf["u_last"]).dt.days.astype(float)
uf["u_span"] = (uf["u_last"]-uf["u_first"]).dt.days.clip(1).astype(float)
uf["u_rate"] = uf["u_n"]/uf["u_span"]*30
uf.drop(columns=["u_last","u_first"], inplace=True)

im = train_inter.merge(eds[["edition_id","book_id","author_id"]], on="edition_id", how="left")
uf = uf.merge(im.groupby("user_id")["book_id"].nunique().reset_index(name="u_nb"), on="user_id", how="left")
uf = uf.merge(im.groupby("user_id")["author_id"].nunique().reset_index(name="u_na"), on="user_id", how="left")
uf = uf.merge(users[["user_id","gender","age"]], on="user_id", how="left")

# genre profile
ugp={}; ugs={}
for uid, grp in im.groupby("user_id"):
    gc=defaultdict(float)
    for et, bid in zip(grp["event_type"], grp["book_id"]):
        if pd.notna(bid):
            for g in bgm.get(int(bid), _EMPTY): gc[g]+=(3 if et==2 else 1)
    ugp[uid]=dict(gc); ugs[uid]=set(gc.keys())
rows=[]
for uid in uf["user_id"]:
    gc=ugp.get(uid,{}); tot=sum(gc.values()) if gc else 0
    rows.append({"user_id":uid,"u_ng":len(gc),
                 "u_tgs":max(gc.values())/max(tot,1) if gc else 0,
                 "u_gent":-sum((v/tot)*np.log2(v/tot+1e-12) for v in gc.values()) if tot else 0})
uf = uf.merge(pd.DataFrame(rows), on="user_id", how="left")

ua = im.groupby(["user_id","author_id"]).agg(
    c=("event_type","count"), r=("event_type",lambda x:(x==2).sum()),
    ar=("rating","mean")).reset_index()
ua_dict = {(row.user_id,row.author_id):(row.c,row.r,row.ar) for row in ua.itertuples()}
ubs = im.groupby("user_id")["book_id"].apply(set).to_dict()

# item stats
itf = train_inter.groupby("edition_id").agg(
    i_pop=("event_type","count"), i_nu=("user_id","nunique"),
    i_reads=("event_type",lambda x:(x==2).sum()),
    i_wish=("event_type",lambda x:(x==1).sum()),
    i_avgr=("rating","mean"), i_stdr=("rating","std"),
    i_last=("event_ts","max"),
).reset_index()
itf["i_rr"]  = itf["i_reads"]/itf["i_pop"].clip(1)
itf["i_rec"] = (max_train_ts - itf["i_last"]).dt.days.astype(float)
itf.drop(columns=["i_last"], inplace=True)

bp = im.groupby("book_id")["event_type"].count().reset_index(name="i_bpop")
ap = im.groupby("author_id")["event_type"].count().reset_index(name="i_apop")
em = eds[["edition_id","book_id","author_id","publication_year","age_restriction","language_id"]].copy()
em = em.merge(bp, on="book_id", how="left").fillna({"i_bpop":0})
em = em.merge(ap, on="author_id", how="left").fillna({"i_apop":0})
em["i_ng"] = em["edition_id"].map(lambda x: len(egm.get(x,_EMPTY)))
e2b_local = em.set_index("edition_id")["book_id"].to_dict()

FEATURE_COLS = [
    "svd","u_n","u_reads","u_wish","u_avgr","u_maxr","u_stdr","u_ned",
    "u_rr","u_rec","u_span","u_rate","u_nb","u_na","gender","age",
    "u_ng","u_tgs","u_gent",
    "i_pop","i_nu","i_reads","i_wish","i_avgr","i_stdr","i_rr","i_rec",
    "i_bpop","i_apop","i_ng","publication_year","age_restriction","language_id",
    "ua_cnt","ua_rd","ua_avgr","ui_sb","ui_jac","ui_wov","ui_newg",
]

def featurize(pairs_df):
    df = pairs_df.copy()
    uids = df["user_id"].values; eids = df["edition_id"].values
    svd = np.zeros(len(df), np.float32)
    for i in range(len(df)):
        ui_=u_enc.get(uids[i]); ei_=e_enc.get(eids[i])
        if ui_ is not None and ei_ is not None:
            svd[i]=np.dot(user_f[ui_], item_f[ei_])
    df["svd"]=svd
    df=df.merge(uf,on="user_id",how="left")
    df=df.merge(itf,on="edition_id",how="left")
    df=df.merge(em,on="edition_id",how="left",suffixes=("","_em"))
    aid=df["edition_id"].map(e2a)
    keys=list(zip(df["user_id"],aid))
    df["ua_cnt"]=[ua_dict.get(k,(0,0,0))[0] for k in keys]
    df["ua_rd"]=[ua_dict.get(k,(0,0,0))[1] for k in keys]
    df["ua_avgr"]=[ua_dict.get(k,(0,0,np.nan))[2] for k in keys]
    sb=np.zeros(len(df),np.int8)
    for i in range(len(df)):
        bid=e2b_local.get(eids[i])
        if bid is not None and bid in ubs.get(uids[i],set()): sb[i]=1
    df["ui_sb"]=sb
    jac=np.zeros(len(df),np.float32)
    wov=np.zeros(len(df),np.float32)
    newg=np.zeros(len(df),np.float32)
    for i in range(len(df)):
        uid=uids[i]; eid=eids[i]
        ug=ugs.get(uid); ig=egm.get(eid,_EMPTY); ig_sz=egs.get(eid,0)
        if not ug or ig_sz==0:
            newg[i]=1.0 if ig_sz else 0.0; continue
        isz=len(ug&ig); usz=len(ug)+ig_sz-isz
        jac[i]=isz/usz if usz else 0
        gp=ugp.get(uid,{}); w=sum(gp.get(g,0) for g in ig); tw=sum(gp.values())
        wov[i]=w/tw if tw else 0
        newg[i]=(ig_sz-isz)/ig_sz
    df["ui_jac"]=jac; df["ui_wov"]=wov; df["ui_newg"]=newg
    df.fillna(0,inplace=True)
    return df

# -- Training data --
print("Building training data …")
tmp2 = train_inter.copy()
tmp2["rel"]=tmp2["event_type"].map({1:1.0,2:3.0})
pos = tmp2.groupby(["user_id","edition_id"])["rel"].max().reset_index().rename(columns={"rel":"label"})
all_eids_train = list(set(train_inter["edition_id"]))
u_inter = train_inter.groupby("user_id")["edition_id"].apply(set).to_dict()
negs=[]
for uid, eset in u_inter.items():
    n=min(len(eset)*5, 500)
    pool=[e for e in all_eids_train if e not in eset]
    if len(pool)>n: chosen=np.random.choice(pool,n,replace=False)
    else: chosen=pool
    for eid in chosen: negs.append({"user_id":uid,"edition_id":eid,"label":0.0})
neg_df=pd.DataFrame(negs)
train_all=pd.concat([pos,neg_df],ignore_index=True).sample(frac=1,random_state=42)
print(f"  pos {len(pos):,}  neg {len(neg_df):,}")

print("Featurizing training data …")
train_feat = featurize(train_all[["user_id","edition_id"]])
train_feat["label"]=train_all["label"].values

print("Training model …")
X=train_feat[FEATURE_COLS].values.astype(np.float32)
y=train_feat["label"].values.astype(np.float32)
model=HistGradientBoostingRegressor(max_iter=500,max_depth=7,learning_rate=0.05,
    min_samples_leaf=30,l2_regularization=1.0,max_bins=255,random_state=42,
    early_stopping=True,validation_fraction=0.1,n_iter_no_change=30)
model.fit(X,y)
print(f"  n_iter={model.n_iter_}")

# -- Score candidates --
print("Scoring candidates …")
cf = featurize(cands[["user_id","edition_id"]])
cf["score"] = model.predict(cf[FEATURE_COLS].values.astype(np.float32))

# -- MMR reranking --
print("MMR reranking …")
MMR_LAMBDA = 0.60
MMR_POOL   = 80

results=[]
for uid, grp in cf.groupby("user_id"):
    gs=grp.nlargest(MMR_POOL,"score")
    eids_=gs["edition_id"].values; scores_=gs["score"].values.astype(np.float64)
    smin,smax=scores_.min(),scores_.max()
    snorm=(scores_-smin)/(smax-smin) if smax>smin else np.ones_like(scores_)
    gsets=[egm.get(int(e),_EMPTY) for e in eids_]
    sel=[]; cov=set(); rem=list(range(len(eids_)))
    for _k in range(20):
        best_i=-1; best_v=-1e18
        for idx in rem:
            rel=snorm[idx]; ig=gsets[idx]; ig_sz=len(ig)
            cov_g=len(ig-cov)/ig_sz if ig_sz else 0
            if sel:
                ds=0.0
                for si in sel:
                    sg=gsets[si]; usz=len(ig|sg)
                    ds+=1.0-len(ig&sg)/usz if usz else 0
                ad=ds/len(sel)
            else: ad=1.0
            v=MMR_LAMBDA*rel+(1-MMR_LAMBDA)*(0.5*cov_g+0.5*ad)
            if v>best_v: best_v=v; best_i=idx
        sel.append(best_i); cov|=gsets[best_i]; rem.remove(best_i)
    for rank,idx in enumerate(sel,1):
        results.append((uid,int(eids_[idx]),rank))
sub = pd.DataFrame(results, columns=["user_id","edition_id","rank"])

# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTE METRICS – exactly as in rulls.txt
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*50)
print("EVALUATING on time-split validation set")
print("="*50)

def ndcg_user(ranked_eids, gt_dict, uid):
    """NDCG@20 for one user."""
    rels = []
    for eid in ranked_eids:
        rels.append(gt_dict.get((uid, eid), 0))

    # DCG
    dcg = sum(r / math.log2(k+2) for k, r in enumerate(rels))

    # IDCG: ideal ordering of rels for THIS user among all possible candidates
    # Actually: ideal = sort all true rels descending, take top 20
    all_rels_for_user = [gt_dict.get((uid, eid), 0) for eid in ranked_eids]
    # We need IDEAL from ALL possible rels for this user
    user_all_rels = sorted([v for (u,e),v in gt_dict.items() if u==uid], reverse=True)
    # But IDCG@20 is based on the best possible 20 rels for this user
    ideal = user_all_rels[:20]
    while len(ideal) < 20:
        ideal.append(0)
    idcg = sum(r / math.log2(k+2) for k, r in enumerate(ideal))

    if idcg == 0:
        return 0.0
    return dcg / idcg

def diversity_user(ranked_eids, gt_dict, uid, egm):
    """Diversity@20 = 0.5*Coverage + 0.5*ILD for one user."""
    rels = [gt_dict.get((uid, eid), 0) for eid in ranked_eids]
    trel = [1 if r > 0 else 0 for r in rels]

    gsets = [egm.get(eid, _EMPTY) for eid in ranked_eids]

    # Coverage
    w = [1.0 / math.log2(k+2) for k in range(20)]
    w_sum = sum(w)
    covered = set()
    cov = 0.0
    for k in range(20):
        if trel[k] == 1 and len(gsets[k]) > 0:
            new_g = len(gsets[k] - covered)
            cov += w[k] * new_g / len(gsets[k])
            covered |= gsets[k]
    coverage = cov / w_sum if w_sum else 0

    # ILD among relevant items
    rel_idx = [k for k in range(20) if trel[k] == 1]
    if len(rel_idx) < 2:
        ild = 0.0
    else:
        dsum = 0.0
        cnt = 0
        for i in range(len(rel_idx)):
            for j in range(i+1, len(rel_idx)):
                gi = gsets[rel_idx[i]]
                gj = gsets[rel_idx[j]]
                union = len(gi | gj)
                if union > 0:
                    dsum += 1.0 - len(gi & gj) / union
                cnt += 1
        ild = dsum / cnt if cnt else 0

    return 0.5 * coverage + 0.5 * ild

ndcgs = []
divs = []
n_eval = 0

for uid in val_users:
    ugrp = sub[sub["user_id"] == uid].sort_values("rank")
    if len(ugrp) == 0:
        continue
    ranked = ugrp["edition_id"].values[:20]
    n = ndcg_user(ranked, gt_dict, uid)
    d = diversity_user(ranked, gt_dict, uid, egm)
    ndcgs.append(n)
    divs.append(d)
    n_eval += 1

mean_ndcg = np.mean(ndcgs) if ndcgs else 0
mean_div  = np.mean(divs)  if divs  else 0
score = 0.7 * mean_ndcg + 0.3 * mean_div

print(f"\nEvaluated on {n_eval} users (with val interactions & in targets)")
print(f"  mean NDCG@20      = {mean_ndcg:.6f}")
print(f"  mean Diversity@20 = {mean_div:.6f}")
print(f"  SCORE (0.7*NDCG + 0.3*Div) = {score:.6f}")
