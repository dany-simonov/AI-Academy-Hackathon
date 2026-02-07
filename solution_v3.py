"""
BookRec Hackathon — Solution v3 (ensemble: temporal + full-data model)
=====================================================================
Improvements over v2:
  - Temporal model with max_iter=1500 (was 500)
  - Additional full-data model (captures general preference patterns)
  - Ensemble: 0.6*temporal + 0.4*full-data
  - λ=0.95 from validated v2 evaluation
  - Additional features: per-user genre reading rate, item conversion
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
OUT  = os.path.join(BASE, "submission.csv")

TOP_K      = 20
SVD_K      = 64
MMR_POOL   = 80
BEST_LAM   = 0.95   # from v2 validation
_EMPTY     = frozenset()

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
    print(f"[load] {len(inter):,} inter  {len(targets):,} users  {len(cands):,} cands  ({time.time()-t0:.1f}s)")
    return inter, users, eds, bg, targets, cands

def build_maps(eds, bg):
    e2b = eds.set_index("edition_id")["book_id"].to_dict()
    e2a = eds.set_index("edition_id")["author_id"].to_dict()
    bgm = bg.groupby("book_id")["genre_id"].apply(frozenset).to_dict()
    egm = {eid: bgm.get(bid, _EMPTY) for eid, bid in e2b.items()}
    egs = {eid: len(g) for eid, g in egm.items()}
    return e2b, e2a, bgm, egm, egs

def do_svd(inter, all_u, all_e, k=SVD_K, half_life=180):
    tmp = inter.copy()
    tmp["rel"] = tmp["event_type"].map({1:1.0, 2:3.0})
    ui = tmp.groupby(["user_id","edition_id"]).agg(
        rel=("rel","max"), ts=("event_ts","max")).reset_index()
    ref = inter["event_ts"].max()
    days = (ref - ui["ts"]).dt.days.values.astype(np.float32)
    ui["val"] = np.log1p(ui["rel"].values) * np.exp(-days / half_life)
    u_enc = {u:i for i,u in enumerate(all_u)}
    e_enc = {e:i for i,e in enumerate(all_e)}
    m = ui["user_id"].isin(u_enc) & ui["edition_id"].isin(e_enc)
    uiv = ui[m]
    R = csr_matrix(
        (uiv["val"].values.astype(np.float32),
         (uiv["user_id"].map(u_enc).values, uiv["edition_id"].map(e_enc).values)),
        shape=(len(u_enc), len(e_enc)))
    ak = min(k, min(R.shape)-1)
    U, s, Vt = svds(R, k=ak)
    idx = np.argsort(-s); U=U[:,idx]; s=s[idx]; Vt=Vt[idx,:]
    print(f"   SVD {R.shape} nnz={R.nnz:,} k={ak}")
    return U*s[np.newaxis,:], Vt.T, u_enc, e_enc

# ─────────────────────────────────────────────────────────────────────────────
class FeatureBuilder:
    COLS = [
        "svd",
        "u_n","u_reads","u_wish","u_avgr","u_maxr","u_stdr","u_ned",
        "u_rr","u_rec","u_span","u_rate","u_nb","u_na","gender","age",
        "u_ng","u_tgs","u_gent",
        "i_pop","i_nu","i_reads","i_wish","i_avgr","i_stdr","i_rr","i_rec",
        "i_bpop","i_apop","i_ng","publication_year","age_restriction","language_id",
        "i_trend",
        "ua_cnt","ua_rd","ua_avgr_feat","ui_sb","ui_jac","ui_wov","ui_newg",
        # new in v3
        "ui_genre_read_rate","i_conversion",
    ]

    def __init__(self, inter, users, eds, e2b, e2a, bgm, egm, egs,
                 user_f, item_f, u_enc, e_enc):
        self.egm=egm; self.egs=egs; self.e2a=e2a
        self.user_f=user_f; self.item_f=item_f; self.u_enc=u_enc; self.e_enc=e_enc
        max_ts = inter["event_ts"].max()

        # user stats
        uf = inter.groupby("user_id").agg(
            u_n=("event_type","count"),
            u_reads=("event_type",lambda x:(x==2).sum()),
            u_wish=("event_type",lambda x:(x==1).sum()),
            u_avgr=("rating","mean"),u_maxr=("rating","max"),u_stdr=("rating","std"),
            u_ned=("edition_id","nunique"),
            u_last=("event_ts","max"),u_first=("event_ts","min"),
        ).reset_index()
        uf["u_rr"]=uf["u_reads"]/uf["u_n"].clip(1)
        uf["u_rec"]=(max_ts-uf["u_last"]).dt.days.astype(float)
        uf["u_span"]=(uf["u_last"]-uf["u_first"]).dt.days.clip(1).astype(float)
        uf["u_rate"]=uf["u_n"]/uf["u_span"]*30
        uf.drop(columns=["u_last","u_first"],inplace=True)

        im = inter.merge(eds[["edition_id","book_id","author_id"]],on="edition_id",how="left")
        uf = uf.merge(im.groupby("user_id")["book_id"].nunique().reset_index(name="u_nb"),on="user_id",how="left")
        uf = uf.merge(im.groupby("user_id")["author_id"].nunique().reset_index(name="u_na"),on="user_id",how="left")
        uf = uf.merge(users[["user_id","gender","age"]],on="user_id",how="left")

        # genre profile
        ugp={}; ugs={}
        # also: user genre-specific reading rate
        self.u_genre_read_rate = {}  # (user, genre) -> read_rate
        for uid, grp in im.groupby("user_id"):
            gc=defaultdict(float); gc_reads=defaultdict(float); gc_total=defaultdict(float)
            for et, bid in zip(grp["event_type"],grp["book_id"]):
                if pd.notna(bid):
                    for g in bgm.get(int(bid),_EMPTY):
                        gc[g]+=(3 if et==2 else 1)
                        gc_total[g]+=1
                        if et==2: gc_reads[g]+=1
            ugp[uid]=dict(gc); ugs[uid]=set(gc.keys())
            for g in gc_total:
                self.u_genre_read_rate[(uid,g)] = gc_reads[g]/gc_total[g]
        rows=[]
        for uid in uf["user_id"]:
            gc=ugp.get(uid,{}); tot=sum(gc.values()) if gc else 0
            rows.append({"user_id":uid,"u_ng":len(gc),
                "u_tgs":max(gc.values())/max(tot,1) if gc else 0,
                "u_gent":-sum((v/tot)*np.log2(v/tot+1e-12) for v in gc.values()) if tot else 0})
        uf = uf.merge(pd.DataFrame(rows),on="user_id",how="left")
        self.uf=uf; self.ugp=ugp; self.ugs=ugs

        ua = im.groupby(["user_id","author_id"]).agg(
            c=("event_type","count"),r=("event_type",lambda x:(x==2).sum()),
            ar=("rating","mean")).reset_index()
        self.ua_dict={(row.user_id,row.author_id):(row.c,row.r,row.ar) for row in ua.itertuples()}
        self.ubs=im.groupby("user_id")["book_id"].apply(set).to_dict()

        # item stats
        itf = inter.groupby("edition_id").agg(
            i_pop=("event_type","count"),i_nu=("user_id","nunique"),
            i_reads=("event_type",lambda x:(x==2).sum()),
            i_wish=("event_type",lambda x:(x==1).sum()),
            i_avgr=("rating","mean"),i_stdr=("rating","std"),
            i_last=("event_ts","max"),
        ).reset_index()
        itf["i_rr"]=itf["i_reads"]/itf["i_pop"].clip(1)
        itf["i_rec"]=(max_ts-itf["i_last"]).dt.days.astype(float)
        itf.drop(columns=["i_last"],inplace=True)

        # trending
        r7=inter[inter["event_ts"]>=max_ts-pd.Timedelta(days=7)]
        r30=inter[inter["event_ts"]>=max_ts-pd.Timedelta(days=30)]
        c7=r7.groupby("edition_id")["event_type"].count().reset_index(name="c7")
        c30=r30.groupby("edition_id")["event_type"].count().reset_index(name="c30")
        tr=c7.merge(c30,on="edition_id",how="outer").fillna(0)
        tr["i_trend"]=tr["c7"]/tr["c30"].clip(1)
        itf=itf.merge(tr[["edition_id","i_trend"]],on="edition_id",how="left")
        itf["i_trend"]=itf["i_trend"].fillna(0)

        # item conversion (reads / total)
        itf["i_conversion"] = itf["i_reads"] / itf["i_pop"].clip(1)

        bp=im.groupby("book_id")["event_type"].count().reset_index(name="i_bpop")
        ap=im.groupby("author_id")["event_type"].count().reset_index(name="i_apop")
        em=eds[["edition_id","book_id","author_id","publication_year","age_restriction","language_id"]].copy()
        em=em.merge(bp,on="book_id",how="left").fillna({"i_bpop":0})
        em=em.merge(ap,on="author_id",how="left").fillna({"i_apop":0})
        em["i_ng"]=em["edition_id"].map(lambda x:len(egm.get(x,_EMPTY)))
        self.itf=itf; self.em=em
        self.e2b_local=em.set_index("edition_id")["book_id"].to_dict()
        self.bgm = bgm

    def featurize(self, pairs_df):
        df=pairs_df.copy()
        uids=df["user_id"].values; eids=df["edition_id"].values; N=len(df)

        svd=np.zeros(N,np.float32)
        for i in range(N):
            ui=self.u_enc.get(uids[i]); ei=self.e_enc.get(eids[i])
            if ui is not None and ei is not None:
                svd[i]=np.dot(self.user_f[ui],self.item_f[ei])
        df["svd"]=svd

        df=df.merge(self.uf,on="user_id",how="left")
        df=df.merge(self.itf,on="edition_id",how="left")
        df=df.merge(self.em,on="edition_id",how="left",suffixes=("","_em"))

        aids=[self.e2a.get(e) for e in eids]
        keys=list(zip(uids,aids))
        df["ua_cnt"]=[self.ua_dict.get(k,(0,0,0))[0] for k in keys]
        df["ua_rd"]=[self.ua_dict.get(k,(0,0,0))[1] for k in keys]
        df["ua_avgr_feat"]=[self.ua_dict.get(k,(0,0,np.nan))[2] for k in keys]

        sb=np.zeros(N,np.int8)
        for i in range(N):
            bid=self.e2b_local.get(eids[i])
            if bid is not None and bid in self.ubs.get(uids[i],set()): sb[i]=1
        df["ui_sb"]=sb

        jac=np.zeros(N,np.float32); wov=np.zeros(N,np.float32)
        newg=np.zeros(N,np.float32); grr=np.zeros(N,np.float32)
        for i in range(N):
            uid=uids[i]; eid=eids[i]
            ug=self.ugs.get(uid); ig=self.egm.get(eid,_EMPTY); ig_sz=self.egs.get(eid,0)
            if not ug or ig_sz==0:
                newg[i]=1.0 if ig_sz else 0.0; continue
            isz=len(ug&ig); usz=len(ug)+ig_sz-isz
            jac[i]=isz/usz if usz else 0
            gp=self.ugp.get(uid,{}); w=sum(gp.get(g,0) for g in ig)
            tw=sum(gp.values()); wov[i]=w/tw if tw else 0
            newg[i]=(ig_sz-isz)/ig_sz
            # user's read rate for this item's genres
            rates = [self.u_genre_read_rate.get((uid,g), 0) for g in ig]
            grr[i] = np.mean(rates) if rates else 0
        df["ui_jac"]=jac; df["ui_wov"]=wov; df["ui_newg"]=newg
        df["ui_genre_read_rate"]=grr

        df.fillna(0,inplace=True)
        return df

# ─────────────────────────────────────────────────────────────────────────────
def make_training(src_inter, label_inter, fb):
    """Create pos/neg training from label_inter, features from fb."""
    t0=time.time()
    lbl=label_inter.copy()
    lbl["rel"]=lbl["event_type"].map({1:1.0,2:3.0})
    pos=lbl.groupby(["user_id","edition_id"])["rel"].max().reset_index().rename(columns={"rel":"label"})

    all_e=list(set(src_inter["edition_id"]))
    u_all=defaultdict(set)
    for uid,eid in zip(src_inter["user_id"],src_inter["edition_id"]): u_all[uid].add(eid)
    for uid,eid in zip(label_inter["user_id"],label_inter["edition_id"]): u_all[uid].add(eid)

    negs=[]
    for uid in pos["user_id"].unique():
        np_=len(pos[pos["user_id"]==uid]); nn=min(np_*5,300)
        excl=u_all[uid]; pool=[e for e in all_e if e not in excl]
        if len(pool)>nn: ch=np.random.choice(pool,nn,replace=False)
        else: ch=pool
        for eid in ch: negs.append({"user_id":uid,"edition_id":eid,"label":0.0})
    neg_df=pd.DataFrame(negs)
    train=pd.concat([pos,neg_df],ignore_index=True).sample(frac=1,random_state=42)
    print(f"   train: {len(pos):,}p + {len(neg_df):,}n = {len(train):,}")

    feat=fb.featurize(train[["user_id","edition_id"]])
    feat["label"]=train["label"].values
    print(f"   featurized ({time.time()-t0:.1f}s)")
    return feat

def train_gbm(feat, max_iter=1500):
    t0=time.time()
    X=feat[FeatureBuilder.COLS].values.astype(np.float32)
    y=feat["label"].values.astype(np.float32)
    mdl=HistGradientBoostingRegressor(
        max_iter=max_iter,max_depth=7,learning_rate=0.05,
        min_samples_leaf=30,l2_regularization=1.0,max_bins=255,
        random_state=42,early_stopping=True,
        validation_fraction=0.1,n_iter_no_change=50)
    mdl.fit(X,y)
    print(f"   GBM n_iter={mdl.n_iter_} ({time.time()-t0:.1f}s)")
    return mdl

# ─────────────────────────────────────────────────────────────────────────────
def mmr_rerank(scored_df, egm, lam=BEST_LAM, pool=MMR_POOL, top_k=TOP_K):
    t0=time.time()
    results=[]
    for uid, grp in scored_df.groupby("user_id"):
        gs=grp.nlargest(pool,"score")
        eids=gs["edition_id"].values; scores=gs["score"].values.astype(np.float64)
        smin,smax=scores.min(),scores.max()
        sn=(scores-smin)/(smax-smin) if smax>smin else np.ones_like(scores)
        gsets=[egm.get(int(e),_EMPTY) for e in eids]
        sel=[]; cov=set(); rem=list(range(len(eids)))
        for _ in range(top_k):
            bi=-1; bv=-1e18
            for idx in rem:
                ig=gsets[idx]; ig_sz=len(ig)
                cg=len(ig-cov)/ig_sz if ig_sz else 0
                if sel:
                    ds=0.0
                    for si in sel:
                        sg=gsets[si]; u=len(ig|sg)
                        ds+=(1-len(ig&sg)/u) if u else 0
                    ad=ds/len(sel)
                else: ad=1.0
                v=lam*sn[idx]+(1-lam)*(0.5*cg+0.5*ad)
                if v>bv: bv=v; bi=idx
            sel.append(bi); cov|=gsets[bi]; rem.remove(bi)
        for rank,idx in enumerate(sel,1):
            results.append((uid,int(eids[idx]),rank))
    out=pd.DataFrame(results,columns=["user_id","edition_id","rank"])
    print(f"   MMR {len(out):,} rows ({time.time()-t0:.1f}s)")
    return out

def validate_sub(sub, cands, targets):
    tu=set(targets["user_id"]); su=set(sub["user_id"])
    assert su==tu, f"Users {len(su)} vs {len(tu)}"
    cm=cands.groupby("user_id")["edition_id"].apply(set).to_dict()
    for uid,g in sub.groupby("user_id"):
        assert len(g)==TOP_K; assert set(g["rank"])==set(range(1,TOP_K+1))
        assert g["edition_id"].nunique()==TOP_K
        assert set(g["edition_id"]).issubset(cm[uid])
    print("   ✓ valid!")

# ─────────────────────────────────────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(sub, gt_dict, egm, target_users):
    ndcgs=[]; divs=[]
    for uid in target_users:
        ugrp=sub[sub["user_id"]==uid]
        if len(ugrp)==0: continue
        ranked=ugrp.sort_values("rank")["edition_id"].values[:TOP_K]
        rels=[gt_dict.get((uid,int(e)),0) for e in ranked]
        dcg=sum(r/math.log2(k+2) for k,r in enumerate(rels))
        all_r=sorted([v for (u,_),v in gt_dict.items() if u==uid],reverse=True)
        ideal=(all_r+[0]*TOP_K)[:TOP_K]
        idcg=sum(r/math.log2(k+2) for k,r in enumerate(ideal))
        ndcg=dcg/idcg if idcg>0 else 0
        trel=[1 if r>0 else 0 for r in rels]
        gsets=[egm.get(int(e),_EMPTY) for e in ranked]
        w=[1/math.log2(k+2) for k in range(TOP_K)]; ws=sum(w)
        cv=0.0; cov=set()
        for k in range(TOP_K):
            if trel[k] and gsets[k]:
                cv+=w[k]*len(gsets[k]-cov)/len(gsets[k]); cov|=gsets[k]
        coverage=cv/ws if ws else 0
        ri=[k for k in range(TOP_K) if trel[k]]
        if len(ri)<2: ild=0.0
        else:
            ds=0.0; cnt=0
            for i in range(len(ri)):
                for j in range(i+1,len(ri)):
                    gi,gj=gsets[ri[i]],gsets[ri[j]]
                    u=len(gi|gj); ds+=(1-len(gi&gj)/u) if u else 0; cnt+=1
            ild=ds/cnt if cnt else 0
        divs.append(0.5*coverage+0.5*ild); ndcgs.append(ndcg)
    mn=np.mean(ndcgs); md=np.mean(divs)
    return mn,md,0.7*mn+0.3*md

# ═══════════════════════════════════════════════════════════════════════════════
def main():
    wall=time.time()
    print("="*60)
    print("  BookRec v3 — ensemble (temporal + full-data)")
    print("="*60)

    inter,users,eds,bg,targets,cands = load_data()
    e2b,e2a,bgm,egm,egs = build_maps(eds,bg)
    all_u=sorted(set(targets["user_id"])|set(inter["user_id"]))
    all_e=sorted(set(inter["edition_id"])|set(cands["edition_id"]))

    max_ts=inter["event_ts"].max()
    val_cut=max_ts-pd.Timedelta(days=30)
    T1=inter[inter["event_ts"]<val_cut].copy()
    T2=inter[inter["event_ts"]>=val_cut].copy()
    print(f"\n[split] T1 {len(T1):,}  T2 {len(T2):,}")

    # GT
    T2c=T2.copy(); T2c["rel"]=T2c["event_type"].map({1:1,2:3})
    gt=T2c.groupby(["user_id","edition_id"])["rel"].max().reset_index()
    gt_dict={(r.user_id,r.edition_id):r.rel for r in gt.itertuples()}

    # ══════════════════════════════════════════════════════════════════════════
    #  MODEL 1: Temporal (T1 features → T2 labels)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n── Model 1: temporal (T1→T2) ──")
    print("[svd T1]")
    uf1,if1,ue1,ee1 = do_svd(T1,all_u,all_e)
    print("[features T1]")
    fb1 = FeatureBuilder(T1,users,eds,e2b,e2a,bgm,egm,egs,uf1,if1,ue1,ee1)
    print("[train T1→T2]")
    tf1 = make_training(T1,T2,fb1)
    model_temporal = train_gbm(tf1, max_iter=1500)

    # ── Quick local eval on proper val candidates ──
    T2_ue=T2.groupby("user_id")["edition_id"].apply(set).to_dict()
    T1_ue=T1.groupby("user_id")["edition_id"].apply(set).to_dict()
    ae=list(set(T1["edition_id"]))
    vc_rows=[]
    for uid,ps in T2_ue.items():
        if uid not in T1_ue: continue
        nn=max(200-len(ps),100)
        excl=T1_ue.get(uid,set())|ps; pool=[e for e in ae if e not in excl]
        if len(pool)>nn: ns=list(np.random.choice(pool,nn,replace=False))
        else: ns=pool
        for eid in list(ps)+ns: vc_rows.append({"user_id":uid,"edition_id":eid})
    vc_df=pd.DataFrame(vc_rows)
    vc_feat=fb1.featurize(vc_df[["user_id","edition_id"]])
    vc_feat["score"]=model_temporal.predict(vc_feat[FeatureBuilder.COLS].values.astype(np.float32))
    val_sub=mmr_rerank(vc_feat,egm,lam=BEST_LAM)
    vu=set(vc_df["user_id"])&set(gt["user_id"])
    mn,md,sc=compute_metrics(val_sub,gt_dict,egm,vu)
    print(f"   LOCAL EVAL (temporal):  NDCG={mn:.5f}  Div={md:.5f}  Score={sc:.5f}")

    # ══════════════════════════════════════════════════════════════════════════
    #  MODEL 2: Full-data (all interactions for features + labels)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n── Model 2: full-data ──")
    print("[svd full]")
    uf_a,if_a,ue_a,ee_a = do_svd(inter,all_u,all_e)
    print("[features full]")
    fb_all = FeatureBuilder(inter,users,eds,e2b,e2a,bgm,egm,egs,uf_a,if_a,ue_a,ee_a)
    print("[train full-data]")
    # For full-data model: positives = all interactions, negatives = random
    tmp=inter.copy()
    tmp["rel"]=tmp["event_type"].map({1:1.0,2:3.0})
    pos_full=tmp.groupby(["user_id","edition_id"])["rel"].max().reset_index().rename(columns={"rel":"label"})
    u_inter_all=inter.groupby("user_id")["edition_id"].apply(set).to_dict()
    ae_full=list(set(inter["edition_id"]))
    negs2=[]
    for uid,eset in u_inter_all.items():
        nn=min(len(eset)*5,500)
        pool=[e for e in ae_full if e not in eset]
        if len(pool)>nn: ch=np.random.choice(pool,nn,replace=False)
        else: ch=pool
        for eid in ch: negs2.append({"user_id":uid,"edition_id":eid,"label":0.0})
    neg2=pd.DataFrame(negs2)
    train_full=pd.concat([pos_full,neg2],ignore_index=True).sample(frac=1,random_state=42)
    print(f"   {len(pos_full):,}p + {len(neg2):,}n = {len(train_full):,}")
    tf_full=fb_all.featurize(train_full[["user_id","edition_id"]])
    tf_full["label"]=train_full["label"].values
    print(f"   featurized")
    model_full = train_gbm(tf_full, max_iter=1500)

    # ══════════════════════════════════════════════════════════════════════════
    #  ENSEMBLE: score competition candidates
    # ══════════════════════════════════════════════════════════════════════════
    print("\n── Ensemble scoring ──")
    cf = fb_all.featurize(cands[["user_id","edition_id"]])
    X_cand = cf[FeatureBuilder.COLS].values.astype(np.float32)

    # Temporal model scored on full-data features
    s1 = model_temporal.predict(X_cand)
    # Full-data model scored on full-data features
    s2 = model_full.predict(X_cand)

    # Normalize each to [0,1] before blending
    s1n = (s1 - s1.min()) / (s1.max()-s1.min()+1e-12)
    s2n = (s2 - s2.min()) / (s2.max()-s2.min()+1e-12)

    # Ensemble: temporal model has the better generalisation signal
    cf["score"] = 0.6*s1n + 0.4*s2n

    print(f"[mmr λ={BEST_LAM}]")
    sub = mmr_rerank(cf,egm,lam=BEST_LAM)
    sub.to_csv(OUT,index=False)
    print(f"   saved → {OUT}")
    validate_sub(sub,cands,targets)
    print(f"\n   Total: {time.time()-wall:.0f}s")

if __name__=="__main__":
    main()
