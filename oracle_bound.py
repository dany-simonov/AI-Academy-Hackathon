import pandas as pd, math, numpy as np
inter = pd.read_csv('materials/interactions.csv')
inter['event_ts']=pd.to_datetime(inter['event_ts'])
max_ts=inter['event_ts'].max(); val_cutoff=max_ts-pd.Timedelta(days=30)
T2=inter[inter['event_ts']>=val_cutoff].copy()
T2['rel']=T2['event_type'].map({1:1,2:3})
gt = T2.groupby(['user_id','edition_id'])['rel'].max().reset_index()
gt_dict = {(r.user_id,r.edition_id):r.rel for r in gt.itertuples()}

c = pd.read_csv('materials/candidates.csv')
users = sorted(set(c['user_id']).intersection(set([u for (u,e) in gt_dict.keys()])))
TOP_K=20
ndcgs=[]; divs=[]
# need book_genres
bg = pd.read_csv('materials/book_genres.csv')
eds = pd.read_csv('materials/editions.csv')
e2b = eds.set_index('edition_id')['book_id'].to_dict()
bgm = bg.groupby('book_id')['genre_id'].apply(lambda x:set(x)).to_dict()
_EMPTY = frozenset()

def get_genres(eid):
    bid = e2b.get(eid)
    if bid is None: return set()
    return bgm.get(bid, set())

for uid in users:
    user_cands = c[c['user_id']==uid]['edition_id'].tolist()
    # rank positives present in cands by rel
    pos_in_cands = [(e, gt_dict.get((uid,e),0)) for e in user_cands]
    pos_sorted = sorted(pos_in_cands, key=lambda x:-x[1])
    ranked = [e for e,_ in pos_sorted][:TOP_K]
    rels = [gt_dict.get((uid,int(e)),0) for e in ranked]
    dcg = sum(r / math.log2(k+2) for k,r in enumerate(rels))
    all_r = sorted([v for (u,e),v in gt_dict.items() if u==uid], reverse=True)
    ideal = (all_r + [0]*TOP_K)[:TOP_K]
    idcg = sum(r / math.log2(k+2) for k,r in enumerate(ideal))
    ndcg = dcg / idcg if idcg>0 else 0
    ndcgs.append(ndcg)
    # diversity
    trel = [1 if r>0 else 0 for r in rels]
    gsets=[get_genres(int(e)) for e in ranked]
    w=[1/math.log2(k+2) for k in range(TOP_K)]; ws=sum(w)
    covered=set(); cov_val=0
    for k in range(TOP_K):
        if trel[k] and gsets[k]:
            cov_val += w[k]*len(gsets[k]-covered)/len(gsets[k])
            covered |= gsets[k]
    coverage = cov_val/ws if ws else 0
    L=[k for k in range(TOP_K) if trel[k]==1]
    if len(L)<2: ild=0
    else:
        s=0; cnt=0
        for i in range(len(L)):
            for j in range(i+1,len(L)):
                gi=gsets[L[i]]; gj=gsets[L[j]]; u=len(gi|gj)
                if u>0: s += 1 - len(gi & gj)/u
                cnt +=1
        ild = s/cnt if cnt else 0
    divs.append(0.5*coverage+0.5*ild)

print('users',len(users))
print('oracle NDCG mean', np.mean(ndcgs))
print('oracle Div mean', np.mean(divs))
print('oracle combined', 0.7*np.mean(ndcgs)+0.3*np.mean(divs))
