import os, math
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'materials')
SUB = os.path.join(BASE, 'submission.csv')

inter = pd.read_csv(os.path.join(DATA, 'interactions.csv'))
inter['event_ts'] = pd.to_datetime(inter['event_ts'])
max_ts = inter['event_ts'].max()
val_cutoff = max_ts - pd.Timedelta(days=30)
T2 = inter[inter['event_ts'] >= val_cutoff].copy()
T2['rel'] = T2['event_type'].map({1:1,2:3})

# ground truth
gt = T2.groupby(['user_id','edition_id'])['rel'].max().reset_index()
gt_dict = {(r.user_id, r.edition_id): r.rel for r in gt.itertuples()}

# load submission
sub = pd.read_csv(SUB)

# restrict to target users present in GT
users = sorted(set(sub['user_id']).intersection({u for (u,e) in gt_dict.keys()}))

TOP_K = 20

# helper
import math
_EMPTY = frozenset()

# need book_genres for genres mapping
bg = pd.read_csv(os.path.join(DATA, 'book_genres.csv'))
eds = pd.read_csv(os.path.join(DATA, 'editions.csv'))
e2b = eds.set_index('edition_id')['book_id'].to_dict()
bgm = bg.groupby('book_id')['genre_id'].apply(lambda x: set(x)).to_dict()

def get_genres(eid):
    bid = e2b.get(eid)
    if bid is None: return set()
    return bgm.get(bid, set())

ndcgs = []
divs = []
count = 0
for uid in users:
    grp = sub[sub['user_id'] == uid].sort_values('rank')
    if len(grp) < TOP_K:
        continue
    ranked = grp['edition_id'].values[:TOP_K]
    rels = [gt_dict.get((uid, int(e)), 0) for e in ranked]
    # ndcg
    dcg = sum(r / math.log2(k+2) for k,r in enumerate(rels))
    all_rels = sorted([v for (u,e),v in gt_dict.items() if u==uid], reverse=True)
    ideal = (all_rels + [0]*TOP_K)[:TOP_K]
    idcg = sum(r / math.log2(k+2) for k,r in enumerate(ideal))
    ndcg = dcg / idcg if idcg>0 else 0.0
    ndcgs.append(ndcg)

    # diversity
    trel = [1 if r>0 else 0 for r in rels]
    gsets = [get_genres(int(e)) for e in ranked]
    w = [1.0 / math.log2(k+2) for k in range(TOP_K)]
    wsum = sum(w)
    covered = set(); coverage = 0.0
    for k in range(TOP_K):
        if trel[k] and len(gsets[k])>0:
            coverage += w[k] * len(gsets[k] - covered) / len(gsets[k])
            covered |= gsets[k]
    coverage = coverage / wsum if wsum>0 else 0.0
    # ILD
    L = [i for i in range(TOP_K) if trel[i]==1]
    if len(L) < 2:
        ild = 0.0
    else:
        s = 0.0; cnt = 0
        for i in range(len(L)):
            for j in range(i+1, len(L)):
                gi = gsets[L[i]]; gj = gsets[L[j]]
                u = len(gi | gj)
                if u>0:
                    s += 1.0 - len(gi & gj) / u
                cnt += 1
        ild = s / cnt if cnt>0 else 0.0
    div = 0.5 * coverage + 0.5 * ild
    divs.append(div)
    count += 1

mean_ndcg = np.mean(ndcgs) if ndcgs else 0.0
mean_div = np.mean(divs) if divs else 0.0
score = 0.7 * mean_ndcg + 0.3 * mean_div
print(f'Users evaluated: {count}')
print(f'NDCG@20 = {mean_ndcg:.6f}')
print(f'Diversity@20 = {mean_div:.6f}')
print(f'Score = {score:.6f}')
