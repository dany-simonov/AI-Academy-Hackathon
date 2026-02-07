import pandas as pd
cur = pd.read_csv('submission_current.csv')
v2 = pd.read_csv('submission.csv')
print('submission_current.csv shape:', cur.shape)
print('submission_v2.csv shape:', v2.shape)
cur_set = set(tuple(x) for x in cur[['user_id','edition_id','rank']].values)
v2_set = set(tuple(x) for x in v2[['user_id','edition_id','rank']].values)
print('Exact match rows:', len(cur_set & v2_set))
print('Rows only in current:', len(cur_set - v2_set))
print('Rows only in v2:', len(v2_set - cur_set))
print('Jaccard similarity:', len(cur_set & v2_set) / len(cur_set | v2_set))
print('\nSample differences (current \\ v2) first 10:')
for r in list(cur_set - v2_set)[:10]:
    print(r)
print('\nSample differences (v2 \\ current) first 10:')
for r in list(v2_set - cur_set)[:10]:
    print(r)
