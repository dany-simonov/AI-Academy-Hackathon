import pandas as pd
s=pd.read_csv('submission_5.3.csv')
t=pd.read_csv('materials/targets.csv')
print('submission users sample:', s['user_id'].unique()[:10])
print('first target users sample:', t['user_id'].values[:10])
print('intersection size:', len(set(s['user_id']).intersection(set(t['user_id']))))
print('example non-intersection (first 20):', sorted(list(set(s['user_id']).difference(set(t['user_id']))))[:20])
