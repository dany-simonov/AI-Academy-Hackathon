import pandas as pd
s=pd.read_csv('submission_5.3.csv')
inter=pd.read_csv('materials/interactions.csv')
inter['event_ts']=pd.to_datetime(inter['event_ts'])
max_ts=inter['event_ts'].max(); val_cutoff=max_ts-pd.Timedelta(days=30)
T2=inter[inter['event_ts']>=val_cutoff]
uid=560
rec=s[s['user_id']==uid].sort_values('rank')['edition_id'].tolist()[:20]
positives=T2[(T2['user_id']==uid)]
print('N recs',len(rec))
print('Top recs',rec[:10])
print('Pos interactions in T2 (edition_ids):',positives['edition_id'].unique()[:20].tolist())
print('Intersection', set(rec)&set(positives['edition_id'].unique()))
