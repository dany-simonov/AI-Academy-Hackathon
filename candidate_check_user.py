import pandas as pd
c = pd.read_csv('materials/candidates.csv')
print('Is 1010822636 in cands for user 560?', ((c['user_id']==560)&(c['edition_id']==1010822636)).any())
print('Is 1010744912 in cands for user 560?', ((c['user_id']==560)&(c['edition_id']==1010744912)).any())
print('Count of cands for user 560', (c['user_id']==560).sum())
