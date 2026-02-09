


import os

import time

import math

import warnings

import numpy as np

import pandas as pd

from collections import defaultdict

from scipy.sparse import csr_matrix

from sklearn.decomposition import TruncatedSVD

from sklearn.ensemble import HistGradientBoostingRegressor



warnings.filterwarnings("ignore")

np.random.seed(42)



# --- 1. Load Data ---

print("Loading data...")

BASE = os.path.dirname(os.path.abspath(__file__))

DATA = os.path.join(BASE, "materials")



interactions = pd.read_csv(os.path.join(DATA, 'interactions.csv'))

users = pd.read_csv(os.path.join(DATA, 'users.csv'))

editions = pd.read_csv(os.path.join(DATA, 'editions.csv'))

book_genres = pd.read_csv(os.path.join(DATA, 'book_genres.csv'))

genres = pd.read_csv(os.path.join(DATA, 'genres.csv'))

targets = pd.read_csv(os.path.join(DATA, 'targets.csv'))

candidates = pd.read_csv(os.path.join(DATA, 'candidates.csv'))



interactions['event_ts'] = pd.to_datetime(interactions['event_ts'])



print("Data loaded.")





# --- 2. Feature Engineering ---

print("Starting feature engineering...")



# Combine genres for each book

e2b = editions.set_index('edition_id')['book_id'].to_dict()

bgm = book_genres.groupby('book_id')['genre_id'].apply(frozenset).to_dict()

_EMPTY = frozenset()

egm = {eid: bgm.get(bid, _EMPTY) for eid, bid in e2b.items()}

egs = {eid: len(gs) for eid, gs in egm.items()}





# User genre preferences

user_genre_prefs = defaultdict(lambda: defaultdict(float))

for user_id, group in interactions.groupby('user_id'):

    for _, row in group.iterrows():

        event_weight = 3 if row['event_type'] == 2 else 1

        item_genres = egm.get(row['edition_id'], _EMPTY)

        for genre_id in item_genres:

            user_genre_prefs[user_id][genre_id] += event_weight



# Normalize genre preferences

for user_id in user_genre_prefs:

    total_weight = sum(user_genre_prefs[user_id].values())

    if total_weight > 0:

        for genre_id in user_genre_prefs[user_id]:

            user_genre_prefs[user_id][genre_id] /= total_weight





# --- 3. Collaborative Filtering with TruncatedSVD ---

print("Training TruncatedSVD model...")



# Create user-item matrix

user_ids = sorted(interactions['user_id'].unique())

item_ids = sorted(interactions['edition_id'].unique())

user_map = {uid: i for i, uid in enumerate(user_ids)}

item_map = {iid: i for i, iid in enumerate(item_ids)}



weights = interactions['event_type'].map({1: 1, 2: 3}).values

rows = interactions['user_id'].map(user_map).values

cols = interactions['edition_id'].map(item_map).values

user_item_matrix = csr_matrix((weights, (rows, cols)), shape=(len(user_ids), len(item_ids)))



# Decompose with SVD

svd = TruncatedSVD(n_components=64, random_state=42)

user_embeddings = svd.fit_transform(user_item_matrix)

item_embeddings = svd.components_.T



user_emb_df = pd.DataFrame(user_embeddings, index=user_ids)

item_emb_df = pd.DataFrame(item_embeddings, index=item_ids)



print("TruncatedSVD training done.")





def create_features(df, interactions, users, editions, user_emb_df, item_emb_df, user_genre_prefs, egm):

    """Create features for the given DataFrame."""

    

    # Basic user features

    user_features = users.copy()

    user_agg = interactions.groupby('user_id').agg(

        u_read_count=('event_type', lambda x: (x == 2).sum()),

        u_wish_count=('event_type', lambda x: (x == 1).sum()),

        u_avg_rating=('rating', 'mean'),

        u_total_interactions=('user_id', 'count')

    ).reset_index()

    user_features = user_features.merge(user_agg, on='user_id', how='left')



    # Basic item features

    item_features = editions[['edition_id', 'publication_year', 'age_restriction', 'language_id']].copy()

    item_agg = interactions.groupby('edition_id').agg(

        i_read_count=('event_type', lambda x: (x == 2).sum()),

        i_wish_count=('event_type', lambda x: (x == 1).sum()),

        i_avg_rating=('rating', 'mean'),

        i_total_interactions=('edition_id', 'count')

    ).reset_index()

    item_features = item_features.merge(item_agg, on='edition_id', how='left')

    

    # Merging features

    df = df.merge(user_features, on='user_id', how='left')

    df = df.merge(item_features, on='edition_id', how='left')

    

    # Merge embeddings

    df = df.merge(user_emb_df, left_on='user_id', right_index=True, how='left')

    df = df.merge(item_emb_df, left_on='edition_id', right_index=True, how='left')

    

    # Genre interaction features

    genre_jaccard = []

    genre_pref_score = []

    

    for _, row in df.iterrows():

        user_id = row['user_id']

        edition_id = row['edition_id']

        

        user_genres = set(user_genre_prefs.get(user_id, {}).keys())

        item_genres = egm.get(edition_id, _EMPTY)

        

        # Jaccard similarity

        if not user_genres or not item_genres:

            jaccard = 0

        else:

            jaccard = len(user_genres.intersection(item_genres)) / len(user_genres.union(item_genres))

        genre_jaccard.append(jaccard)

        

        # Genre preference score

        score = 0

        if item_genres:

            for genre_id in item_genres:

                score += user_genre_prefs.get(user_id, {}).get(genre_id, 0)

            score /= len(item_genres)

        genre_pref_score.append(score)

        

    df['genre_jaccard'] = genre_jaccard

    df['genre_pref_score'] = genre_pref_score

    

    return df.fillna(0)



# --- 4. Prepare Training Data and Validation ---

print("Preparing training data and validation...")















# User-based split for validation







val_users = targets['user_id'].sample(frac=0.2, random_state=42)







train_users = targets[~targets['user_id'].isin(val_users)]















# Validation data: candidates for validation users







val_df = candidates[candidates['user_id'].isin(val_users)]







val_labels = interactions[interactions['user_id'].isin(val_users)]







val_labels['label'] = val_labels['event_type'].map({1: 1, 2: 3})







val_df = val_df.merge(val_labels[['user_id', 'edition_id', 'label']], on=['user_id', 'edition_id'], how='left').fillna(0)















# Training data: interactions for training users, with negative sampling







train_df_pos = interactions[interactions['user_id'].isin(train_users['user_id'])].copy()







train_df_pos['label'] = train_df_pos['event_type'].map({1: 1, 2: 3})















train_df_rows = []







all_items = interactions['edition_id'].unique()







for user_id in train_users['user_id']:







    interacted_items = train_df_pos[train_df_pos['user_id'] == user_id]['edition_id'].unique()







    num_negatives = len(interacted_items) * 5 # 5x negatives







    non_interacted_items = np.setdiff1d(all_items, interacted_items)







    







    if len(non_interacted_items) > 0:







        negative_samples = np.random.choice(







            non_interacted_items,







            size=min(num_negatives, len(non_interacted_items)),







            replace=False







        )







        for item_id in negative_samples:







            train_df_rows.append({'user_id': user_id, 'edition_id': item_id, 'label': 0})















train_df_neg = pd.DataFrame(train_df_rows)







train_df = pd.concat([train_df_pos[['user_id', 'edition_id', 'label']], train_df_neg])















# Create features







train_features = create_features(train_df, interactions, users, editions, user_emb_df, item_emb_df, user_genre_prefs, egm)







val_features = create_features(val_df, interactions, users, editions, user_emb_df, item_emb_df, user_genre_prefs, egm)















print("Training data and validation prepared.")















# --- 5. Train Gradient Boosting Regressor ---







print("Training HistGradientBoostingRegressor...")















X_train = train_features.drop(columns=['user_id', 'edition_id', 'label'])







y_train = train_features['label']















# We don't need groups for a simple regressor







# train_groups = train_features.groupby('user_id').size().to_list()















X_val = val_features.drop(columns=['user_id', 'edition_id', 'label'])







y_val = val_features['label']







# val_groups = val_features.groupby('user_id').size().to_list()



















gbr = HistGradientBoostingRegressor(



    max_iter=1000,



    learning_rate=0.05,



    max_depth=7,



    min_samples_leaf=20,



    l2_regularization=1.0,



    max_bins=255,



    random_state=42,



    early_stopping=True,



    validation_fraction=0.1,



    n_iter_no_change=50,



)







gbr.fit(X_train, y_train)







print("HistGradientBoostingRegressor training done.")







# --- 6. Generate Submission ---



print("Generating submission...")







# Create features for all candidates



cand_features = create_features(candidates, interactions, users, editions, user_emb_df, item_emb_df, user_genre_prefs, egm)



cand_features['score'] = gbr.predict(cand_features.drop(columns=['user_id', 'edition_id']))







def mmr_rerank(df, egm, top_k=20, lambda_val=0.8):



    results = []



    for user_id, group in df.groupby('user_id'):



        ranked_items = group.sort_values('score', ascending=False).to_dict('records')



        



        selected_items = []



        if not ranked_items:



            continue



            



        # First item is the one with the highest score



        selected_items.append(ranked_items.pop(0))



        



        while len(selected_items) < top_k and ranked_items:



            best_item = None



            best_score = -1e9



            



            for item in ranked_items:



                relevance_score = item['score']



                diversity_score = 0



                



                # Calculate diversity against already selected items



                for sel_item in selected_items:



                    item_genres = egm.get(item['edition_id'], _EMPTY)



                    sel_item_genres = egm.get(sel_item['edition_id'], _EMPTY)



                    



                    if not item_genres or not sel_item_genres:



                        jaccard = 0



                    else:



                        jaccard = len(item_genres.intersection(sel_item_genres)) / len(item_genres.union(sel_item_genres))



                    diversity_score += (1 - jaccard)



                



                if selected_items:



                    diversity_score /= len(selected_items)



                



                # MMR score



                mmr_score = lambda_val * relevance_score - (1 - lambda_val) * diversity_score



                



                if mmr_score > best_score:



                    best_score = mmr_score



                    best_item = item



            if best_item is not None:



                selected_items.append(best_item)



                ranked_items.remove(best_item)



            else: # Should not happen if ranked_items is not empty



                break











        for rank, item in enumerate(selected_items, 1):



            results.append({'user_id': user_id, 'edition_id': item['edition_id'], 'rank': rank})



            



    return pd.DataFrame(results)







submission_df = mmr_rerank(cand_features, egm, lambda_val=0.8)







OUT_PATH = os.path.join(BASE, "submission_4.csv")



submission_df.to_csv(OUT_PATH, index=False)







print(f"Submission file created at {OUT_PATH}")



print("Done.")












