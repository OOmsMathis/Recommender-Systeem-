# models.py

from collections import defaultdict
import numpy as np
import pandas as pd
import nltk
from surprise import AlgoBase, KNNWithMeans, SVD as SurpriseSVD
from surprise.prediction_algorithms.predictions import PredictionImpossible
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from surprise import SVDpp
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os 
import requests
import zipfile
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

import constants as C_module
C = C_module.Constant()
from loaders import load_ratings, load_items

def download_and_extract_zip(url, extract_to="mlsmm2156"):
    zip_path = "data_temp.zip"

    # Télécharger le ZIP
    print("📥 Téléchargement du fichier ZIP...")
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Échec du téléchargement: {response.status_code}")

    with open(zip_path, "wb") as f:
        f.write(response.content)

    # Décompresser le ZIP
    print("📦 Décompression...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    # Supprimer le fichier ZIP après extraction
    os.remove(zip_path)
    print(f"✅ Données extraites dans: {extract_to}")

# Exécution prioritaire au lancement du script
if not os.path.exists("mlsmm2156/data/small/content"):
    download_and_extract_zip(
        r"https://my.microsoftpersonalcontent.com/personal/3814d4299f55d577/_layouts/15/download.aspx?UniqueId=b41e2d33-7aaa-4722-80aa-dc843ab0fe0f&Translate=false&tempauth=v1e.eyJzaXRlaWQiOiJmY2ZmMmE2MC1lMjQ4LTQ5ZjAtOWU3MS00ZmJjNzg2NDY3M2EiLCJhcHBpZCI6IjAwMDAwMDAwLTAwMDAtMDAwMC0wMDAwLTAwMDA0ODE3MTBhNCIsImF1ZCI6IjAwMDAwMDAzLTAwMDAtMGZmMS1jZTAwLTAwMDAwMDAwMDAwMC9teS5taWNyb3NvZnRwZXJzb25hbGNvbnRlbnQuY29tQDkxODgwNDBkLTZjNjctNGM1Yi1iMTEyLTM2YTMwNGI2NmRhZCIsImV4cCI6IjE3NDkxMzQzMDUifQ.19U4iYgenk51ZnK7D7QT2oikCNQBwraxmXpFBJ-uMwQT_iAImfcyCjGNcWKrFx_eMTNWYVmN67IdidqCaZQcVY-K-7RPVshv3LxZO_T1EDiMM4KnJqidpAOvF-DTQTgyl4kMB3FA2WUxXEjkP24n1K99E15OnqC35FpjAbL6zB5_7dXZ1Lp5RQ09Yrb7tnlLdkzbz4UDYp8HT4TYcpMVdzoOyKN9ohx1kd9UqCcnanyexJMOREeF-W65lz8a4b-ZpEoVPMZdiQEpRfrFiirB8ZjrLfhQcs8WkFmW5mZV1s3WyJVjLdJgaKkqXRUIICvA-s5WyqvNLycM9bnHFjBml75cTpxDQSV09ULjP9EXGpgelN3YmMQ_f5_eUSUIF6mtQqBialbgMvwiV3217NFWPw.0G2HCTh9R-uhkr7pECc2eEAT66uQC2Xozs4t41Emj-o&ApiVersion=2.0&AVOverride=1"
    )


print("models.py: Chargement des données globales...")
try:
    df_items_global = load_items()
    df_ratings_global = load_ratings()
    print("models.py: Données globales (items, ratings) chargées.")
    if df_items_global.empty: print("models.py: ATTENTION - df_items_global est vide après chargement !")
    if df_ratings_global.empty: print("models.py: ATTENTION - df_ratings_global est vide après chargement !")
except Exception as e:
    print(f"models.py: ERREUR FATALE lors du chargement des données globales: {e}")
    _cols_items_cb = [getattr(C, 'ITEM_ID_COL', 'movieId'), getattr(C, 'LABEL_COL', 'title'), getattr(C, 'RELEASE_YEAR_COL', 'release_year'), getattr(C, 'GENRES_COL', 'genres'), getattr(C, 'VOTE_AVERAGE_COL', 'vote_average')]
    df_items_global = pd.DataFrame(columns=_cols_items_cb)
    _cols_ratings_cb = [getattr(C, 'ITEM_ID_COL', 'movieId'), getattr(C, 'RATING_COL', 'rating'), getattr(C, 'USER_ID_COL', 'userId')]
    df_ratings_global = pd.DataFrame(columns=_cols_ratings_cb)


def get_top_n(predictions, n=10):
    top_n_dict = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions: top_n_dict[uid].append((iid, est))
    for uid, user_ratings in top_n_dict.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n_dict[uid] = user_ratings[:n]
    return top_n_dict

class ContentBased(AlgoBase):
    def __init__(self, features_methods, regressor_method, alpha=1.0):
        AlgoBase.__init__(self)
        self.features_methods = features_methods
        self.regressor_method = regressor_method
        self.ridge_alpha = getattr(self, 'alpha', alpha)

        # Les features de contenu sont créées ici sans précalcul
        self.content_features = self.create_content_features(self.features_methods)

    def create_content_features(self, features_methods):
        """Content Analyzer: Crée les features de contenu pour les items sans précalcul."""
        df_items = load_items()
        df_ratings = load_ratings()
        df_features = pd.DataFrame(index=df_items.index)

        if features_methods is None:
            return pd.DataFrame(index=df_items.index)
        if isinstance(features_methods, str):
            features_methods = [features_methods]

        df_tmdb = None
        if any(f.startswith('tmdb_') for f in features_methods):
            tmdb_path = C.CONTENT_PATH / "tmdb_full_features.csv"
            df_tmdb = pd.read_csv(tmdb_path).drop_duplicates('movieId').set_index('movieId')
            if 'tmdb_profit' in features_methods:
                df_tmdb['profit'] = df_tmdb['revenue'] - df_tmdb['budget']

        for feature_method in features_methods:
            if feature_method == "title_length":
                df_title_length = df_items[C.LABEL_COL].apply(lambda x: len(x)).to_frame('title_length')
                df_title_length['title_length'] = df_title_length['title_length'].fillna(0).astype(int)
                df_features = pd.concat([df_features, df_title_length], axis=1)

            elif feature_method == "Year_of_release":
                year = df_items[C.LABEL_COL].str.extract(r'\((\d{4})\)')[0].astype(float)
                df_year = year.to_frame(name='year_of_release')
                df_features = pd.concat([df_features, df_year], axis=1)

            elif feature_method == "average_ratings":
                average_rating = df_ratings.groupby('movieId')[C.RATING_COL].mean().rename('average_rating').to_frame()
                df_features = df_features.join(average_rating, how='left')

            elif feature_method == "count_ratings":
                rating_count = df_ratings.groupby('movieId')[C.RATING_COL].size().rename('rating_count').to_frame()
                df_features = df_features.join(rating_count, how='left')

            elif feature_method == "Genre_binary":
                df_genre_list = df_items[C.GENRES_COL].str.split('|').explode().to_frame('genre_list')
                df_dummies = pd.get_dummies(df_genre_list['genre_list'])
                df_genres = df_dummies.groupby(df_genre_list.index).sum()
                df_genres = df_genres.reindex(df_items.index).fillna(0).astype(int)
                df_features = pd.concat([df_features, df_genres], axis=1)

            elif feature_method == "Genre_tfidf":
                tfidf_vectorizer = TfidfVectorizer()
                df_items['genre_string'] = df_items[C.GENRES_COL].fillna('').str.replace('|', ' ', regex=False)
                tfidf_matrix = tfidf_vectorizer.fit_transform(df_items['genre_string'])
                tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=df_items.index, columns=tfidf_vectorizer.get_feature_names_out())
                df_features = pd.concat([df_features, tfidf_df], axis=1)

            elif feature_method == "Tags_tfidf":
                tags_path = C.CONTENT_PATH / "tags.csv"
                df_tags_local = pd.read_csv(tags_path)
                df_tags_local = df_tags_local.dropna(subset=['tag'])
                df_tags_local['tag'] = df_tags_local['tag'].astype(str)
                df_tags_grouped = df_tags_local.groupby('movieId')['tag'].agg(' '.join).to_frame('tags')
                tfidf_vectorizer = TfidfVectorizer()
                tfidf_matrix = tfidf_vectorizer.fit_transform(df_tags_grouped['tags'])
                tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=df_tags_grouped.index, columns=tfidf_vectorizer.get_feature_names_out())
                tfidf_df = tfidf_df.reindex(df_items.index, fill_value=0)
                df_features = pd.concat([df_features, tfidf_df], axis=1)

            elif feature_method == "tmdb_vote_average":
                df_tmdb_col = df_tmdb[['vote_average']].copy()
                df_features = df_features.join(df_tmdb_col, how='left')

            elif feature_method == "title_tfidf":
                lemmatizer = WordNetLemmatizer()
                stop_words = set(stopwords.words('english'))
                df_items['title_string_processed'] = df_items[C.LABEL_COL].fillna('').apply(lambda x: ' '.join(
                                lemmatizer.lemmatize(word) for word in x.split() if word.lower() not in stop_words
                            ))
                tfidf_vectorizer = TfidfVectorizer()
                tfidf_matrix = tfidf_vectorizer.fit_transform(df_items['title_string_processed'])
                tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=df_items.index, columns=tfidf_vectorizer.get_feature_names_out())
                df_features = pd.concat([df_features, tfidf_df], axis=1)

            elif feature_method == "genome_tags":
                genome_tags_path = C.CONTENT_PATH / "genome-tags.csv"
                genome_scores_path = C.CONTENT_PATH / "genome-scores.csv"
                df_scores = pd.read_csv(genome_scores_path)
                df_tags_g = pd.read_csv(genome_tags_path)
                df_merged_g = df_scores.merge(df_tags_g, on='tagId')
                df_genome_features = df_merged_g.pivot_table(index='movieId', columns='tag', values='relevance', fill_value=0)
                df_genome_features = df_genome_features.reindex(df_items.index, fill_value=0)
                # Genome features (relevance scores, not scaled)
                df_features = pd.concat([df_features, df_genome_features], axis=1)

            elif feature_method == "tfidf_relevance":
                genome_tags_path = C.CONTENT_PATH / "genome-tags.csv"
                genome_scores_path = C.CONTENT_PATH / "genome-scores.csv"
                df_tags_g = pd.read_csv(genome_tags_path)
                df_scores_g = pd.read_csv(genome_scores_path)
                df_merged_g = df_scores_g.merge(df_tags_g, on='tagId')
                df_merged_g['tag'] = df_merged_g['tag'].astype(str)
                df_texts = df_merged_g.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).to_frame('tags')
                tfidf_vectorizer = TfidfVectorizer()
                tfidf_matrix = tfidf_vectorizer.fit_transform(df_texts['tags'])
                tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=df_texts.index, columns=tfidf_vectorizer.get_feature_names_out())
                tfidf_df = tfidf_df.reindex(df_items.index, fill_value=0)
                # StandardScaler sur les valeurs TF-IDF
                scaler = StandardScaler()
                tfidf_scaled = scaler.fit_transform(tfidf_df)
                tfidf_scaled_df = pd.DataFrame(tfidf_scaled, index=tfidf_df.index, columns=tfidf_df.columns)
                df_features = pd.concat([df_features, tfidf_scaled_df], axis=1)

            elif feature_method == "tmdb_popularity":
                df_tmdb_col = df_tmdb[['popularity']].copy()
                scaler = StandardScaler()
                df_tmdb_col['popularity'] = df_tmdb_col['popularity'].fillna(0)
                df_tmdb_col['popularity_scaled'] = scaler.fit_transform(df_tmdb_col[['popularity']])
                df_features = df_features.join(df_tmdb_col[['popularity_scaled']], how='left')

            elif feature_method == "tmdb_budget":
                df_tmdb_col = df_tmdb[['budget']].copy()
                scaler = StandardScaler()
                # Remplir les valeurs manquantes par 0 avant le scaling
                df_tmdb_col['budget'] = df_tmdb_col['budget'].fillna(0)
                df_tmdb_col['budget_scaled'] = scaler.fit_transform(df_tmdb_col[['budget']])
                df_features = df_features.join(df_tmdb_col[['budget_scaled']], how='left')

            elif feature_method == "tmdb_revenue":
                df_tmdb_col = df_tmdb[['revenue']].copy()
                scaler = StandardScaler()
                # Remplir les valeurs manquantes par 0 avant le scaling
                df_tmdb_col['revenue'] = df_tmdb_col['revenue'].fillna(0)
                df_tmdb_col['revenue_scaled'] = scaler.fit_transform(df_tmdb_col[['revenue']])
                df_features = df_features.join(df_tmdb_col[['revenue_scaled']], how='left')

            elif feature_method == "tmdb_profit":
                df_tmdb_col = df_tmdb[['profit']].copy()
                scaler = StandardScaler()
                # Remplir les valeurs manquantes par 0 avant le scaling
                df_tmdb_col['profit'] = df_tmdb_col['profit'].fillna(0)
                df_tmdb_col['profit_scaled'] = scaler.fit_transform(df_tmdb_col[['profit']])
                df_features = df_features.join(df_tmdb_col[['profit_scaled']], how='left')

            elif feature_method == "tmdb_runtime":
                df_tmdb_col = df_tmdb[['runtime']].copy()
                scaler = StandardScaler()
                # Remplir les valeurs manquantes par 0 avant le scaling
                df_tmdb_col['runtime'] = df_tmdb_col['runtime'].fillna(0)
                df_tmdb_col['runtime_scaled'] = scaler.fit_transform(df_tmdb_col[['runtime']])
                df_features = df_features.join(df_tmdb_col[['runtime_scaled']], how='left')

            elif feature_method == "tmdb_vote_count":
                df_tmdb_col = df_tmdb[['vote_count']].copy()
                scaler = StandardScaler()
                # Remplir les valeurs manquantes par 0 avant le scaling
                df_tmdb_col['vote_count'] = df_tmdb_col['vote_count'].fillna(0)
                df_tmdb_col['vote_count_scaled'] = scaler.fit_transform(df_tmdb_col[['vote_count']])
                df_features = df_features.join(df_tmdb_col[['vote_count_scaled']], how='left')

            elif feature_method == "tmdb_cast":
                df_tmdb_col = df_tmdb[['cast']].copy()
                df_tmdb_col['cast'] = df_tmdb_col['cast'].fillna('')
                tfidf_vectorizer = TfidfVectorizer()
                tfidf_matrix = tfidf_vectorizer.fit_transform(df_tmdb_col['cast'])
                tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=df_tmdb_col.index, columns=tfidf_vectorizer.get_feature_names_out())
                tfidf_df = tfidf_df.reindex(df_items.index, fill_value=0)
                df_features = pd.concat([df_features, tfidf_df], axis=1)

            elif feature_method == "tmdb_director":
                df_tmdb_col = df_tmdb[['director']].copy()
                df_tmdb_col['director'] = df_tmdb_col['director'].fillna('')
                tfidf_vectorizer = TfidfVectorizer()
                tfidf_matrix = tfidf_vectorizer.fit_transform(df_tmdb_col['director'])
                tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=df_tmdb_col.index, columns=tfidf_vectorizer.get_feature_names_out())
                tfidf_df = tfidf_df.reindex(df_items.index, fill_value=0)
                df_features = pd.concat([df_features, tfidf_df], axis=1)

            elif feature_method == "tmdb_original_language":
                df_tmdb_col = df_tmdb[['original_language']].copy()
                df_tmdb_col['original_language'] = df_tmdb_col['original_language'].fillna('unknown')
                df_lang_dummies = pd.get_dummies(df_tmdb_col['original_language'], prefix='lang')
                df_lang_dummies = df_lang_dummies.reindex(df_items.index, fill_value=0)
                df_features = pd.concat([df_features, df_lang_dummies], axis=1)

            else:
                raise NotImplementedError(f'Feature method {feature_method} not yet implemented or misconfigured.')

        df_features = df_features.fillna(0)
        return df_features

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        self.user_profile = {u: None for u in trainset.all_users()}
        for u in self.user_profile:
            user_items = trainset.ur[u]
            if len(user_items) > 0:
                user_ratings = self.trainset.ur[u]
                df_user = pd.DataFrame(user_ratings, columns=['inner_item_id', 'user_ratings'])
                df_user["item_id"] = df_user["inner_item_id"].map(self.trainset.to_raw_iid)
                df_user = df_user.merge(self.content_features, how='left', left_on='item_id', right_index=True)
                feature_names = list(self.content_features.columns)
                X = df_user[feature_names].values
                y = df_user['user_ratings'].values
                X = np.nan_to_num(X)

                if self.regressor_method == 'linear':
                    model = LinearRegression(fit_intercept=True)
                elif self.regressor_method == 'lasso':
                    model = Lasso(alpha=0.1)
                elif self.regressor_method == 'random_forest':
                    model = RandomForestRegressor(n_estimators=10, max_depth=10, random_state=42)
                elif self.regressor_method == 'neural_network':
                    model = MLPRegressor(hidden_layer_sizes=(60, 60), max_iter=2500, learning_rate_init=0.01, alpha=0.0001, random_state=42)
                elif self.regressor_method == 'decision_tree':
                    model = DecisionTreeRegressor(
                        max_depth=getattr(self, 'decision_tree_max_depth', 10),
                        random_state=getattr(self, 'decision_tree_random_state', 42)
                    )
                elif self.regressor_method == 'ridge':
                    model = Ridge(alpha=getattr(self, 'ridge_alpha', 1.0))
                elif self.regressor_method == 'gradient_boosting':
                    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
                elif self.regressor_method == 'knn':
                    model = KNeighborsRegressor(n_neighbors=5)
                elif self.regressor_method == 'elastic_net':
                    model = ElasticNet(alpha=0.1, l1_ratio=0.5)
                else:
                    self.user_profile[u] = None
                    continue

                model.fit(X, y)
                self.user_profile[u] = model
            else:
                self.user_profile[u] = None

        return self

    def estimate(self, u, i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unknown.')

        if self.user_profile[u] is None:
            return self.trainset.global_mean

        raw_item_id = self.trainset.to_raw_iid(i)
        if raw_item_id in self.content_features.index:
            item_features = self.content_features.loc[raw_item_id].values.reshape(1, -1)
            item_features = np.nan_to_num(item_features)
        else:
            return self.trainset.global_mean

        score = self.user_profile[u].predict(item_features)[0]
        min_rating, max_rating = self.trainset.rating_scale
        score = max(min_rating, min(max_rating, score))
        return score

class UserBased(AlgoBase): # Inchangé
    def __init__(self, k=40, min_k=1, sim_options={'name': 'msd', 'user_based': True}, verbose=False):
        AlgoBase.__init__(self)
        self.k, self.min_k, self.sim_options, self.verbose = k, min_k, sim_options, verbose
        self.knn = KNNWithMeans(k=self.k, min_k=self.min_k, sim_options=self.sim_options, verbose=self.verbose)
    def fit(self, trainset): 
        AlgoBase.fit(self, trainset); 
        self.knn.fit(trainset); 
        return self
    def estimate(self, u, i): 
        return self.knn.estimate(u, i)

class SVDAlgo(AlgoBase): # Inchangé
    def __init__(self, n_factors=100, n_epochs=20, biased=True, lr_all=0.005, reg_all=0.02, random_state=None, verbose=False):
        AlgoBase.__init__(self)
        self.svd_model = SurpriseSVD(n_factors=n_factors, n_epochs=n_epochs, biased=biased, lr_all=lr_all, reg_all=reg_all, random_state=random_state, verbose=verbose)
    def fit(self, trainset): AlgoBase.fit(self, trainset); self.svd_model.fit(trainset); return self
    def estimate(self, u, i): return self.svd_model.estimate(u, i)

class ModelSVDpp(SVDpp):
    def __init__(self, n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=1, **kwargs):
        super().__init__(n_factors=n_factors,n_epochs=n_epochs,lr_all=lr_all,reg_all=reg_all,random_state=random_state,**kwargs)