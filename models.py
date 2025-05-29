# standard library imports
from collections import defaultdict

# third parties imports
import numpy as np
import random as rd
import pandas as pd
import nltk
import heapq

from surprise import AlgoBase
from surprise import KNNWithMeans
from surprise.prediction_algorithms.predictions import PredictionImpossible
from surprise import SVD
from sklearn.linear_model import LinearRegression
from surprise import PredictionImpossible
from loaders import load_ratings
from loaders import load_items
import constants as C
from constants import Constant as C
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import StandardScaler

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

df_items = load_items()
df_ratings = load_ratings()

def get_top_n(predictions, n):
    """Return the top-N recommendation for each user from a set of predictions.
    Source: inspired by https://github.com/NicolasHug/Surprise/blob/master/examples/top_n_recommendations.py
    and modified by cvandekerckh for random tie breaking

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.
    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    rd.seed(0)

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        rd.shuffle(user_ratings)
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


# First algorithm
class ModelBaseline1(AlgoBase):
    def __init__(self):
        AlgoBase.__init__(self)

    def estimate(self, u, i):
        return 2


# Second algorithm
class ModelBaseline2(AlgoBase):
    def __init__(self):
        AlgoBase.__init__(self)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        rd.seed(0)

    def estimate(self, u, i):
        return rd.uniform(self.trainset.rating_scale[0], self.trainset.rating_scale[1])


# Third algorithm
class ModelBaseline3(AlgoBase):
    def __init__(self):
        AlgoBase.__init__(self)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        self.the_mean = np.mean([r for (_, _, r) in self.trainset.all_ratings()])

        return self

    def estimate(self, u, i):
        return self.the_mean
    
# Fourth algorithm
class ModelBaseline4(SVD):
    def __init__(
        self,
        n_factors=100,
        n_epochs=20,
        lr_all=0.005,
        reg_all=0.02,
        biased=True,
        random_state=1,
        **kwargs
    ):
        super().__init__(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all,
            biased=biased,
            random_state=random_state,
            **kwargs
        )


class UserBased(AlgoBase):
    def __init__(self, k=3, min_k=1, sim_options={}, **kwargs):
        AlgoBase.__init__(self, sim_options=sim_options, **kwargs)
        self.k = k
        self.min_k = min_k

        
    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        
        self.compute_rating_matrix()
        
        self.compute_similarity_matrix()
        
        self.mean_ratings = []
        for u in range(self.trainset.n_users):
            user_ratings = []
            for (_, rating) in self.trainset.ur[u]: 
                user_ratings.append(rating)
            if user_ratings:
                mean_rating = np.mean(user_ratings)
            else:
                mean_rating = float('nan')  
            self.mean_ratings.append(mean_rating)

    
    def estimate(self, u, i):
            if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
                raise PredictionImpossible('User and/or item is unknown.') 
            
            estimate = self.mean_ratings[u]

            
            neighbors = []
            for (v, rating) in self.trainset.ir[i]:  
                if v == u:
                    continue  

                sim_uv = self.sim[u, v] 

                if sim_uv > 0 and not np.isnan(self.ratings_matrix[v, i]): 
                    mean_v = self.mean_ratings[v]  
                    neighbors.append((sim_uv, rating - mean_v))

            
            top_k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda x: x[0])

            
            actual_k = 0
            weighted_sum = 0.0
            sum_sim = 0.0

            for sim, rating_diff in top_k_neighbors:
                if actual_k == self.k:
                    break
                weighted_sum += sim * rating_diff
                sum_sim += sim
                actual_k += 1

            # Check
            if actual_k >= self.min_k and sum_sim > 0:
                estimate += weighted_sum / sum_sim

            return estimate


                            
    def compute_rating_matrix(self):
        # -- implement here the compute_rating_matrix function --
        self.ratings_matrix = np.empty((self.trainset.n_users, self.trainset.n_items))
        self.ratings_matrix[:] = np.nan
        for u in range(self.trainset.n_users): 
            for i, rating in self.trainset.ur[u]:
                self.ratings_matrix[u, i] = rating

    
    def compute_similarity_matrix(self):
        m = self.trainset.n_users
        ratings_matrix = self.ratings_matrix
        min_support = self.sim_options.get('min_support', 1)
        sim_name = self.sim_options.get("name", "msd") 

        # Similarity matrix
        self.sim = np.eye(m)

        for i in range(m):
            for j in range(i + 1, m):  
                row_i = ratings_matrix[i]
                row_j = ratings_matrix[j]

                if sim_name == "jaccard":
                    sim = self.jaccard_similarity(row_i, row_j)
                    support = np.sum(~np.isnan(row_i) & ~np.isnan(row_j))
                elif sim_name == "msd":
                    diff = row_i - row_j
                    support = np.sum(~np.isnan(diff))
                    if support >= min_support:
                        msd = np.nanmean((diff[~np.isnan(diff)]) ** 2)
                        sim = 1 / (1 + msd)
                    else:
                        sim = 0
                else:
                    
                    diff = row_i - row_j
                    support = np.sum(~np.isnan(diff))
                    if support >= min_support:
                        msd = np.nanmean((diff[~np.isnan(diff)]) ** 2)
                        sim = 1 / (1 + msd)
                    else:
                        sim = 0

                if support >= min_support:
                    self.sim[i, j] = sim
                    self.sim[j, i] = sim

    def jaccard_similarity(self, row_i, row_j):
        
        mask_i = ~np.isnan(row_i)
        mask_j = ~np.isnan(row_j)

        intersection = np.sum(mask_i & mask_j)
        union = np.sum(mask_i | mask_j)

        if union == 0:
            return 0.0
        return intersection / union



class ContentBased(AlgoBase):
    def __init__(self, features_methods, regressor_method):
        AlgoBase.__init__(self)
        self.features_methods = features_methods
        self.regressor_method = regressor_method

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

            elif feature_method == "Tags":
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
                    model = DecisionTreeRegressor(max_depth=10, random_state=42)
                elif self.regressor_method == 'ridge':
                    model = Ridge(alpha=0.15)
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

class CustomSurpriseAlgo(AlgoBase):
    """
    Algorithme flexible pour Surprise, permettant de choisir entre :
    - 'knn_with_means'
    - 'baseline'
    - 'zscore'
    - 'basic'
    """
    def __init__(self, mode='knn_with_means', k=40, min_k=1, sim_options=None, **kwargs):
        AlgoBase.__init__(self)
        self.mode = mode
        self.k = k
        self.min_k = min_k
        self.sim_options = sim_options if sim_options is not None else {}
        self.kwargs = kwargs

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        self.trainset = trainset

        if self.mode == 'knn_with_means':
            self.algo = KNNWithMeans(
                k=self.k,
                min_k=self.min_k,
                sim_options=self.sim_options,
                **self.kwargs
            )
            self.algo.fit(trainset)
        elif self.mode == 'baseline':
            self.global_mean = np.mean([r for (_, _, r) in self.trainset.all_ratings()])
            self.user_means = {}
            self.item_means = {}
            for u in range(self.trainset.n_users):
                ratings = [r for (_, r) in self.trainset.ur[u]]
                self.user_means[u] = np.mean(ratings) if ratings else self.global_mean
            for i in range(self.trainset.n_items):
                ratings = [r for (_, r) in self.trainset.ir[i]]
                self.item_means[i] = np.mean(ratings) if ratings else self.global_mean
        elif self.mode == 'zscore':
            self.user_means = {}
            self.user_stds = {}
            for u in range(self.trainset.n_users):
                ratings = [r for (_, r) in self.trainset.ur[u]]
                self.user_means[u] = np.mean(ratings) if ratings else 0
                self.user_stds[u] = np.std(ratings) if ratings else 1
        elif self.mode == 'basic':
            self.global_mean = np.mean([r for (_, _, r) in self.trainset.all_ratings()])
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        return self

    def estimate(self, u, i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unknown.')

        if self.mode == 'knn_with_means':
            return self.algo.estimate(u, i)
        elif self.mode == 'baseline':
            return (self.user_means[u] + self.item_means[i]) / 2
        elif self.mode == 'zscore':
            mean = self.user_means[u]
            std = self.user_stds[u]
            if std == 0:
                return mean
            return mean  # Peut être amélioré pour un vrai z-score
        elif self.mode == 'basic':
            return self.global_mean
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
