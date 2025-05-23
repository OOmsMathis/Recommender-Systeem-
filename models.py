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
    def __init__(self, random_state=1):
        super().__init__(n_factors=100, random_state=random_state)


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
    def __init__(self, features_methods, regressor_method): # Changé en pluriel
        AlgoBase.__init__(self)
        self.features_methods = features_methods  # Changé en pluriel
        self.regressor_method = regressor_method
        # Appel avec la variable d'instance (maintenant au pluriel)
        self.content_features = self.create_content_features(self.features_methods)

        

    def create_content_features(self, features_methods):
        """Content Analyzer"""
        df_items = load_items()
        df_features = pd.DataFrame(index=df_items.index)
        if features_methods is None:
            df_features = pd.DataFrame(index=df_items.index)
        if isinstance(features_methods, str):
            features_methods = [features_methods]
        
        for feature_method in features_methods:

            if feature_method == "title_length":
                df_title_length = df_items[C.LABEL_COL].apply(lambda x: len(x)).to_frame('title_length')
                df_title_length['title_length'] = df_title_length['title_length'].fillna(0).astype(int)
                mean_title_length = int(df_title_length['title_length'].replace(0, np.nan).mean())
                df_title_length.loc[df_title_length['title_length'] == 0, 'title_length'] = mean_title_length
                title_length_min = df_title_length['title_length'].min()
                title_length_max = df_title_length['title_length'].max()
                df_title_length['title_length'] = (df_title_length['title_length'] - title_length_min) / (title_length_max - title_length_min)
                df_features = pd.concat([df_features, df_title_length], axis=1)

            elif feature_method == "Year_of_release":
                year = df_items[C.LABEL_COL].str.extract(r'\((\d{4})\)')[0].astype(float)
                df_year = year.to_frame(name='year_of_release')
                mean_year = df_year.replace(0, np.nan).mean().iloc[0]
                df_year['year_of_release'] = df_year['year_of_release'].fillna(mean_year).astype(int)
                year_min = df_year['year_of_release'].min()
                year_max = df_year['year_of_release'].max()
                df_year['year_of_release'] = (df_year['year_of_release'] - year_min) / (year_max - year_min)
                df_features = pd.concat([df_features, df_year], axis=1)

            elif feature_method == "average_ratings":
                average_rating = df_ratings.groupby('movieId')[C.RATING_COL].mean().rename('average_rating').to_frame()
                global_avg = df_ratings[C.RATING_COL].mean()
                average_rating['average_rating'] = average_rating['average_rating'].fillna(global_avg)
                avg_rating_min = average_rating['average_rating'].min()
                avg_rating_max = average_rating['average_rating'].max()
                average_rating['average_rating'] = (average_rating['average_rating'] - avg_rating_min) / (avg_rating_max - avg_rating_min)
                df_features = df_features.join(average_rating, how='left')

            elif feature_method == "count_ratings":
                rating_count = df_ratings.groupby('movieId')[C.RATING_COL].size().rename('rating_count').to_frame()
                rating_count['rating_count'] = rating_count['rating_count'].fillna(0).astype(int)
                mean_rating_count = int(rating_count['rating_count'].replace(0, np.nan).mean())
                rating_count.loc[rating_count['rating_count'] == 0, 'rating_count'] = mean_rating_count
                rating_count_min = rating_count['rating_count'].min()
                rating_count_max = rating_count['rating_count'].max()
                rating_count['rating_count'] = (rating_count['rating_count'] - rating_count_min) / (rating_count_max - rating_count_min)
                df_features = df_features.join(rating_count, how='left')

            elif feature_method == "Genre_binary":
                df_genre_list = df_items[C.GENRES_COL].str.split('|').explode().to_frame('genre_list')
                df_dummies = pd.get_dummies(df_genre_list['genre_list'])
                df_genres = df_dummies.groupby(df_genre_list.index).sum()
                df_genres = df_genres.reindex(df_items.index).fillna(0).astype(int)
                df_features = pd.concat([df_features, df_genres], axis=1)

            elif feature_method == "Genre_tfidf":
                df_items['genre_string'] = df_items[C.GENRES_COL].fillna('').str.replace('|', ' ')
                tfidf = TfidfVectorizer()
                tfidf_matrix = tfidf.fit_transform(df_items['genre_string'])
                tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=df_items.index, columns=tfidf.get_feature_names_out())
                df_features = pd.concat([df_features, tfidf_df], axis=1)

            elif feature_method == "Tags":
                tags_path = str(C.CONTENT_PATH / "tags.csv")
                df_tags = pd.read_csv(tags_path)
                df_tags = df_tags.dropna(subset=['tag'])
                df_tags['tag'] = df_tags['tag'].astype(str)
                df_tags_grouped = df_tags.groupby('movieId')['tag'].agg(' '.join).to_frame('tags')
                tfidf = TfidfVectorizer()
                tfidf_matrix = tfidf.fit_transform(df_tags_grouped['tags'])
                tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=df_tags_grouped.index, columns=tfidf.get_feature_names_out())
                df_features = pd.concat([df_features, tfidf_df], axis=1)

            elif feature_method == "tmdb_vote_average":
                tmdb_path = str(C.CONTENT_PATH / "tmdb_full_features.csv")
                df_tmdb = pd.read_csv(tmdb_path)
                df_tmdb = df_tmdb[['movieId', 'vote_average']].drop_duplicates('movieId')
                df_tmdb = df_tmdb.set_index('movieId')
                mean_vote = df_tmdb['vote_average'].mean()
                df_tmdb['vote_average'] = df_tmdb['vote_average'].fillna(mean_vote)
                min_vote = df_tmdb['vote_average'].min()
                max_vote = df_tmdb['vote_average'].max()
                df_tmdb['vote_average'] = (df_tmdb['vote_average'] - min_vote) / (max_vote - min_vote)
                df_features = df_features.join(df_tmdb, how='left')
                    
            elif feature_method == "title_tfidf":
                # Combine titles into a single string per item
                df_items['title_string'] = df_items[C.LABEL_COL].fillna('')
                tfidf = TfidfVectorizer()
                tfidf_matrix = tfidf.fit_transform(df_items['title_string'])
                tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=df_items.index, columns=tfidf.get_feature_names_out())
                nltk.download('stopwords')
                nltk.download('wordnet')
                nltk.download('omw-1.4')
                lemmatizer = WordNetLemmatizer()
                stop_words = set(stopwords.words('english'))
                # Preprocess titles: remove stopwords and apply lemmatization
                df_items['title_string'] = df_items[C.LABEL_COL].fillna('').apply(lambda x: ' '.join(
                        lemmatizer.lemmatize(word) for word in x.split() if word.lower() not in stop_words
                    )
                )
                tfidf = TfidfVectorizer()
                tfidf_matrix = tfidf.fit_transform(df_items['title_string'])
                tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=df_items.index, columns=tfidf.get_feature_names_out())
                df_features = pd.concat([df_features, tfidf_df], axis=1)
            

            #attention, genomes_tags n'est pas ici car on a plus le fichier mais encore dispo dans content_based.ipynb
            else:
                raise NotImplementedError(f'Feature method {feature_method} not yet implemented')
        return df_features
    

    def fit(self, trainset):
        """Profile Learner"""
        self.content_features = self.create_content_features(self.features_methods)
        AlgoBase.fit(self, trainset)
        self.user_profile = {u: None for u in trainset.all_users()}
        for u in self.user_profile:
            user_items = trainset.ur[u]
            if len(user_items) > 0:
                # Sépare les item_ids internes et les notes
                user_ratings = self.trainset.ur[u]
                df_user = pd.DataFrame(user_ratings, columns=['inner_item_id', 'user_ratings'])
                # Conversion des item_id internes (Surprise) en item_id "raw" (MovieLens)
                df_user["item_id"] = df_user["inner_item_id"].map(self.trainset.to_raw_iid)
                # Fusion avec les features de contenu (sur l'index = item_id raw)
                df_user = df_user.merge(self.content_features, how='left', left_on='item_id', right_index=True)
                # Préparation des features et des cibles pour l'entraînement
                feature_names = list(self.content_features.columns)
                X = df_user[feature_names].values
                y = df_user['user_ratings'].values
                # Gère les NaNs dans les features
                X = np.nan_to_num(X)

     
                if self.regressor_method == 'linear': # Use linear regression
                    model = LinearRegression(fit_intercept=True)
                elif self.regressor_method == 'lasso':
                    model = Lasso(alpha=0.1)
                elif self.regressor_method == 'random_forest':
                    model = RandomForestRegressor(n_estimators=10, max_depth=10, random_state=42)
                elif self.regressor_method== 'neural_network':
                    model = MLPRegressor(hidden_layer_sizes=(60, 60), max_iter=2500, learning_rate_init=0.01, alpha=0.0001, random_state=42)
                elif self.regressor_method == 'decision_tree':
                    model = DecisionTreeRegressor(max_depth=10, random_state=42)
                elif self.regressor_method == 'ridge':
                    model = Ridge(alpha=1.0)
                elif self.regressor_method == 'gradient_boosting':
                    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
                elif  self.regressor_method == 'knn':
                    model = KNeighborsRegressor(n_neighbors=5)
                elif self.regressor_method == 'elastic_net':
                    model = ElasticNet(alpha=0.1, l1_ratio=0.5)

                else:
                    self.user_profile[u] = None
                    
                model.fit(X, y)
                self.user_profile[u] = model

            else:
             self.user_profile[u] = None
             
        
    def estimate(self, u, i):
        """Scoring component used for item filtering"""
        # First, handle cases for unknown users and items
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')

        if self.user_profile[u] is None:
            return self.trainset.global_mean

        raw_item_id = self.trainset.to_raw_iid(i)
        if raw_item_id in self.content_features.index:
            item_features = self.content_features.loc[raw_item_id].values.reshape(1, -1)
        else:
            return self.trainset.global_mean
    
        if self.regressor_method == 'linear':
            score = self.user_profile[u].predict(item_features)[0]
        elif self.regressor_method in [
        'linear',
        'lasso',
        'random_forest',
        'neural_network',
        'decision_tree',
        'ridge',
        'gradient_boosting',
        'knn',
        'elastic_net' ]:
          score = self.user_profile[u].predict(item_features)[0]

        else:
            score=None
            

        return score
