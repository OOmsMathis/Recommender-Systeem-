# standard library imports
from collections import defaultdict
###
# third parties imports
import numpy as np
import random as rd
import pandas as pd
from surprise import AlgoBase
from surprise import KNNWithMeans
from surprise import SVD
from sklearn.linear_model import LinearRegression
from surprise import PredictionImpossible
from loaders import load_ratings
from loaders import load_items
import constants as C
from constants import Constant as C

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


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


# Fourth Model
class ModelBaseline4(SVD):
    def __init__(self,random_state = 1):
        SVD.__init__(self, n_factors=100)

#content_based
class ContentBased(AlgoBase):
    def __init__(self, features_methods, regressor_method):
        AlgoBase.__init__(self)
        self.features_method = features_methods
        self.regressor_method = regressor_method
        self.content_features = self.create_content_features(features_methods)
        

    def create_content_features(self, features_methods):
        """Content Analyzer"""
        df_items = load_items()
        df_ratings = load_ratings()
        df_features = pd.DataFrame(index=df_items.index)
        if features_methods is None:
           df_features = pd.DataFrame(index=df_items.index)
        if isinstance(features_methods, str):
         features_methods = [features_methods]
        
        for feature_method in features_methods:
         if feature_method == "title_length": # a naive method that creates only 1 feature based on title length
            df_title_length = df_items[C.LABEL_COL].apply(lambda x: len(x)).to_frame('title_length')
            df_title_length['title_length'] = df_title_length['title_length'].fillna(0).astype(int)
            mean_title_length = int(df_title_length['title_length'].replace(0, np.nan).mean())
            df_title_length.loc[df_title_length['title_length'] == 0, 'title_length'] = mean_title_length
            # Normaliser la longueur des titre entre 0 et 1
            title_length_min = df_title_length['title_length'].min()
            title_length_max = df_title_length['title_length'].max()
            df_title_length['title_length'] = (df_title_length['title_length'] - title_length_min) / (title_length_max - title_length_min)
            df_features = pd.concat([df_features, df_title_length], axis=1)
         elif feature_method == "Year_of_release":
            year = df_items[C.LABEL_COL].str.extract(r'\((\d{4})\)')[0].astype(float)
            df_year = year.to_frame(name='year_of_release')
            mean_year = df_year.replace(0, np.nan).mean().iloc[0]
            df_year['year_of_release'] = df_year['year_of_release'].fillna(mean_year).astype(int)
             # Normaliser les dates de sortie 
            year_min = df_year['year_of_release'].min()
            year_max = df_year['year_of_release'].max()
            df_year['year_of_release'] = (df_year['year_of_release'] - year_min) / (year_max - year_min)
            df_features = pd.concat([df_features, df_year], axis=1)
         elif feature_method =="average_ratings":
            # moyenne des notes par films
            average_rating = df_ratings.groupby('movieId')[C.RATING_COL].mean().rename('average_rating').to_frame()
            global_avg = df_ratings['rating'].mean()
            average_rating['average_rating'] = average_rating['average_rating'].fillna(global_avg)
            # Normaliser la moyenne des notes par films
            avg_rating_min = average_rating['average_rating'].min()
            avg_rating_max = average_rating['average_rating'].max()
            average_rating['average_rating'] = (average_rating['average_rating'] - avg_rating_min) / (avg_rating_max - avg_rating_min)
            df_features = df_features.join(average_rating, how='left')
         elif feature_method =="count_ratings":
             # Count the number of ratings for each movie
            rating_count = df_ratings.groupby('movieId')[C.RATING_COL].size().rename('rating_count').to_frame()
            rating_count['rating_count'] = rating_count['rating_count'].fillna(0).astype(int)
            mean_rating_count = int(rating_count['rating_count'].replace(0, np.nan).mean())
            rating_count.loc[rating_count['rating_count'] == 0, 'rating_count'] = mean_rating_count
                # Normalize the rating count
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
                df_features = df_features.fillna(0)
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
       
         elif feature_method == "genome_tags":
                tags_path = C.CONTENT_PATH / "genome-tags.csv"
                scores_path = C.CONTENT_PATH / "genome-scores.csv"
                df_scores = pd.read_csv(scores_path)
                df_tags = pd.read_csv(tags_path)
                 # Étape 2 : Merge pour récupérer les noms des tags
                df_merged = df_scores.merge(df_tags, on='tagId')
                # Étape 3 : Pivot → films × tags, valeurs = relevance
                df_features = df_merged.pivot_table(index='movieId', columns='tag', values='relevance', fill_value=0)
        
         elif feature_method == "tfidf_relevance":
                tags_path = C.CONTENT_PATH / "genome-tags.csv"
                scores_path = C.CONTENT_PATH / "genome-scores.csv"
                # Charger les données
                df_tags = pd.read_csv(tags_path)
                df_scores = pd.read_csv(scores_path)
                # Fusionner pour obtenir les noms des tags
                df_merged = df_scores.merge(df_tags, on='tagId')
                # Grouper les tags pertinents par film en texte
                df_merged['tag'] = df_merged['tag'].astype(str)
                df_texts = df_merged.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).to_frame('tags')
                # Appliquer TF-IDF
                tfidf = TfidfVectorizer()
                tfidf_matrix = tfidf.fit_transform(df_texts['tags'])
                # Créer le DataFrame final de features
                df_features = pd.DataFrame(tfidf_matrix.toarray(), index=df_texts.index, columns=tfidf.get_feature_names_out())
         elif feature_method == "visuals":
                visual_path = C.VISUAL/ "LLVisualFeatures13K_QuantileLog.csv"
                df_visual = pd.read_csv(visual_path).set_index("ML_Id")

                # Nettoyage
                df_visual = df_visual.fillna(0)

                # Normalisation colonne par colonne
                for col in df_visual.columns:
                    col_min = df_visual[col].min()
                    col_max = df_visual[col].max()
                    if col_max != col_min:
                        df_visual[col] = (df_visual[col] - col_min) / (col_max - col_min)

                # Fusion dans df_features
                df_features = df_features.join(df_visual, how='left')
                 # Très important : remplacer les NaN après fusion
                df_features = df_features.fillna(0)
         elif feature_method == "tmdb_popularity":
                tmdb_path = C.CONTENT_PATH / "tmdb_full_features.csv"
                df_tmdb = pd.read_csv(tmdb_path)
                df_tmdb = df_tmdb[['movieId', 'popularity']].drop_duplicates('movieId')
                df_tmdb = df_tmdb.set_index('movieId')
                mean_popularity = df_tmdb['popularity'].mean()
                df_tmdb['popularity'] = df_tmdb['popularity'].fillna(mean_popularity)
                min_popularity = df_tmdb['popularity'].min()
                max_popularity = df_tmdb['popularity'].max()
                df_tmdb['popularity'] = (df_tmdb['popularity'] - min_popularity) / (max_popularity - min_popularity)
                df_features = df_features.join(df_tmdb, how='left')
                df_features = df_features.fillna(0)

         elif feature_method == "tmdb_budget":
                tmdb_path = C.CONTENT_PATH / "tmdb_full_features.csv"
                df_tmdb = pd.read_csv(tmdb_path)
                df_tmdb = df_tmdb[['movieId', 'budget']].drop_duplicates('movieId')
                df_tmdb = df_tmdb.set_index('movieId')
                mean_budget = df_tmdb['budget'].mean()
                df_tmdb['budget'] = df_tmdb['budget'].fillna(mean_budget)
                min_budget = df_tmdb['budget'].min()
                max_budget = df_tmdb['budget'].max()
                df_tmdb['budget'] = (df_tmdb['budget'] - min_budget) / (max_budget - min_budget)
                df_features = df_features.join(df_tmdb, how='left')
                df_features = df_features.fillna(0)
            
         elif feature_method == "tmdb_revenue":
                tmdb_path = C.CONTENT_PATH / "tmdb_full_features.csv"
                df_tmdb = pd.read_csv(tmdb_path)
                df_tmdb = df_tmdb[['movieId', 'revenue']].drop_duplicates('movieId')
                df_tmdb = df_tmdb.set_index('movieId')
                mean_revenue = df_tmdb['revenue'].mean()
                df_tmdb['revenue'] = df_tmdb['revenue'].fillna(mean_revenue)
                min_revenue = df_tmdb['revenue'].min()
                max_revenue = df_tmdb['revenue'].max()
                df_tmdb['revenue'] = (df_tmdb['revenue'] - min_revenue) / (max_revenue - min_revenue)
                df_features = df_features.join(df_tmdb, how='left')
                df_features = df_features.fillna(0)
            
         elif feature_method == "tmdb_profit":
                tmdb_path = C.CONTENT_PATH / "tmdb_full_features.csv"
                df_tmdb = pd.read_csv(tmdb_path)
                df_tmdb['profit'] = df_tmdb['revenue'] - df_tmdb['budget']
                df_tmdb = df_tmdb[['movieId', 'profit']].drop_duplicates('movieId')
                df_tmdb = df_tmdb.set_index('movieId')
                mean_profit = df_tmdb['profit'].mean()
                df_tmdb['profit'] = df_tmdb['profit'].fillna(mean_profit)
                min_profit = df_tmdb['profit'].min()
                max_profit = df_tmdb['profit'].max()
                df_tmdb['profit'] = (df_tmdb['profit'] - min_profit) / (max_profit - min_profit)
                df_features = df_features.join(df_tmdb, how='left')
                df_features = df_features.fillna(0)
            
         elif feature_method == "tmdb_runtime":
                tmdb_path = C.CONTENT_PATH / "tmdb_full_features.csv"
                df_tmdb = pd.read_csv(tmdb_path)
                df_tmdb = df_tmdb[['movieId', 'runtime']].drop_duplicates('movieId')
                df_tmdb = df_tmdb.set_index('movieId')
                mean_runtime = df_tmdb['runtime'].mean()
                df_tmdb['runtime'] = df_tmdb['runtime'].fillna(mean_runtime)
                min_runtime = df_tmdb['runtime'].min()
                max_runtime = df_tmdb['runtime'].max()
                df_tmdb['runtime'] = (df_tmdb['runtime'] - min_runtime) / (max_runtime - min_runtime)
                df_features = df_features.join(df_tmdb, how='left')
                df_features = df_features.fillna(0)
            
         elif feature_method == "tmdb_vote_count":
                tmdb_path = C.CONTENT_PATH / "tmdb_full_features.csv"
                df_tmdb = pd.read_csv(tmdb_path)
                df_tmdb = df_tmdb[['movieId', 'vote_count']].drop_duplicates('movieId')
                df_tmdb = df_tmdb.set_index('movieId')
                mean_vote_count = df_tmdb['vote_count'].mean()
                df_tmdb['vote_count'] = df_tmdb['vote_count'].fillna(mean_vote_count)
                min_vote_count = df_tmdb['vote_count'].min()
                max_vote_count = df_tmdb['vote_count'].max()
                df_tmdb['vote_count'] = (df_tmdb['vote_count'] - min_vote_count) / (max_vote_count - min_vote_count)
                df_features = df_features.join(df_tmdb, how='left')
                df_features = df_features.fillna(0)
                df_features = df_features.fillna(0)
            
         elif feature_method == "tmdb_cast":
                tmdb_path = C.CONTENT_PATH / "tmdb_full_features.csv"
                df_tmdb = pd.read_csv(tmdb_path)
                df_tmdb = df_tmdb[['movieId', 'cast']].drop_duplicates('movieId')
                df_tmdb['cast'] = df_tmdb['cast'].fillna('')
                tfidf = TfidfVectorizer()
                tfidf_matrix = tfidf.fit_transform(df_tmdb['cast'])
                tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=df_tmdb['movieId'], columns=tfidf.get_feature_names_out())
                df_features = df_features.join(tfidf_df, how='left')
                df_features = df_features.fillna(0)
            
         elif feature_method == "tmdb_director":
                tmdb_path = C.CONTENT_PATH / "tmdb_full_features.csv"
                df_tmdb = pd.read_csv(tmdb_path)
                df_tmdb = df_tmdb[['movieId', 'director']].drop_duplicates('movieId')
                df_tmdb['director'] = df_tmdb['director'].fillna('')
                tfidf = TfidfVectorizer()
                tfidf_matrix = tfidf.fit_transform(df_tmdb['director'])
                tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=df_tmdb['movieId'], columns=tfidf.get_feature_names_out())
                df_features = df_features.join(tfidf_df, how='left')
                df_features = df_features.fillna(0)

         elif feature_method == "tmdb_original_language":
                tmdb_path = C.CONTENT_PATH / "tmdb_full_features.csv"
                df_tmdb = pd.read_csv(tmdb_path)
                df_tmdb = df_tmdb[['movieId', 'original_language']].drop_duplicates('movieId')
                df_tmdb = df_tmdb.set_index('movieId')
                df_tmdb['original_language'] = df_tmdb['original_language'].fillna('unknown')
                # One-hot encoding des langues
                df_lang_dummies = pd.get_dummies(df_tmdb['original_language'], prefix='lang')
                # Gérer les valeurs manquantes après le merge
                df_lang_dummies = df_lang_dummies.reindex(df_features.index, fill_value=0)
                df_features = df_features.join(tfidf_df, how='left')
                df_features = df_features.fillna(0)
         else: # (implement other feature creations here)
            raise NotImplementedError(f'Feature method {features_methods} not yet implemented')
        return df_features

    def fit(self, trainset):
        """Profile Learner"""
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
                    model = Ridge(alpha=15.0)
                elif self.regressor_method == 'gradient_boosting':
                    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
                elif  self.regressor_method == 'knn':
                    model = KNeighborsRegressor(n_neighbors=5)
                elif self.regressor_method == 'elastic_net':
                    model = ElasticNet(alpha=0.1, l1_ratio=0.5)

                else:
                    raise ValueError(f"Unknown regressor method: {self.regressor_method}")
                    
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
