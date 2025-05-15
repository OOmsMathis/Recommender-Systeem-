# standard library imports
from collections import defaultdict

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
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor


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

""" 
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

 """

#content_based
class ContentBased(AlgoBase):
    def __init__(self, features_method, regressor_method):
        AlgoBase.__init__(self)
        self.features_method = features_method 
        self.regressor_method = regressor_method
        self.content_features = self.create_content_features(features_method)

        

    def create_content_features(self, features_method):
        """Content Analyzer"""
        df_items = load_items()
        df_features = pd.DataFrame(index=df_items.index)
        if features_method is None:
           df_features = pd.DataFrame(index=df_items.index)
        elif features_method == "title_length": # a naive method that creates only 1 feature based on title length
            df_title_length = df_items[C.LABEL_COL].apply(lambda x: len(x)).to_frame('title_length')
            df_title_length['title_length'] = df_title_length['title_length'].fillna(0).astype(int)
            mean_title_length = int(df_title_length['title_length'].replace(0, np.nan).mean())
            df_title_length.loc[df_title_length['title_length'] == 0, 'title_length'] = mean_title_length
            # Normaliser la longueur des titre entre 0 et 1
            title_length_min = df_title_length['title_length'].min()
            title_length_max = df_title_length['title_length'].max()
            df_title_length['title_length'] = (df_title_length['title_length'] - title_length_min) / (title_length_max - title_length_min)
            df_features = pd.concat([df_features, df_title_length], axis=1)
        elif features_method == "Year_of_release":
            year = df_items[C.LABEL_COL].str.extract(r'\((\d{4})\)')[0].astype(float)
            df_year = year.to_frame(name='year_of_release')
            mean_year = df_year.replace(0, np.nan).mean().iloc[0]
            df_year['year_of_release'] = df_year['year_of_release'].fillna(mean_year).astype(int)
             # Normaliser les dates de sortie 
            year_min = df_year['year_of_release'].min()
            year_max = df_year['year_of_release'].max()
            df_year['year_of_release'] = (df_year['year_of_release'] - year_min) / (year_max - year_min)
            df_features = df_features.join(df_year, how='left')
        else: # (implement other feature creations here)
            raise NotImplementedError(f'Feature method {features_method} not yet implemented')
        return df_features
    

    def fit(self, trainset):
        """Profile Learner"""
        self.content_features = self.create_content_features(self.features_method)
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

