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

import constants as C_module
C = C_module.Constant()
from loaders import load_ratings, load_items

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
    def __init__(self, features_methods, regressor_method):
        AlgoBase.__init__(self)
        self.features_methods = features_methods if isinstance(features_methods, list) else [features_methods]
        self.regressor_method = regressor_method
        self.content_features = None
        self.user_models = {}

    def _normalize_column(self, series_in, fill_zero_with_mean=True):
        series = series_in.squeeze().copy().astype(float)
        if series.isnull().all(): return pd.Series(0, index=series.index, name=series.name)
        
        mean_val_fill = series.mean() 
        series.fillna(mean_val_fill if not pd.isna(mean_val_fill) else 0, inplace=True)

        if fill_zero_with_mean:
            non_zero_series = series[series != 0]
            if not non_zero_series.empty:
                mean_val_no_zeros = non_zero_series.mean()
                if not pd.isna(mean_val_no_zeros):
                    series = series.replace(0, mean_val_no_zeros)
        
        min_val, max_val = series.min(), series.max()
        if max_val == min_val: return pd.Series(0.5 if min_val != 0 else 0, index=series.index, name=series.name)
        return (series - min_val) / (max_val - min_val)

    def create_content_features(self, features_methods_list):
        item_id_col_name = getattr(C, 'ITEM_ID_COL', 'movieId')
        if df_items_global.empty or item_id_col_name not in df_items_global.columns:
            print(f"ContentBased: ERREUR - df_items_global est vide ou manque {item_id_col_name}. Aucune feature créée.")
            return pd.DataFrame(index=df_items_global.index if item_id_col_name not in df_items_global.columns else pd.Index([]))


        try:
            df_items_indexed = df_items_global.set_index(item_id_col_name)
        except KeyError:
            print(f"ContentBased: ERREUR - '{item_id_col_name}' non trouvé dans df_items_global pour indexation.")
            return pd.DataFrame(index=df_items_global.index)


        df_features = pd.DataFrame(index=df_items_indexed.index)
        if not features_methods_list: return df_features
        
        # print(f"ContentBased: Création des features pour les méthodes: {features_methods_list}")

        for feature_method in features_methods_list:
            current_feature_df_list = []
            try:
                if feature_method == "title_length":
                    col = getattr(C, 'LABEL_COL', 'title')
                    if col not in df_items_indexed.columns: print(f"CB: Col {col} manquante pour title_length"); continue
                    series = df_items_indexed[col].fillna('').apply(lambda x: len(str(x)))
                    current_feature_df_list.append(self._normalize_column(series, fill_zero_with_mean=False).to_frame('title_length'))

                elif feature_method == "Year_of_release":
                    col = getattr(C, 'RELEASE_YEAR_COL', 'release_year')
                    if col not in df_items_indexed.columns: print(f"CB: Col {col} manquante pour Year_of_release"); continue
                    current_feature_df_list.append(self._normalize_column(df_items_indexed[col].copy(), fill_zero_with_mean=True).to_frame('year_of_release'))

                elif feature_method == "average_ratings":
                    rating_col, item_id_col = getattr(C, 'RATING_COL', 'rating'), getattr(C, 'ITEM_ID_COL', 'movieId')
                    if df_ratings_global.empty or rating_col not in df_ratings_global or item_id_col not in df_ratings_global : print(f"CB: Données ratings manquantes pour average_ratings"); continue
                    avg_rat = df_ratings_global.groupby(item_id_col)[rating_col].mean()
                    glob_avg = df_ratings_global[rating_col].mean()
                    series = avg_rat.reindex(df_items_indexed.index).fillna(glob_avg if not pd.isna(glob_avg) else 0)
                    current_feature_df_list.append(self._normalize_column(series, fill_zero_with_mean=False).to_frame('average_ml_rating'))
                
                elif feature_method == "count_ratings":
                    rating_col, item_id_col = getattr(C, 'RATING_COL', 'rating'), getattr(C, 'ITEM_ID_COL', 'movieId')
                    if df_ratings_global.empty or rating_col not in df_ratings_global or item_id_col not in df_ratings_global: print(f"CB: Données ratings manquantes pour count_ratings"); continue
                    count_rat = df_ratings_global.groupby(item_id_col)[rating_col].size()
                    current_feature_df_list.append(self._normalize_column(count_rat.reindex(df_items_indexed.index), fill_zero_with_mean=True).to_frame('ml_rating_count'))

                elif feature_method == "Genre_binary":
                    col = getattr(C, 'GENRES_COL', 'genres')
                    if col not in df_items_indexed.columns: print(f"CB: Col {col} manquante pour Genre_binary"); continue
                    genre_strings = df_items_indexed[col].fillna('').astype(str)
                    s = genre_strings.str.split('|').explode()
                    valid_genres = s[s.str.strip().ne('') & s.str.strip().ne('(no genres listed)')]
                    if not valid_genres.empty:
                        dummies = pd.get_dummies(valid_genres, prefix='genre')
                        current_feature_df_list.append(dummies.groupby(dummies.index).sum().astype(int))
                    else: print("CB: Aucun genre valide trouvé pour Genre_binary")


                elif feature_method == "Tags_tfidf":
                    tags_file = C.CONTENT_PATH / C.TAGS_FILENAME
                    item_id_c, tag_c, user_id_c = getattr(C, 'ITEM_ID_COL', 'movieId'), getattr(C, 'TAG_COL', 'tag'), getattr(C, 'USER_ID_COL', 'userId')
                    if not tags_file.is_file(): print(f"CB: Fichier tags {tags_file} non trouvé pour Tags_tfidf"); continue
                    try:
                        df_tags = pd.read_csv(tags_file)
                        if not all(c in df_tags.columns for c in [item_id_c, tag_c, user_id_c]): print(f"CB: Colonnes manquantes dans tags.csv pour Tags_tfidf"); continue
                        df_tags = df_tags.dropna(subset=[tag_c])
                        df_tags[tag_c] = df_tags[tag_c].astype(str).str.lower()
                        grouped_tags = df_tags.groupby(item_id_c)[tag_c].apply(lambda x: ' '.join(sorted(list(x.unique())))).to_frame('tags_combined')
                        if not grouped_tags.empty and not grouped_tags['tags_combined'].str.strip().eq('').all():
                            tfidf = TfidfVectorizer(max_features=100, stop_words='english')
                            matrix = tfidf.fit_transform(grouped_tags['tags_combined'])
                            current_feature_df_list.append(pd.DataFrame(matrix.toarray(), index=grouped_tags.index, columns=[f"tfidf_tag_{f}" for f in tfidf.get_feature_names_out()]))
                        else: print("CB: Aucun tag combiné pour Tags_tfidf")
                    except Exception as e_tag: print(f"ContentBased: Erreur Tags_tfidf: {e_tag}")

                elif feature_method == "tmdb_vote_average":
                    col = getattr(C, 'VOTE_AVERAGE_COL', 'vote_average')
                    if col not in df_items_indexed.columns: print(f"CB: Col {col} manquante pour tmdb_vote_average"); continue
                    current_feature_df_list.append(self._normalize_column(df_items_indexed[col], fill_zero_with_mean=False).to_frame('tmdb_vote_average'))

                elif feature_method == "title_tfidf":
                    col = getattr(C, 'LABEL_COL', 'title')
                    if col not in df_items_indexed.columns: print(f"CB: Col {col} manquante pour title_tfidf"); continue
                    try: 
                        from nltk.corpus import stopwords; from nltk.stem import WordNetLemmatizer
                        nltk.data.find('corpora/wordnet.zip'); nltk.data.find('corpora/stopwords.zip'); nltk.data.find('corpora/omw-1.4.zip')
                    except LookupError: nltk.download(['wordnet', 'stopwords', 'omw-1.4'], quiet=True)
                    
                    lemmatizer, stop_words = WordNetLemmatizer(), set(stopwords.words('english'))
                    processed_titles = df_items_indexed[col].fillna('').apply(lambda x: ' '.join(
                        [lemmatizer.lemmatize(w) for w in nltk.word_tokenize(x.lower()) if w.isalpha() and w not in stop_words and len(w) > 1]))
                    if not processed_titles.str.strip().eq('').all():
                        tfidf = TfidfVectorizer(max_features=100)
                        matrix = tfidf.fit_transform(processed_titles)
                        current_feature_df_list.append(pd.DataFrame(matrix.toarray(), index=df_items_indexed.index, columns=[f"tfidf_title_{f}" for f in tfidf.get_feature_names_out()]))
                    else: print("CB: Aucun titre traité pour title_tfidf")
                else: 
                    print(f"ContentBased: AVERTISSEMENT - Méthode de feature '{feature_method}' non implémentée ou ignorée.")
                
                for df_to_add in current_feature_df_list:
                    if not df_to_add.empty: df_features = df_features.join(df_to_add, how='left')
            except Exception as e_feat_proc: print(f"ContentBased: ERREUR feature '{feature_method}': {e_feat_proc}")
        
        df_features = df_features.loc[:,~df_features.columns.duplicated()].fillna(0)
        if df_features.empty and len(features_methods_list) > 0 :
            print(f"ContentBased: ATTENTION - DataFrame de features est vide alors que des méthodes étaient spécifiées: {features_methods_list}")
        # print(f"ContentBased: Features créées, shape: {df_features.shape}.")
        return df_features

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        # print(f"ContentBased Fit: Début. Régresseur: {self.regressor_method}, Features: {self.features_methods}")
        self.content_features = self.create_content_features(self.features_methods)
        
        if self.content_features is None or self.content_features.empty:
            print("ContentBased Fit: ATTENTION - content_features vide après création. Aucun profil utilisateur appris.")
            self.user_models = {u_inner_id: None for u_inner_id in self.trainset.all_users()}
            return

        self.user_models = {}
        for u_inner_id in self.trainset.all_users():
            user_raw_id = self.trainset.to_raw_uid(u_inner_id)
            user_ratings_inner = self.trainset.ur[u_inner_id]
            
            min_ratings_for_model = 3
            if not user_ratings_inner or len(user_ratings_inner) < min_ratings_for_model:
                self.user_models[u_inner_id] = None; continue

            item_raw_ids = [self.trainset.to_raw_iid(inner_iid) for inner_iid, rating in user_ratings_inner]
            ratings = np.array([rating for inner_iid, rating in user_ratings_inner])
            
            # S'assurer que self.content_features a un index avant de faire .reindex
            if self.content_features.index.empty and item_raw_ids:
                 print(f"ContentBased Fit: ATTENTION - L'index de content_features est vide pour user {user_raw_id}.")
                 self.user_models[u_inner_id] = None; continue

            user_item_features_df = self.content_features.reindex(item_raw_ids).fillna(0)
            X = user_item_features_df.values
            y = ratings

            if X.shape[0] < min_ratings_for_model or X.shape[1] == 0 :
                # print(f"ContentBased Fit: Matrice X ({X.shape}) inadéquate pour user {user_raw_id}.")
                self.user_models[u_inner_id] = None; continue
            
            # if user_raw_id == -1: # DEBUG pour nouvel utilisateur
            #     print(f"  DEBUG CB Fit (New User {user_raw_id}): X shape: {X.shape}, y shape: {y.shape}, y (ratings): {y}")
            #     print(f"  DEBUG CB Fit (New User {user_raw_id}): user_item_features_df (head):\n{user_item_features_df.head()}")

            model = None
            if self.regressor_method == 'linear': model = LinearRegression(fit_intercept=True)
            elif self.regressor_method == 'ridge': model = Ridge(alpha=1.0) 
            elif self.regressor_method == 'lasso': model = Lasso(alpha=0.05)
            else:
                print(f"ContentBased Fit: ERREUR - Méthode de régression '{self.regressor_method}' non reconnue pour user {user_raw_id}.")
                self.user_models[u_inner_id] = None; continue 
            
            try:
                model.fit(X, y)
                self.user_models[u_inner_id] = model
                # if user_raw_id == -1 and hasattr(model, 'coef_'): 
                #     print(f"  DEBUG CB Fit (New User {user_raw_id}): Modèle {self.regressor_method} entraîné. Coefs (10 premiers): {model.coef_[:10]}")
            except Exception as e_fit_model:
                print(f"ContentBased Fit: ERREUR pendant model.fit() pour user {user_raw_id} ({self.regressor_method}): {e_fit_model}")
                self.user_models[u_inner_id] = None
        
        num_models_learned = len([m for m in self.user_models.values() if m is not None])
        # print(f"ContentBased Fit: Terminé. {num_models_learned} modèles utilisateurs appris sur {self.trainset.n_users}.")
        
    def estimate(self, u_inner_id, i_inner_id):
        user_raw_id = self.trainset.to_raw_uid(u_inner_id)
        if not (self.trainset.knows_user(u_inner_id) and self.trainset.knows_item(i_inner_id)):
            raise PredictionImpossible('User et/ou item inconnu.')

        user_model = self.user_models.get(u_inner_id)
        
        if user_model is None:
            # if user_raw_id == -1: print(f"CB Estimate (New User {user_raw_id}): Pas de modèle perso. Retourne global_mean: {self.trainset.global_mean:.2f}")
            return self.trainset.global_mean

        if self.content_features is None or self.content_features.empty:
            # if user_raw_id == -1: print(f"CB Estimate (New User {user_raw_id}): content_features vide. Retourne global_mean.")
            return self.trainset.global_mean

        raw_item_id = self.trainset.to_raw_iid(i_inner_id)
        if raw_item_id not in self.content_features.index:
            # if user_raw_id == -1: print(f"CB Estimate (New User {user_raw_id}): Item {raw_item_id} non trouvé dans CF. Retourne global_mean.")
            return self.trainset.global_mean
    
        item_features_vector = self.content_features.loc[raw_item_id].values.reshape(1, -1)
        item_features_vector = np.nan_to_num(item_features_vector)
    
        try:
            score = user_model.predict(item_features_vector)[0]
            # if user_raw_id == -1: print(f"  DEBUG CB Estimate (New User {user_raw_id}): Item {raw_item_id}, Score brut: {score:.2f}")
            return np.clip(score, self.trainset.rating_scale[0], self.trainset.rating_scale[1])
        except Exception:
            # if user_raw_id == -1: print(f"CB Estimate (New User {user_raw_id}): Erreur user_model.predict pour Item {raw_item_id}. Retourne global_mean.")
            return self.trainset.global_mean

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