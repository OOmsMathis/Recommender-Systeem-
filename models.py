# models.py

from collections import defaultdict
import numpy as np
import pandas as pd
import re
import nltk # Utilisé par ContentBased pour title_tfidf
from surprise import AlgoBase, KNNWithMeans, SVD as SurpriseSVD
from surprise.prediction_algorithms.predictions import PredictionImpossible
from sklearn.linear_model import LinearRegression, Ridge, Lasso # ElasticNet
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.neural_network import MLPRegressor
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction.text import TfidfVectorizer

import constants as C_module
C = C_module.Constant()
from loaders import load_ratings, load_items # df_items_global et df_ratings_global sont chargés ici

print("models.py: Chargement des données globales...")
try:
    df_items_global = load_items()
    df_ratings_global = load_ratings() # Ceci est le dataset MovieLens général
    print("models.py: Données globales (items, ratings MovieLens) chargées.")
    if df_items_global.empty: print("models.py: ATTENTION - df_items_global est vide après chargement !")
    if df_ratings_global.empty: print("models.py: ATTENTION - df_ratings_global (MovieLens) est vide après chargement !")
except Exception as e:
    print(f"models.py: ERREUR FATALE lors du chargement des données globales: {e}")
    # Fallback avec des DataFrames vides pour éviter des crashs à l'import
    _cols_items_cb = [getattr(C, col_name, col_name.lower()) for col_name in ['ITEM_ID_COL', 'LABEL_COL', 'RELEASE_YEAR_COL', 'GENRES_COL', 'VOTE_AVERAGE_COL']]
    df_items_global = pd.DataFrame(columns=_cols_items_cb)
    _cols_ratings_cb = [getattr(C, col_name, col_name.lower()) for col_name in ['ITEM_ID_COL', 'RATING_COL', 'USER_ID_COL']]
    df_ratings_global = pd.DataFrame(columns=_cols_ratings_cb)


def get_top_n(predictions, n=10): # Utilisé par les scripts d'évaluation, pas directement ici
    top_n_dict = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions: top_n_dict[uid].append((iid, est))
    for uid, user_ratings in top_n_dict.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n_dict[uid] = user_ratings[:n]
    return top_n_dict

class ContentBased(AlgoBase):
    def __init__(self, features_methods, regressor_method, items_df_for_features=None, ratings_df_for_features=None):
        AlgoBase.__init__(self)
        self.features_methods = features_methods if isinstance(features_methods, list) else [features_methods]
        self.regressor_method = regressor_method
        self.content_features = None # Sera un DataFrame pandas
        self.user_models = {} # dictionnaire: {inner_user_id: trained_regressor_model}
        
        # Permettre de passer des dataframes spécifiques pour la création de features
        # Utile si on entraîne sur un subset (ex: pour un utilisateur spécifique avec son contexte)
        # Par défaut, utilise les dataframes globaux.
        self._items_df = items_df_for_features if items_df_for_features is not None else df_items_global
        self._ratings_df = ratings_df_for_features if ratings_df_for_features is not None else df_ratings_global


    def _normalize_column(self, series_in, fill_zero_with_mean=True):
        series = series_in.squeeze().copy()
        if not pd.api.types.is_numeric_dtype(series):
            series = pd.to_numeric(series, errors='coerce')

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

    def create_content_features(self): # Changé pour utiliser self._items_df et self._ratings_df
        # print(f"ContentBased: Création des features avec les méthodes: {self.features_methods}")
        item_id_col_name = getattr(C, 'ITEM_ID_COL', 'movieId')
        
        if self._items_df.empty or item_id_col_name not in self._items_df.columns:
            print(f"ContentBased: ERREUR - self._items_df est vide ou manque '{item_id_col_name}'. Aucune feature créée.")
            return pd.DataFrame(index=self._items_df.index if item_id_col_name not in self._items_df.columns else pd.Index([]))

        try:
            df_items_indexed = self._items_df.set_index(item_id_col_name)
        except KeyError:
            print(f"ContentBased: ERREUR - '{item_id_col_name}' non trouvé dans self._items_df pour indexation.")
            return pd.DataFrame(index=self._items_df.index)
        except Exception as e:
            print(f"ContentBased: ERREUR inattendue lors de l'indexation de self._items_df: {e}")
            return pd.DataFrame(index=self._items_df.index)


        df_features = pd.DataFrame(index=df_items_indexed.index)
        if not self.features_methods:
            print("ContentBased: Aucune méthode de feature spécifiée.")
            return df_features
        
        for feature_method in self.features_methods:
            current_feature_df_list = [] # Pour stocker les DFs de la feature actuelle (ex: dummies de Genre_binary)
            try:
                if feature_method == "title_length":
                    col = getattr(C, 'LABEL_COL', 'title')
                    if col not in df_items_indexed.columns: print(f"CB: Col {col} manquante pour title_length"); continue
                    series = df_items_indexed[col].fillna('').astype(str).apply(lambda x: len(x))
                    current_feature_df_list.append(self._normalize_column(series, fill_zero_with_mean=False).to_frame('title_length'))

                elif feature_method == "Year_of_release":
                    col = getattr(C, 'RELEASE_YEAR_COL', 'release_year')
                    if col not in df_items_indexed.columns: print(f"CB: Col {col} manquante pour Year_of_release"); continue
                    current_feature_df_list.append(self._normalize_column(df_items_indexed[col].copy(), fill_zero_with_mean=True).to_frame('year_of_release'))
                
                elif feature_method == "average_ratings": # MovieLens average ratings
                    rating_col, item_id_col_ratings = getattr(C, 'RATING_COL', 'rating'), getattr(C, 'ITEM_ID_COL', 'movieId')
                    if self._ratings_df.empty or rating_col not in self._ratings_df or item_id_col_ratings not in self._ratings_df:
                        print(f"CB: Données ratings (self._ratings_df) manquantes ou colonnes incorrectes pour average_ratings"); continue
                    avg_rat = self._ratings_df.groupby(item_id_col_ratings)[rating_col].mean()
                    glob_avg = self._ratings_df[rating_col].mean() # Moyenne globale pour remplir les NaN
                    series = avg_rat.reindex(df_items_indexed.index).fillna(glob_avg if not pd.isna(glob_avg) else 0)
                    current_feature_df_list.append(self._normalize_column(series, fill_zero_with_mean=False).to_frame('average_ml_rating'))

                elif feature_method == "Genre_binary":
                    col = getattr(C, 'GENRES_COL', 'genres')
                    if col not in df_items_indexed.columns: print(f"CB: Col {col} manquante pour Genre_binary"); continue
                    
                    genre_strings = df_items_indexed[col].fillna('').astype(str)
                    # Gérer les cas où les genres sont déjà des listes (si TMDB_FILENAME est utilisé et que GENRES_COL vient de là)
                    if not genre_strings.empty and isinstance(genre_strings.iloc[0], list): # Si c'est une liste de dicts [{id:XX, name:YY},..]
                        # Supposons que 'name' est la clé pour le nom du genre
                        genre_series = genre_strings.apply(lambda x: '|'.join(g['name'] for g in x if isinstance(g, dict) and 'name' in g) if isinstance(x, list) else '')
                    else: # Cas standard où c'est une chaîne "Action|Adventure"
                        genre_series = genre_strings

                    s = genre_series.str.split('|').explode()
                    valid_genres = s[s.str.strip().ne('') & s.str.strip().ne('(no genres listed)')]
                    if not valid_genres.empty:
                        dummies = pd.get_dummies(valid_genres, prefix='genre')
                        # Some genre names might conflict if they contain special characters not suitable for column names
                        dummies.columns = [re.sub(r'\W+', '_', col_name) for col_name in dummies.columns]
                        current_feature_df_list.append(dummies.groupby(dummies.index).sum().astype(int))
                    else: print("CB: Aucun genre valide trouvé pour Genre_binary")
                
                elif feature_method == "Tags_tfidf":
                    tags_file = C.CONTENT_PATH / C.TAGS_FILENAME
                    item_id_c_tags, tag_c = getattr(C, 'ITEM_ID_COL', 'movieId'), getattr(C, 'TAG_COL', 'tag')
                    # USER_ID_COL n'est pas nécessaire ici si on aggrège les tags par item
                    if not tags_file.is_file(): print(f"CB: Fichier tags {tags_file} non trouvé"); continue
                    try:
                        df_tags_source = pd.read_csv(tags_file)
                        if not all(c in df_tags_source.columns for c in [item_id_c_tags, tag_c]):
                            print(f"CB: Colonnes manquantes dans {C.TAGS_FILENAME} pour Tags_tfidf"); continue
                        
                        df_tags = df_tags_source.dropna(subset=[tag_c])
                        df_tags[tag_c] = df_tags[tag_c].astype(str).str.lower()
                        # Grouper les tags par item, en s'assurant que l'index est bien l'ID de l'item
                        grouped_tags = df_tags.groupby(item_id_c_tags)[tag_c].apply(
                            lambda x: ' '.join(sorted(list(x.unique())))
                        ).reindex(df_items_indexed.index).fillna('').to_frame('tags_combined')

                        if not grouped_tags.empty and not grouped_tags['tags_combined'].str.strip().eq('').all():
                            tfidf_vectorizer_tags = TfidfVectorizer(max_features=100, stop_words='english', min_df=2) # min_df pour éviter les tags trop rares
                            matrix_tags = tfidf_vectorizer_tags.fit_transform(grouped_tags['tags_combined'])
                            df_tfidf_tags = pd.DataFrame(matrix_tags.toarray(), index=grouped_tags.index, 
                                                         columns=[f"tfidf_tag_{f}" for f in tfidf_vectorizer_tags.get_feature_names_out()])
                            current_feature_df_list.append(df_tfidf_tags)
                        else: print("CB: Aucun tag combiné pour Tags_tfidf après reindexation et remplissage.")
                    except Exception as e_tag: print(f"ContentBased: Erreur lors du traitement Tags_tfidf: {e_tag}")
                
                elif feature_method == "tmdb_vote_average": # Utilise VOTE_AVERAGE_COL de df_items (qui peut venir de TMDB)
                    col = getattr(C, 'VOTE_AVERAGE_COL', 'vote_average')
                    if col not in df_items_indexed.columns: print(f"CB: Col {col} manquante pour tmdb_vote_average"); continue
                    current_feature_df_list.append(self._normalize_column(df_items_indexed[col], fill_zero_with_mean=False).to_frame('tmdb_vote_average_norm'))

                elif feature_method == "title_tfidf":
                    col_title_tfidf = getattr(C, 'LABEL_COL', 'title')
                    if col_title_tfidf not in df_items_indexed.columns: print(f"CB: Col {col_title_tfidf} manquante pour title_tfidf"); continue
                    try: 
                        from nltk.corpus import stopwords; from nltk.stem import WordNetLemmatizer
                        # S'assurer que les ressources NLTK sont disponibles
                        try: nltk.data.find('corpora/wordnet.zip')
                        except LookupError: nltk.download('wordnet', quiet=True)
                        try: nltk.data.find('corpora/stopwords.zip')
                        except LookupError: nltk.download('stopwords', quiet=True)
                        try: nltk.data.find('corpora/omw-1.4.zip')
                        except LookupError: nltk.download('omw-1.4', quiet=True)
                        try: nltk.data.find('tokenizers/punkt')
                        except LookupError: nltk.download('punkt', quiet=True)
                    except ImportError: print("CB: NLTK non trouvé, title_tfidf ne peut être généré."); continue
                    
                    lemmatizer, stop_words_set = WordNetLemmatizer(), set(stopwords.words('english'))
                    
                    processed_titles = df_items_indexed[col_title_tfidf].fillna('').astype(str).apply(lambda x: ' '.join(
                        [lemmatizer.lemmatize(w) for w in nltk.word_tokenize(x.lower()) if w.isalpha() and w not in stop_words_set and len(w) > 1]))
                    
                    if not processed_titles.str.strip().eq('').all():
                        tfidf_vectorizer_titles = TfidfVectorizer(max_features=100, min_df=2) # min_df pour éviter les termes trop rares
                        matrix_titles = tfidf_vectorizer_titles.fit_transform(processed_titles)
                        df_tfidf_titles = pd.DataFrame(matrix_titles.toarray(), index=df_items_indexed.index, 
                                                       columns=[f"tfidf_title_{f}" for f in tfidf_vectorizer_titles.get_feature_names_out()])
                        current_feature_df_list.append(df_tfidf_titles)
                    else: print("CB: Aucun titre traité pour title_tfidf après nettoyage.")

                else: 
                    print(f"ContentBased: AVERTISSEMENT - Méthode de feature '{feature_method}' non implémentée ou ignorée.")

                # Concaténer toutes les features générées par cette méthode (utile si une méthode produit plusieurs DFs)
                for df_to_add in current_feature_df_list:
                    if not df_to_add.empty:
                        # S'assurer que les index correspondent avant de joindre
                        df_features = df_features.join(df_to_add.reindex(df_features.index), how='left')
            
            except Exception as e_feat_proc:
                print(f"ContentBased: ERREUR lors du traitement de la feature '{feature_method}': {e_feat_proc}")
        
        df_features = df_features.loc[:,~df_features.columns.duplicated()].fillna(0) # Supprimer les colonnes dupliquées et remplir les NaN restants par 0
        if df_features.empty and self.features_methods:
            print(f"ContentBased: ATTENTION - DataFrame de features est vide alors que des méthodes étaient spécifiées: {self.features_methods}")
        # print(f"ContentBased: Features créées, shape: {df_features.shape}. Colonnes: {df_features.columns.tolist()[:10]}...")
        self.content_features = df_features # Stocker les features calculées


    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        # print(f"ContentBased Fit: Début. Régresseur: {self.regressor_method}, Features: {self.features_methods}")
        
        # Créer les features de contenu si elles n'ont pas déjà été calculées
        # ou si on veut les recalculer à chaque fit (par exemple si les _items_df/_ratings_df ont changé)
        self.create_content_features() 
        
        if self.content_features is None or self.content_features.empty:
            print("ContentBased Fit: ATTENTION - content_features vide après création. Aucun profil utilisateur appris.")
            self.user_models = {u_inner_id: None for u_inner_id in self.trainset.all_users()}
            return self

        self.user_models = {} # Réinitialiser les modèles utilisateurs
        for u_inner_id in self.trainset.all_users():
            user_raw_id = self.trainset.to_raw_uid(u_inner_id) # ID brut de l'utilisateur
            user_ratings_tuples = self.trainset.ur[u_inner_id] # Liste de tuples (inner_item_id, rating)
            
            # Il faut un minimum de notes pour entraîner un modèle de régression de manière fiable
            min_ratings_for_model = max(3, self.content_features.shape[1] // 10) # Heuristique: au moins 3 notes, ou 10% du nb de features
            if not user_ratings_tuples or len(user_ratings_tuples) < min_ratings_for_model:
                self.user_models[u_inner_id] = None # Pas assez de données pour cet utilisateur
                continue

            # Extraire les IDs bruts des items notés par l'utilisateur et leurs ratings
            item_raw_ids_for_user = [self.trainset.to_raw_iid(inner_iid) for inner_iid, rating in user_ratings_tuples]
            ratings_for_user = np.array([rating for inner_iid, rating in user_ratings_tuples])
            
            # Obtenir les features de contenu pour ces items
            # S'assurer que self.content_features a un index avant de faire .reindex
            if self.content_features.index.empty and item_raw_ids_for_user:
                 print(f"ContentBased Fit: ATTENTION - L'index de content_features est vide pour user {user_raw_id}.")
                 self.user_models[u_inner_id] = None; continue

            user_item_features_df = self.content_features.reindex(item_raw_ids_for_user).fillna(0)
            
            X_train_user = user_item_features_df.values
            y_train_user = ratings_for_user

            # Vérifier si la matrice X a des données et des features
            if X_train_user.shape[0] < min_ratings_for_model or X_train_user.shape[1] == 0 :
                # print(f"ContentBased Fit: Matrice X ({X_train_user.shape}) inadéquate pour user {user_raw_id}.")
                self.user_models[u_inner_id] = None; continue
            
            model_instance = None
            try:
                if self.regressor_method == 'linear': model_instance = LinearRegression(fit_intercept=True)
                elif self.regressor_method == 'ridge': model_instance = Ridge(alpha=1.0) # Alpha peut être ajusté
                elif self.regressor_method == 'lasso': model_instance = Lasso(alpha=0.1) # Alpha peut être ajusté
                else:
                    print(f"ContentBased Fit: ERREUR - Méthode de régression '{self.regressor_method}' non reconnue pour user {user_raw_id}.")
                    self.user_models[u_inner_id] = None; continue
                
                model_instance.fit(X_train_user, y_train_user)
                self.user_models[u_inner_id] = model_instance
            except Exception as e_fit_model:
                print(f"ContentBased Fit: ERREUR pendant model_instance.fit() pour user {user_raw_id} ({self.regressor_method}): {e_fit_model}")
                self.user_models[u_inner_id] = None
        
        num_models_learned = len([m for m in self.user_models.values() if m is not None])
        # print(f"ContentBased Fit: Terminé. {num_models_learned} modèles utilisateurs appris sur {self.trainset.n_users} utilisateurs dans le trainset.")
        return self
        
    def estimate(self, u_inner_id, i_inner_id):
        # Gérer le cas où u_inner_id est une chaîne 'UKN__...' (utilisateur non trouvé dans le trainset par AlgoBase.predict)
        if isinstance(u_inner_id, str) and u_inner_id.startswith('UKN__'):
            return self.trainset.global_mean 

        # Gérer le cas où i_inner_id est une chaîne 'UKN__...'
        if isinstance(i_inner_id, str) and i_inner_id.startswith('UKN__'):
            return self.trainset.global_mean

        # Si u_inner_id est un entier, il devrait être connu du trainset s'il n'est pas UKN__
        # Mais le user_model pourrait ne pas exister si pas assez de notes pour lui dans fit()
        user_model = self.user_models.get(u_inner_id)
        if user_model is None:
            return self.trainset.global_mean # Pas de modèle spécifique pour cet utilisateur

        if self.content_features is None or self.content_features.empty:
            # print(f"ContentBased Estimate: content_features est vide. Retourne global_mean.")
            return self.trainset.global_mean

        # Convertir l'ID interne de l'item en ID brut pour chercher ses features
        raw_item_id = self.trainset.to_raw_iid(i_inner_id)
        if raw_item_id not in self.content_features.index:
            # print(f"ContentBased Estimate: Item {raw_item_id} non trouvé dans content_features. Retourne global_mean.")
            return self.trainset.global_mean # Item inconnu ou sans features
    
        item_features_vector = self.content_features.loc[raw_item_id].values.reshape(1, -1)
        item_features_vector = np.nan_to_num(item_features_vector) # S'assurer qu'il n'y a pas de NaN
    
        try:
            score = user_model.predict(item_features_vector)[0]
            # Clipper le score dans l'échelle de notation définie
            return np.clip(score, self.trainset.rating_scale[0], self.trainset.rating_scale[1])
        except Exception as e_predict:
            # print(f"ContentBased Estimate: Erreur lors de user_model.predict() pour item {raw_item_id}: {e_predict}. Retourne global_mean.")
            return self.trainset.global_mean

# --- Modèles collaboratifs standards ---
class UserBased(AlgoBase):
    def __init__(self, k=40, min_k=1, sim_options={'name': 'msd', 'user_based': True}, verbose=False, **kwargs):
        AlgoBase.__init__(self)
        self.k = k
        self.min_k = min_k
        self.sim_options = sim_options
        self.verbose = verbose
        # S'assurer que KNNWithMeans est bien initialisé
        self.knn = KNNWithMeans(k=self.k, min_k=self.min_k, sim_options=self.sim_options, verbose=self.verbose, **kwargs)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        self.knn.fit(trainset)
        return self

    def estimate(self, u, i):
        # Gérer le cas où u ou i sont des chaînes 'UKN__...'
        if (isinstance(u, str) and u.startswith('UKN__')) or \
           (isinstance(i, str) and i.startswith('UKN__')):
            # Si l'utilisateur ou l'item est inconnu du trainset original,
            # KNNWithMeans pourrait avoir du mal. Retourner la moyenne globale est plus sûr.
            # La méthode estimate de KNNWithMeans devrait déjà gérer cela et retourner une prédiction (souvent la moyenne).
            # Mais si elle levait une exception pour 'UKN__...' passé directement, ceci est une sécurité.
            pass # Laisser knn.estimate gérer, il a sa propre logique pour les inconnus
        
        try:
            return self.knn.estimate(u, i)
        except PredictionImpossible: # Au cas où knn.estimate lèverait cela pour des raisons internes
            return self.trainset.global_mean


class SVDAlgo(AlgoBase):
    def __init__(self, n_factors=100, n_epochs=20, biased=True, lr_all=0.005, reg_all=0.02, random_state=None, verbose=False, **kwargs):
        AlgoBase.__init__(self)
        self.svd_model = SurpriseSVD(n_factors=n_factors, n_epochs=n_epochs, biased=biased, 
                                     lr_all=lr_all, reg_all=reg_all, 
                                     random_state=random_state, verbose=verbose, **kwargs)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        self.svd_model.fit(trainset)
        return self

    def estimate(self, u, i):
        # Gérer le cas où u ou i sont des chaînes 'UKN__...'
        if (isinstance(u, str) and u.startswith('UKN__')) or \
           (isinstance(i, str) and i.startswith('UKN__')):
            # Laisser svd_model.estimate gérer, il devrait retourner la moyenne pour les inconnus.
            pass
            
        try:
            return self.svd_model.estimate(u, i)
        except PredictionImpossible:
            return self.trainset.global_mean