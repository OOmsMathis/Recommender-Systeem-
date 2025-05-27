# models.py

from collections import defaultdict
import numpy as np
import pandas as pd
import nltk # Importé ici car utilisé dans ContentBased.create_content_features
from surprise import AlgoBase, KNNWithMeans, SVD as SurpriseSVD
from surprise.prediction_algorithms.predictions import PredictionImpossible
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords # Importé ici
from nltk.stem import WordNetLemmatizer # Importé ici

import constants as C_module
C = C_module.Constant()
from loaders import load_ratings, load_items

# --- Téléchargements NLTK ---
# Déplacés à l'intérieur de create_content_features pour title_tfidf
# pour n'être appelés que si cette feature est utilisée.
# Ou tu peux les laisser ici si tu préfères qu'ils soient faits à l'import.
# nltk.download('stopwords', quiet=True)
# nltk.download('wordnet', quiet=True)
# nltk.download('omw-1.4', quiet=True)


# --- Chargement global des données ---
print("Chargement des données globales pour models.py...")
try:
    df_items_global = load_items()
    df_ratings_global = load_ratings()
    print("Données globales (items, ratings) chargées.")
    if df_items_global.empty: print("ATTENTION: df_items_global est vide après chargement !")
    if df_ratings_global.empty: print("ATTENTION: df_ratings_global est vide après chargement !")
except Exception as e:
    print(f"ERREUR FATALE lors du chargement des données globales: {e}")
    df_items_global = pd.DataFrame()
    df_ratings_global = pd.DataFrame()

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
        self.features_methods = features_methods
        self.regressor_method = regressor_method
        self.content_features = None 
        self.user_models = {}
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words_set = set(stopwords.words('english'))

    def _normalize_column(self, series_in, fill_zero_with_mean=True):
        series = series_in.squeeze().copy().astype(float) # Assure que c'est une Series float
        if series.isnull().all(): return pd.Series(0, index=series.index, name=series.name)
        
        # Remplir NaN: si fill_zero_with_mean, la moyenne est calculée sans les zéros initiaux.
        # Sinon, les NaN sont juste remplis avec la moyenne globale (ou 0 si la moyenne est aussi NaN).
        if fill_zero_with_mean:
            mean_val = series.replace(0, np.nan).mean() # Moyenne sans compter les zéros
        else:
            mean_val = series.mean()
        
        series.fillna(mean_val if not pd.isna(mean_val) else 0, inplace=True)
        
        min_val, max_val = series.min(), series.max()
        if max_val == min_val: return pd.Series(0.5 if min_val != 0 else 0, index=series.index, name=series.name) # Retourne 0.5 si toutes valeurs identiques non nulles
        return (series - min_val) / (max_val - min_val)

    def create_content_features(self, features_methods_list):
        if df_items_global.empty:
            print("ERREUR: df_items_global est vide. Impossible de créer les features de contenu.")
            return pd.DataFrame()
        
        # df_items_global a movieId comme COLONNE. On l'indexe pour la création des features.
        try:
            df_items_indexed = df_items_global.set_index(C.ITEM_ID_COL)
        except KeyError:
            print(f"ERREUR create_content_features: '{C.ITEM_ID_COL}' non dans df_items_global. Colonnes: {df_items_global.columns.tolist()}")
            return pd.DataFrame()

        df_features = pd.DataFrame(index=df_items_indexed.index) # Indexé par movieId

        if not features_methods_list: return df_features
        if isinstance(features_methods_list, str): features_methods_list = [features_methods_list]
        
        print(f"Création des features ContentBased pour les méthodes: {features_methods_list}")

        for feature_method in features_methods_list:
            print(f"  Processing feature: {feature_method}")
            current_feature_df_list = [] # Liste pour stocker les DFs de cette feature avant concat

            try:
                if feature_method == "title_length":
                    if C.LABEL_COL not in df_items_indexed.columns: continue
                    series = df_items_indexed[C.LABEL_COL].apply(lambda x: len(str(x)))
                    df_title_length = self._normalize_column(series).to_frame('title_length')
                    current_feature_df_list.append(df_title_length)

                elif feature_method == "Year_of_release":
                # Utilise directement la colonne C.RELEASE_YEAR_COL créée dans loaders.py
                    if C.RELEASE_YEAR_COL in df_items_indexed.columns:
                        series = df_items_indexed[C.RELEASE_YEAR_COL]
                        # La normalisation gérera les NaN restants
                        df_year = self._normalize_column(series).to_frame('year_of_release')
                        current_feature_df_list.append(df_year)
                    else:
                        print(f"AVERTISSEMENT: Colonne '{C.RELEASE_YEAR_COL}' pour Year_of_release non trouvée dans df_items_indexed.")
                    

                elif feature_method == "average_ratings":
                    if df_ratings_global.empty: continue
                    avg_rat = df_ratings_global.groupby(C.ITEM_ID_COL)[C.RATING_COL].mean()
                    glob_avg = df_ratings_global[C.RATING_COL].mean() # Peut être NaN si df_ratings_global est vide
                    series = avg_rat.reindex(df_items_indexed.index).fillna(glob_avg if not pd.isna(glob_avg) else 0)
                    df_avg_rating = self._normalize_column(series, fill_zero_with_mean=False).to_frame('average_rating') # Ne pas traiter 0 comme spécial ici
                    current_feature_df_list.append(df_avg_rating)
                
                elif feature_method == "count_ratings":
                    if df_ratings_global.empty: continue
                    count_rat = df_ratings_global.groupby(C.ITEM_ID_COL)[C.RATING_COL].size()
                    series = count_rat.reindex(df_items_indexed.index) # fill_zero_with_mean=True par défaut pour normalizer
                    df_rating_count = self._normalize_column(series).to_frame('rating_count')
                    current_feature_df_list.append(df_rating_count)

                elif feature_method == "Genre_binary":
                    if C.GENRES_COL not in df_items_indexed.columns: continue
                    genre_list = df_items_indexed[C.GENRES_COL].fillna('').str.split('|').explode()
                    genre_list = genre_list[genre_list.str.strip().ne('') & genre_list.str.strip().ne('(no genres listed)')]
                    if not genre_list.empty:
                        dummies = pd.get_dummies(genre_list, prefix='genre')
                        df_genres_binary = dummies.groupby(dummies.index).sum() # Index est movieId
                        current_feature_df_list.append(df_genres_binary)

                elif feature_method == "Genre_tfidf":
                    if C.GENRES_COL not in df_items_indexed.columns: continue
                    genre_strings = df_items_indexed[C.GENRES_COL].fillna('').str.replace('|', ' ')
                    if not genre_strings.str.strip().eq('').all():
                        tfidf = TfidfVectorizer(max_features=50) # Moins de features pour genres seuls
                        tfidf_matrix = tfidf.fit_transform(genre_strings)
                        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=df_items_indexed.index, columns=[f"tfidf_genre_{f}" for f in tfidf.get_feature_names_out()])
                        current_feature_df_list.append(tfidf_df)
                
                elif feature_method == "Tags_tfidf": # Renommé car c'est ce que fait ton code
                    tags_filepath = C.CONTENT_PATH / C.TAGS_FILENAME
                    try:
                        df_tags_file = pd.read_csv(tags_filepath)
                        # Gérer 'userID' vs 'userId'
                        tag_user_col = C.USER_ID_TAGS_COL if C.USER_ID_TAGS_COL in df_tags_file.columns else C.USER_ID_COL
                        if tag_user_col not in df_tags_file.columns:
                             raise KeyError(f"Colonne user ID pour tags non trouvée: ni '{C.USER_ID_TAGS_COL}' ni '{C.USER_ID_COL}'")
                        if C.ITEM_ID_COL not in df_tags_file.columns or C.TAG_COL not in df_tags_file.columns:
                             raise KeyError(f"Colonnes '{C.ITEM_ID_COL}' ou '{C.TAG_COL}' manquantes dans {tags_filepath}")

                        df_tags_file = df_tags_file.dropna(subset=[C.TAG_COL])
                        df_tags_file[C.TAG_COL] = df_tags_file[C.TAG_COL].astype(str)
                        df_tags_grouped = df_tags_file.groupby(C.ITEM_ID_COL)[C.TAG_COL].agg(lambda x: ' '.join(x.unique())).to_frame('tags_combined')
                        
                        if not df_tags_grouped['tags_combined'].str.strip().eq('').all():
                            tfidf_tags = TfidfVectorizer(max_features=100) # Nombre de features pour TFIDF sur tags
                            tfidf_matrix_tags = tfidf_tags.fit_transform(df_tags_grouped['tags_combined'])
                            tfidf_df_tags = pd.DataFrame(tfidf_matrix_tags.toarray(), index=df_tags_grouped.index, columns=[f"tfidf_tag_{f}" for f in tfidf_tags.get_feature_names_out()])
                            current_feature_df_list.append(tfidf_df_tags) # Sera joint à df_features plus tard
                    except FileNotFoundError: print(f"AVERTISSEMENT: Fichier {tags_filepath} non trouvé pour 'Tags_tfidf'.")
                    except KeyError as e_key: print(f"AVERTISSEMENT: Erreur de clé pour 'Tags_tfidf': {e_key}")

                elif feature_method == "tmdb_vote_average":
                    if C.VOTE_AVERAGE_COL in df_items_indexed.columns:
                        series = df_items_indexed[C.VOTE_AVERAGE_COL]
                        df_vote_avg = self._normalize_column(series, fill_zero_with_mean=False).to_frame('tmdb_vote_average') # Normaliser TMDB ratings
                        current_feature_df_list.append(df_vote_avg)

                elif feature_method == "title_tfidf":
                    if C.LABEL_COL not in df_items_indexed.columns: continue
                    # NLTK downloads ici si tu veux qu'ils soient spécifiques à cette feature
                    try: stopwords.words('english')
                    except LookupError: nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)

                    titles_processed = df_items_indexed[C.LABEL_COL].fillna('').apply(lambda x: ' '.join(
                        [self.lemmatizer.lemmatize(word) for word in nltk.word_tokenize(x.lower()) 
                         if word.isalpha() and word not in self.stop_words_set and len(word) > 1]
                    ))
                    if not titles_processed.str.strip().eq('').all():
                        tfidf_title = TfidfVectorizer(max_features=100) # Nombre de features pour TFIDF sur titres
                        tfidf_matrix_title = tfidf_title.fit_transform(titles_processed)
                        tfidf_df_title = pd.DataFrame(tfidf_matrix_title.toarray(), index=df_items_indexed.index, columns=[f"tfidf_title_{f}" for f in tfidf_title.get_feature_names_out()])
                        current_feature_df_list.append(tfidf_df_title)
                else:
                    print(f"AVERTISSEMENT: Méthode de feature '{feature_method}' non implémentée.")
                    continue
                
                # Concaténation pour la feature actuelle
                for df_to_add in current_feature_df_list:
                    if not df_to_add.empty:
                        df_features = df_features.join(df_to_add, how='left') # Join sur l'index (movieId)

            except Exception as e:
                print(f"ERREUR globale lors du traitement de la feature '{feature_method}': {e}")
        
        df_features = df_features.loc[:,~df_features.columns.duplicated()] # Gérer colonnes dupliquées
        df_features = df_features.fillna(0) # Remplir les NaNs finaux
        print(f"DataFrame de features final créé avec shape: {df_features.shape}.")
        return df_features

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        self.content_features = self.create_content_features(self.features_methods)
        
        if self.content_features is None or self.content_features.empty:
            print("ContentBased FIT: content_features est vide. Aucun profil utilisateur ne sera appris.")
            self.user_models = {u_inner_id: None for u_inner_id in self.trainset.all_users()}
            return

        self.user_models = {}
        print(f"Entraînement de ContentBased avec régresseur: {self.regressor_method}")
        for u_inner_id in self.trainset.all_users():
            user_ratings_inner = self.trainset.ur[u_inner_id]
            if not user_ratings_inner or len(user_ratings_inner) < 2:
                self.user_models[u_inner_id] = None; continue

            item_raw_ids = [self.trainset.to_raw_iid(inner_iid) for inner_iid, rating in user_ratings_inner]
            ratings = np.array([rating for inner_iid, rating in user_ratings_inner])
            
            user_item_features_df = self.content_features.reindex(item_raw_ids).fillna(0)
            X = user_item_features_df.values
            y = ratings

            if X.shape[0] != y.shape[0] or X.shape[1] == 0: # Si pas de features ou incohérence
                self.user_models[u_inner_id] = None; continue
            
            model = None
            if self.regressor_method == 'linear': model = LinearRegression(fit_intercept=True)
            elif self.regressor_method == 'lasso': model = Lasso(alpha=0.1, max_iter=1000, tol=1e-3)
            elif self.regressor_method == 'random_forest': model = RandomForestRegressor(n_estimators=10, max_depth=10, random_state=42, min_samples_leaf=2) # Plus léger pour tests
            elif self.regressor_method == 'neural_network': model = MLPRegressor(hidden_layer_sizes=(60,30), max_iter=500, random_state=42, early_stopping=True, tol=1e-3) # Ajusté
            elif self.regressor_method == 'decision_tree': model = DecisionTreeRegressor(max_depth=10, random_state=42, min_samples_leaf=2)
            elif self.regressor_method == 'ridge': model = Ridge(alpha=1.0)
            elif self.regressor_method == 'gradient_boosting': model = GradientBoostingRegressor(n_estimators=20, learning_rate=0.1, max_depth=3, random_state=42, min_samples_leaf=2) # Plus léger
            elif self.regressor_method == 'knn': model = KNeighborsRegressor(n_neighbors=5)
            elif self.regressor_method == 'elastic_net': model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000, tol=1e-3)
            
            if model:
                try:
                    model.fit(X, y)
                    self.user_models[u_inner_id] = model
                except: self.user_models[u_inner_id] = None
            else: self.user_models[u_inner_id] = None
        print("Entraînement de ContentBased terminé.")
        
    def estimate(self, u_inner_id, i_inner_id):
        if not (self.trainset.knows_user(u_inner_id) and self.trainset.knows_item(i_inner_id)):
            raise PredictionImpossible('User/item inconnu.')

        user_model = self.user_models.get(u_inner_id)
        if user_model is None: return self.trainset.global_mean

        if self.content_features is None or self.content_features.empty: return self.trainset.global_mean

        raw_item_id = self.trainset.to_raw_iid(i_inner_id)
        if raw_item_id in self.content_features.index:
            item_features_vector = self.content_features.loc[raw_item_id].values.reshape(1, -1)
            item_features_vector = np.nan_to_num(item_features_vector) # Important
        else: return self.trainset.global_mean
    
        try:
            score = user_model.predict(item_features_vector)[0]
            return np.clip(score, C.RATINGS_SCALE[0], C.RATINGS_SCALE[1])
        except: return self.trainset.global_mean

# --- UserBased et SVDAlgo (inchangées) ---
class UserBased(AlgoBase): # ... (code comme avant)
    def __init__(self, k=40, min_k=1, sim_options={'name': 'msd', 'user_based': True}, verbose=False):
        AlgoBase.__init__(self)
        self.k, self.min_k, self.sim_options, self.verbose = k, min_k, sim_options, verbose
        self.knn = None
    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        self.knn = KNNWithMeans(k=self.k, min_k=self.min_k, sim_options=self.sim_options, verbose=self.verbose)
        self.knn.fit(trainset)
        return self
    def estimate(self, u, i):
        if self.knn is None: raise PredictionImpossible("Not fitted.")
        return self.knn.estimate(u, i)

class SVDAlgo(AlgoBase): # ... (code comme avant)
    def __init__(self, n_factors=100, n_epochs=20, biased=True, lr_all=0.005, reg_all=0.02, random_state=None, verbose=False):
        AlgoBase.__init__(self)
        self.n_factors, self.n_epochs, self.biased = n_factors, n_epochs, biased
        self.lr_all, self.reg_all, self.random_state, self.verbose = lr_all, reg_all, random_state, verbose
        self.svd_model = None
    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        self.svd_model = SurpriseSVD(n_factors=self.n_factors, n_epochs=self.n_epochs, biased=self.biased, lr_all=self.lr_all, reg_all=self.reg_all, random_state=self.random_state, verbose=self.verbose)
        self.svd_model.fit(trainset)
        return self
    def estimate(self, u, i):
        if self.svd_model is None: raise PredictionImpossible("Not fitted.")
        return self.svd_model.estimate(u, i)