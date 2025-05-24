# third parties imports
from pathlib import Path


class Constant:

    DATA_PATH = Path('data/small')  # -- fill here the dataset size to use

    # Content
    CONTENT_PATH = DATA_PATH / 'content'
    # - item
    ITEMS_FILENAME = 'movies.csv'
    ITEM_ID_COL = 'movieId'
    LABEL_COL = 'title'
    GENRES_COL = 'genres'
    # Evidence
    EVIDENCE_PATH = DATA_PATH / 'evidence'
    # - ratings
    RATINGS_FILENAME = 'ratings.csv'
    USER_ID_COL = 'userId'
    RATING_COL = 'rating'
    TIMESTAMP_COL = 'timestamp'
    USER_ITEM_RATINGS = [USER_ID_COL, ITEM_ID_COL, RATING_COL]

    # --- Bibliothèques Implicites Utilisateur (pour recommender_building.py) ---
    IMPLICIT_LIBRARIES_PATH = Path('data/implicit_libraries')
    
    # Le Workshop 2 suggère ML_SMALL_RECS_PATH. Utilisons cela pour la cohérence.
    MODELS_STORAGE_PATH = Path('data/small/recs')
    # {user_id} et {model_name} seront remplacés par des valeurs
    MODEL_STORAGE_FILE_TEMPLATE = "model_user{user_id}_{model_name}.pkl" 
    TRAINSET_STORAGE_FILE_TEMPLATE = "trainset_user{user_id}.pkl"

    IMPLICIT_LIBRARY_FILENAME_TEMPLATE = "library_{user_name}.csv"
    MODEL_STORAGE_FILE_TEMPLATE_NAMED = "model_{user_name}_{model_name}.pkl"
    TRAINSET_STORAGE_FILE_TEMPLATE_NAMED = "trainset_{user_name}.pkl"

    # Modèle général (non personnalisé pour un utilisateur implicite spécifique)
    GENERAL_MODEL_NAME = "svd_general_model.pkl" # Exemple de nom
    GENERAL_MODEL_PATH = MODELS_STORAGE_PATH / GENERAL_MODEL_NAME

    # --- Noms de Colonnes (si besoin de les standardiser) ---
    USER_ID_COL = 'userId'
    MOVIE_ID_COL = 'movieId'
    RATING_COL = 'rating'
    TIMESTAMP_COL = 'timestamp'
    TITLE_COL = 'title'
    GENRES_COL = 'genres'

    # --- Configurations diverses ---
    DEFAULT_N_RECOMMENDATIONS = 10

    # Rating scale
    RATINGS_SCALE = (1, 5)  # -- fill in here the ratings scale as a tuple (min_value, max_value)
    EVALUATION_PATH = Path('data/small/evaluations')



    # --- AJOUTS POUR content.py ---
    # Noms de fichiers (si pas déjà présents ou si vous voulez les standardiser)
    LINKS_FILENAME = "links.csv"
    TAGS_FILENAME = "tags.csv"
    TMDB_FEATURES_FILENAME = "tmdb_full_features.csv" # Nom de votre fichier TMDB-

   
