# constants.py
from pathlib import Path

class Constant:
    DATA_PATH = Path('data/small')
    
    # Fichier pour mapper Prénom (optionnel) à UserID numérique
    USER_PRENOM_TO_ID_MAP_FILE = DATA_PATH / '_user_prenom_to_id_map.json'

    # Content
    CONTENT_PATH = DATA_PATH / 'content'
    ITEMS_FILENAME = 'movies.csv'
    ITEM_ID_COL = 'movieId'
    LABEL_COL = 'title'   
    RELEASE_YEAR_COL = 'release_year'
    GENRES_COL = 'genres'
    
    LINKS_FILENAME = 'links.csv'
    TMDB_ID_COL = 'tmdbId'
    IMDB_ID_COL = 'imdbId'

    TAGS_FILENAME = 'tags.csv'
    USER_ID_COL = 'userId' # Important: utilisé pour les ratings
    TAG_COL = 'tag'
    TIMESTAMP_COL = 'timestamp'

    TMDB_FILENAME = 'tmdb_full_features.csv'
    RUNTIME_COL = 'runtime'
    CAST_COL = 'cast'
    DIRECTORS_COL = 'directors'
    VOTE_COUNT_COL = 'vote_count'
    VOTE_AVERAGE_COL = 'vote_average'
    POPULARITY_COL = 'popularity'
    BUDGET_COL = 'budget'
    REVENUE_COL = 'revenue'
    ORIGINAL_LANGUAGE_COL = 'original_language'

    # Evidence (Source de vérité pour les ratings)
    EVIDENCE_PATH = DATA_PATH / 'evidence'
    RATINGS_FILENAME = 'ratings.csv' # CE FICHIER SERA MIS À JOUR
    RATING_COL = 'rating'
    USER_ITEM_RATINGS_COLS = [USER_ID_COL, ITEM_ID_COL, RATING_COL]

    RATINGS_SCALE = (0.5, 5.0)
    
    EVALUATION_PATH = DATA_PATH / 'evaluations'
    MODELS_RECS_PATH = DATA_PATH / 'recs' # Chemin pour les modèles généraux sauvegardés

    NEW_USER_MOVIES_TO_RATE_COUNT = 20 # Nombre de films à présenter pour notation
    NEW_USER_MIN_RATINGS_FOR_SAVE = 5  # Nombre minimum de notes pour sauvegarder le profil