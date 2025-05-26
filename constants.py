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

    # Links
    LINKS_FILENAME = 'links.csv'
    ITEMS_ID_COL = 'movieId'
    TMDB_ID_COL = 'tmdbId'
    IMDB_ID_COL = 'imdbId'

    # tags
    TAGS_FILENAME = 'tags.csv'
    USER_ID_COL = 'userID'
    ITEMS_ID_COL = 'movieId'
    TAG_COL = 'tag'
    TIMESTAMP_COL = 'timestamp'

    # Tmdb
    TMDB_FILENAME = 'tmdb_full_features.csv'
    ITEMS_ID_COL = 'movieId'
    RUNTIME_COL = 'runtime'
    GENRES_COL = 'genres'
    CAST_COL = 'cast'
    DIRECTORS_COL = 'directors'
    VOTE_COUNT_COL = 'vote_count'
    VOTE_AVERAGE_COL = 'vote_average'
    POPULARITY_COL = 'popularity'
    BUDGET_COL = 'budget'
    REVENUE_COL = 'revenue'
    ORIGINAL_LANGUAGE_COL = 'original_language'

    # Evidence
    EVIDENCE_PATH = DATA_PATH / 'evidence'
    # - ratings
    RATINGS_FILENAME = 'ratings.csv'
    USER_ID_COL = 'userId'
    RATING_COL = 'rating'
    TIMESTAMP_COL = 'timestamp'
    USER_ITEM_RATINGS = [USER_ID_COL, ITEM_ID_COL, RATING_COL]

    # Rating scale
    RATINGS_SCALE = (1, 5)  # -- fill in here the ratings scale as a tuple (min_value, max_value)
    EVALUATION_PATH = Path('data/small/evaluations')

   
