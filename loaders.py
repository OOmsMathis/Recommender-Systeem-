# loaders.py

import pandas as pd
from pathlib import Path
import constants as C_module
C = C_module.Constant()

def load_ratings():
    ratings_filepath = C.EVIDENCE_PATH / C.RATINGS_FILENAME
    df_ratings = pd.read_csv(ratings_filepath)
    # Pas de vérifications de colonnes ici pour rester concis,
    # mais en production, c'est une bonne idée.
    print(f"Chargé df_ratings: {df_ratings.shape} depuis {ratings_filepath}")
    return df_ratings

def load_items():
    """
    Charge movies.csv et le fusionne avec tmdb_full_features.csv.
    'movieId' reste une colonne.
    """
    movies_filepath = C.CONTENT_PATH / C.ITEMS_FILENAME
    tmdb_filepath = C.CONTENT_PATH / C.TMDB_FILENAME

    df_movies = pd.read_csv(movies_filepath)
    print(f"Chargé df_movies: {df_movies.shape} depuis {movies_filepath}")
    
    df_items_rich = df_movies.copy()

    try:
        df_tmdb = pd.read_csv(tmdb_filepath, low_memory=False)
        print(f"Chargé df_tmdb: {df_tmdb.shape} depuis {tmdb_filepath}")

        # Renommer 'id' de TMDB en C.ITEM_ID_COL si besoin
        if 'id' in df_tmdb.columns and C.ITEM_ID_COL not in df_tmdb.columns:
            df_tmdb = df_tmdb.rename(columns={'id': C.ITEM_ID_COL})
        
        # Fusion simple, en priorisant les colonnes de df_movies si conflit (sauf pour la clé)
        # Ou en ne prenant que les colonnes de tmdb qui ne sont pas dans df_movies
        tmdb_cols_to_add = df_tmdb.columns.difference(df_movies.columns).tolist()
        if C.ITEM_ID_COL not in tmdb_cols_to_add: # S'assurer que la clé est là pour la fusion
            tmdb_cols_to_add.append(C.ITEM_ID_COL)
        
        # Conserver les colonnes uniques de TMDB et la clé de jointure
        df_tmdb_subset = df_tmdb[list(set(tmdb_cols_to_add))]


        df_items_rich = pd.merge(
            df_movies, 
            df_tmdb_subset,
            on=C.ITEM_ID_COL, 
            how='left' # Garde tous les films de movies.csv
        )
        print(f"Après fusion avec TMDB: {df_items_rich.shape} lignes")
    except FileNotFoundError:
        print(f"AVERTISSEMENT: Fichier TMDB '{tmdb_filepath}' non trouvé. df_items sera seulement movies.csv.")
    except Exception as e:
        print(f"AVERTISSEMENT: Erreur lors du chargement/fusion de TMDB: {e}. df_items sera seulement movies.csv.")
            
    print(f"DataFrame d'items final (df_items_global): {df_items_rich.shape} lignes")
    # print(f"Colonnes disponibles dans load_items: {df_items_rich.columns.tolist()}")
    return df_items_rich