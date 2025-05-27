# loaders.py

import pandas as pd
import re
from pathlib import Path
import constants as C_module
C = C_module.Constant() # Instancier la classe Constant
import ast 

# --- Fonctions Helpers ---
def parse_literal_eval_column(series):
    """Helper function to safely parse stringified lists/dicts in a column."""
    def safe_eval(x):
        if isinstance(x, str) and x.startswith(('[', '{')):
            try:
                return ast.literal_eval(x)
            except (ValueError, SyntaxError):
                return [] 
        elif pd.isna(x):
            return [] 
        return x
    return series.apply(safe_eval)

def extract_names(data_list, key='name', max_items=5):
    """Helper function to extract 'name' from a list of dicts."""
    if not isinstance(data_list, list):
        return '' 
    names = [item[key] for item in data_list if isinstance(item, dict) and key in item]
    return ', '.join(names[:max_items])

# --- Fonctions de Chargement ---
def load_ratings():
    ratings_filepath = C.EVIDENCE_PATH / C.RATINGS_FILENAME
    try:
        df_ratings = pd.read_csv(ratings_filepath)
        expected_cols = [C.USER_ID_COL, C.ITEM_ID_COL, C.RATING_COL]
        if not all(col in df_ratings.columns for col in expected_cols):
            print(f"AVERTISSEMENT (load_ratings): Colonnes attendues {expected_cols} non toutes trouvées dans {ratings_filepath}. Colonnes présentes: {df_ratings.columns.tolist()}")
        print(f"Chargé df_ratings: {df_ratings.shape} depuis {ratings_filepath}")
        return df_ratings
    except FileNotFoundError:
        print(f"ERREUR (load_ratings): Fichier ratings non trouvé à '{ratings_filepath}'.")
        raise
    except Exception as e:
        print(f"ERREUR (load_ratings): Erreur inattendue lors du chargement de {ratings_filepath}: {e}")
        raise

def load_items():
    movies_filepath = C.CONTENT_PATH / C.ITEMS_FILENAME
    tmdb_filepath = C.CONTENT_PATH / C.TMDB_FILENAME
    links_filepath = C.CONTENT_PATH / C.LINKS_FILENAME

    try:
        df_movies = pd.read_csv(movies_filepath)
        print(f"Chargé df_movies: {df_movies.shape} depuis {movies_filepath}")

        if C.ITEM_ID_COL not in df_movies.columns:
            raise KeyError(f"Colonne '{C.ITEM_ID_COL}' non trouvée dans {movies_filepath}.")
        
        if C.LABEL_COL in df_movies.columns:
            df_movies[C.RELEASE_YEAR_COL] = df_movies[C.LABEL_COL].str.extract(r'\((\d{4})\)\s*$', expand=False).astype(float)
            print(f"Colonne '{C.RELEASE_YEAR_COL}' créée à partir de '{C.LABEL_COL}' de {C.ITEMS_FILENAME}.")
        else:
            print(f"AVERTISSEMENT (load_items): Colonne titre '{C.LABEL_COL}' non trouvée. Année non extraite du titre.")
            df_movies[C.RELEASE_YEAR_COL] = pd.NA 

        df_items_rich = df_movies.copy()

        if C.GENRES_COL in df_items_rich.columns:
            df_items_rich[C.GENRES_COL] = df_items_rich[C.GENRES_COL].astype(str).str.replace(r'\s*,\s*', '|', regex=True)
            # S'assurer qu'il n'y a pas de pipes doubles si le remplacement en a créé
            df_items_rich[C.GENRES_COL] = df_items_rich[C.GENRES_COL].str.replace(r'\|\|+', '|', regex=True)
            df_items_rich[C.GENRES_COL] = df_items_rich[C.GENRES_COL].str.strip('|') # Enlever les pipes au début/fin
            print(f"  Colonne '{C.GENRES_COL}' nettoyée (virgules remplacées par des pipes).")

        try:
            df_tmdb = pd.read_csv(tmdb_filepath, low_memory=False)
            print(f"Chargé df_tmdb: {df_tmdb.shape} depuis {tmdb_filepath}")

            if 'id' in df_tmdb.columns and C.ITEM_ID_COL not in df_tmdb.columns:
                df_tmdb = df_tmdb.rename(columns={'id': C.ITEM_ID_COL})
            
            if C.ITEM_ID_COL not in df_tmdb.columns:
                print(f"AVERTISSEMENT (load_items): Colonne '{C.ITEM_ID_COL}' non trouvée dans {tmdb_filepath}. TMDB non fusionné.")
            else: 
                tmdb_cols_to_select = [C.ITEM_ID_COL]
                defined_tmdb_constants_attributes = [
                    'GENRES_COL', 'RUNTIME_COL', 'CAST_COL', 'DIRECTORS_COL',
                    'VOTE_COUNT_COL', 'VOTE_AVERAGE_COL', 
                    'POPULARITY_COL', 'BUDGET_COL', 'REVENUE_COL', 'ORIGINAL_LANGUAGE_COL',
                    
                ]
                
                # Vérifier si CREW_COL est explicitement demandé via les constantes pour le parsing
                # S'il n'est pas dans defined_tmdb_constants_attributes, il ne sera pas sélectionné par défaut.
                # Si C.DIRECTORS_COL est défini et existe, on le prendra. Sinon, on ne fait rien avec crew pour les directeurs.
                
                for attr_name in defined_tmdb_constants_attributes:
                    if hasattr(C, attr_name): 
                        col_name = getattr(C, attr_name) 
                        if col_name in df_tmdb.columns and col_name not in tmdb_cols_to_select:
                            tmdb_cols_to_select.append(col_name)
                
                print(f"Colonnes TMDB sélectionnées pour fusion (basées sur constants.py): {tmdb_cols_to_select}")
                df_tmdb_subset = df_tmdb[tmdb_cols_to_select].copy()

                # Parsing de CAST_COL si elle est sélectionnée et définie
                if hasattr(C, 'CAST_COL') and C.CAST_COL in df_tmdb_subset.columns:
                    print(f"  Parsing (si str JSON) de la colonne TMDB: {C.CAST_COL}")
                    df_tmdb_subset[C.CAST_COL] = parse_literal_eval_column(df_tmdb_subset[C.CAST_COL])
                    # Optionnel: créer TMDB_CAST_NAMES_COL si tu l'as défini dans constants.py
                    if hasattr(C, 'TMDB_CAST_NAMES_COL'): # TMDB_CAST_NAMES_COL est un exemple de nouvelle constante
                         df_tmdb_subset[C.TMDB_CAST_NAMES_COL] = df_tmdb_subset[C.CAST_COL].apply(lambda x: extract_names(x, key='name', max_items=5))


                # Parsing de DIRECTORS_COL si elle est sélectionnée et définie (et si c'est une chaîne JSON-like)
                # On ne touche PAS à CREW_COL pour obtenir les directeurs, sauf si tu l'as explicitement demandé
                # via tes feature_methods dans ContentBased.
                if hasattr(C, 'DIRECTORS_COL') and C.DIRECTORS_COL in df_tmdb_subset.columns:
                    # Si C.DIRECTORS_COL est elle-même une chaîne JSON de personnes (improbable mais possible)
                    # ou si elle contient déjà les noms des directeurs sous forme de chaîne simple.
                    # Pour l'instant, on assume qu'elle pourrait nécessiter un parsing si c'est une liste de dicts.
                    # Si c'est déjà une chaîne de noms (ex: "Director1, Director2"), pas besoin de parse_literal_eval_column.
                    # On va supposer qu'elle peut être JSON-like pour être flexible, mais si ce n'est pas le cas,
                    # parse_literal_eval_column ne devrait pas la modifier si ce n'est pas une chaîne commençant par [ ou {.
                    print(f"  Parsing (si str JSON) de la colonne TMDB: {C.DIRECTORS_COL}")
                    df_tmdb_subset[C.DIRECTORS_COL] = parse_literal_eval_column(df_tmdb_subset[C.DIRECTORS_COL])
                    # Si après parsing c'est une liste de dicts, et que tu veux une chaîne de noms :
                    # (Cela suppose que la structure de C.DIRECTORS_COL est une liste de dicts avec une clé 'name')
                    # Tu pourrais avoir besoin d'une constante C.TMDB_DIRECTOR_NAMES_COL pour la colonne finale des noms extraits.
                    # if isinstance(df_tmdb_subset[C.DIRECTORS_COL].iloc[0], list) and hasattr(C, 'TMDB_DIRECTOR_NAMES_COL'):
                    #    df_tmdb_subset[C.TMDB_DIRECTOR_NAMES_COL] = df_tmdb_subset[C.DIRECTORS_COL].apply(lambda x: extract_names(x, key='name', max_items=2))


                cols_to_drop_from_main = df_items_rich.columns.intersection(df_tmdb_subset.columns).tolist()
                if C.ITEM_ID_COL in cols_to_drop_from_main:
                    cols_to_drop_from_main.remove(C.ITEM_ID_COL)
                
                df_items_rich = pd.merge(
                    df_items_rich.drop(columns=cols_to_drop_from_main, errors='ignore'), 
                    df_tmdb_subset, 
                    on=C.ITEM_ID_COL, 
                    how='left'
                )
                print(f"Après fusion avec TMDB (colonnes sélectionnées): {df_items_rich.shape} lignes")

        except FileNotFoundError:
            print(f"AVERTISSEMENT (load_items): Fichier TMDB '{tmdb_filepath}' non trouvé.")
        except Exception as e:
            print(f"AVERTISSEMENT (load_items): Erreur lors du chargement/fusion de TMDB: {e}.")
        
        # Fusionner avec links.csv pour obtenir tmdbId
        try:
            df_links = pd.read_csv(links_filepath)
            print(f"Chargé df_links: {df_links.shape} depuis {links_filepath}")
            if C.ITEM_ID_COL in df_links.columns and C.TMDB_ID_COL in df_links.columns:
                df_links_subset = df_links[[C.ITEM_ID_COL, C.TMDB_ID_COL]].copy()
                df_links_subset[C.TMDB_ID_COL] = pd.to_numeric(df_links_subset[C.TMDB_ID_COL], errors='coerce').astype('Int64')
                
                if C.TMDB_ID_COL in df_items_rich.columns:
                    df_items_rich = df_items_rich.drop(columns=[C.TMDB_ID_COL], errors='ignore')

                df_items_rich = pd.merge(df_items_rich, df_links_subset, on=C.ITEM_ID_COL, how='left')
                print(f"Après fusion avec links (pour {C.TMDB_ID_COL}): {df_items_rich.shape} lignes.")
            else:
                print(f"AVERTISSEMENT (load_items): Colonnes '{C.ITEM_ID_COL}' ou '{C.TMDB_ID_COL}' manquantes dans {links_filepath}.")
        except FileNotFoundError:
            print(f"AVERTISSEMENT (load_items): Fichier links '{links_filepath}' non trouvé.")
        except Exception as e:
            print(f"AVERTISSEMENT (load_items): Erreur lors du chargement/fusion de links: {e}.")
                
        if C.RELEASE_YEAR_COL in df_items_rich.columns:
            df_items_rich[C.RELEASE_YEAR_COL] = pd.to_numeric(df_items_rich[C.RELEASE_YEAR_COL], errors='coerce')

        print(f"DataFrame d'items final (df_items_global): {df_items_rich.shape} lignes")
        print(f"Colonnes disponibles dans df_items_global à la fin de load_items: {df_items_rich.columns.tolist()}")
        return df_items_rich

    except FileNotFoundError: 
        print(f"ERREUR Critique (load_items): Fichier movies.csv '{movies_filepath}' non trouvé.")
        raise
    except KeyError as e: 
        print(f"ERREUR Critique de clé (load_items) sur {movies_filepath}: {e}.")
        raise
    except Exception as e:
        print(f"Une erreur inattendue est survenue dans load_items: {e}")
        raise