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

def clean_genre_string(genre_str):
    """
    Nettoie une chaîne de genres.
    Ex: " Action ,Adventure | |  Sci-Fi " -> "Action|Adventure|Sci-Fi"
    """
    if pd.isna(genre_str) or not isinstance(genre_str, str):
        return '' # Retourne une chaîne vide pour les NaN ou les types incorrects
    
    # Remplacer les séparateurs courants (virgule, point-virgule) par des pipes
    cleaned_genres = re.sub(r'\s*[,;/]\s*', '|', genre_str)
    
    # Diviser par pipe, nettoyer chaque genre, et rejoindre
    genres_list = [g.strip() for g in cleaned_genres.split('|') if g.strip()] # Nettoie et enlève les vides
    
    return '|'.join(genres_list)


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

        # **Nettoyage amélioré de la colonne GENRES_COL**
        if C.GENRES_COL in df_items_rich.columns:
            print(f"Nettoyage de la colonne '{C.GENRES_COL}'...")
            df_items_rich[C.GENRES_COL] = df_items_rich[C.GENRES_COL].apply(clean_genre_string)
            print(f"  Colonne '{C.GENRES_COL}' nettoyée et standardisée avec '|' comme séparateur.")

        try:
            df_tmdb = pd.read_csv(tmdb_filepath, low_memory=False)
            print(f"Chargé df_tmdb: {df_tmdb.shape} depuis {tmdb_filepath}")

            if 'id' in df_tmdb.columns and C.ITEM_ID_COL not in df_tmdb.columns:
                df_tmdb = df_tmdb.rename(columns={'id': C.ITEM_ID_COL}) # S'assurer que la colonne ID est bien nommée
            
            if C.ITEM_ID_COL not in df_tmdb.columns:
                print(f"AVERTISSEMENT (load_items): Colonne '{C.ITEM_ID_COL}' non trouvée dans {tmdb_filepath}. TMDB non fusionné.")
            else: 
                tmdb_cols_to_select = [C.ITEM_ID_COL] # Toujours inclure l'ID pour la fusion
                defined_tmdb_constants_attributes = [
                    'GENRES_COL', 'RUNTIME_COL', 'CAST_COL', 'DIRECTORS_COL',
                    'VOTE_COUNT_COL', 'VOTE_AVERAGE_COL', 
                    'POPULARITY_COL', 'BUDGET_COL', 'REVENUE_COL', 'ORIGINAL_LANGUAGE_COL',
                ]
                
                for attr_name in defined_tmdb_constants_attributes:
                    if hasattr(C, attr_name): 
                        col_name = getattr(C, attr_name) 
                        # S'assurer que la colonne existe dans df_tmdb et n'est pas déjà sélectionnée
                        if col_name in df_tmdb.columns and col_name not in tmdb_cols_to_select:
                            tmdb_cols_to_select.append(col_name)
                
                print(f"Colonnes TMDB sélectionnées pour fusion (basées sur constants.py): {tmdb_cols_to_select}")
                df_tmdb_subset = df_tmdb[tmdb_cols_to_select].copy()

                # Si GENRES_COL vient aussi de TMDB, il faut aussi le nettoyer
                if C.GENRES_COL in df_tmdb_subset.columns:
                    print(f"Nettoyage de la colonne '{C.GENRES_COL}' provenant de TMDB...")
                    # TMDB genres sont souvent des listes de dicts JSON stringifiées.
                    # Exemple: "[{'id': 28, 'name': 'Action'}, {'id': 12, 'name': 'Adventure'}]"
                    # Il faut d'abord parser cela pour extraire les noms, puis les joindre par '|'.
                    def parse_tmdb_genres(genre_json_str):
                        if pd.isna(genre_json_str) or not isinstance(genre_json_str, str):
                            return ''
                        try:
                            genre_list = ast.literal_eval(genre_json_str)
                            if isinstance(genre_list, list):
                                names = [item['name'] for item in genre_list if isinstance(item, dict) and 'name' in item]
                                return '|'.join(sorted(list(set(names)))) # Trier pour consistance
                            return ''
                        except (ValueError, SyntaxError):
                            return '' # Si ce n'est pas un JSON valide, retourner vide
                    
                    df_tmdb_subset[C.GENRES_COL] = df_tmdb_subset[C.GENRES_COL].apply(parse_tmdb_genres)
                    print(f"  Colonne '{C.GENRES_COL}' de TMDB nettoyée.")


                if hasattr(C, 'CAST_COL') and C.CAST_COL in df_tmdb_subset.columns:
                    print(f"  Parsing (si str JSON) de la colonne TMDB: {C.CAST_COL}")
                    df_tmdb_subset[C.CAST_COL] = parse_literal_eval_column(df_tmdb_subset[C.CAST_COL])
                    if hasattr(C, 'TMDB_CAST_NAMES_COL'):
                         df_tmdb_subset[C.TMDB_CAST_NAMES_COL] = df_tmdb_subset[C.CAST_COL].apply(lambda x: extract_names(x, key='name', max_items=5))

                if hasattr(C, 'DIRECTORS_COL') and C.DIRECTORS_COL in df_tmdb_subset.columns:
                    print(f"  Parsing (si str JSON) de la colonne TMDB: {C.DIRECTORS_COL}")
                    df_tmdb_subset[C.DIRECTORS_COL] = parse_literal_eval_column(df_tmdb_subset[C.DIRECTORS_COL])
                    # Si vous avez besoin d'extraire les noms des directeurs à partir d'une structure plus complexe (ex: 'crew')
                    # vous devrez adapter cette partie. Pour l'instant, on assume que C.DIRECTORS_COL est soit une liste de noms
                    # soit une structure que parse_literal_eval_column peut gérer.

                # Fusion avec df_items_rich
                # S'il y a des colonnes communes (autre que ITEM_ID_COL), celles de TMDB (df_tmdb_subset) prendront le dessus
                # si elles ne sont pas écrasées par celles de df_movies.
                # La gestion des colonnes de genre est importante ici.
                # Si df_movies a 'genres' et df_tmdb a 'genres', laquelle utiliser ?
                # Actuellement, on nettoie les deux. La fusion va utiliser les colonnes de df_tmdb_subset
                # pour les colonnes qui existent dans les deux, sauf si on les renomme ou les gère spécifiquement.

                # Pour les genres, si df_items_rich a déjà une colonne GENRES_COL (de movies.csv)
                # et que df_tmdb_subset a aussi une colonne GENRES_COL, la fusion va créer GENRES_COL_x et GENRES_COL_y.
                # Il faut décider laquelle garder ou comment les combiner.
                # Option 1: Prioriser TMDB si disponible.
                if C.GENRES_COL in df_items_rich.columns and C.GENRES_COL in df_tmdb_subset.columns:
                    # On va fusionner, puis on utilisera la colonne de TMDB (_y) si elle n'est pas vide, sinon celle de movies.csv (_x)
                    df_items_rich = pd.merge(
                        df_items_rich, 
                        df_tmdb_subset, 
                        on=C.ITEM_ID_COL, 
                        how='left',
                        suffixes=('_movies', '_tmdb') # Suffixes pour différencier
                    )
                    # Combiner les colonnes de genres
                    genres_col_tmdb = C.GENRES_COL + '_tmdb'
                    genres_col_movies = C.GENRES_COL + '_movies'
                    
                    # Utiliser les genres TMDB si présents et non vides, sinon ceux de movies.csv
                    df_items_rich[C.GENRES_COL] = df_items_rich[genres_col_tmdb].fillna('')
                    mask_tmdb_empty = df_items_rich[C.GENRES_COL] == ''
                    df_items_rich.loc[mask_tmdb_empty, C.GENRES_COL] = df_items_rich.loc[mask_tmdb_empty, genres_col_movies].fillna('')
                    
                    # Supprimer les colonnes temporaires _x et _y pour les genres et autres colonnes dupliquées
                    cols_to_drop_after_merge = [col for col in df_items_rich.columns if col.endswith('_movies') or col.endswith('_tmdb')]
                    df_items_rich.drop(columns=cols_to_drop_after_merge, inplace=True, errors='ignore')

                else: # Si GENRES_COL n'est que dans l'un ou l'autre, ou si on ne veut pas de suffixes
                    cols_to_drop_from_main = df_items_rich.columns.intersection(df_tmdb_subset.columns).tolist()
                    if C.ITEM_ID_COL in cols_to_drop_from_main:
                        cols_to_drop_from_main.remove(C.ITEM_ID_COL)
                    
                    df_items_rich = pd.merge(
                        df_items_rich.drop(columns=cols_to_drop_from_main, errors='ignore'), 
                        df_tmdb_subset, 
                        on=C.ITEM_ID_COL, 
                        how='left'
                    )
                print(f"Après fusion avec TMDB: {df_items_rich.shape} lignes")
        except FileNotFoundError:
            print(f"AVERTISSEMENT (load_items): Fichier TMDB '{tmdb_filepath}' non trouvé.")
        except Exception as e:
            print(f"AVERTISSEMENT (load_items): Erreur lors du chargement/fusion de TMDB: {e}.")
        
        try:
            df_links = pd.read_csv(links_filepath)
            print(f"Chargé df_links: {df_links.shape} depuis {links_filepath}")
            if C.ITEM_ID_COL in df_links.columns and C.TMDB_ID_COL in df_links.columns:
                df_links_subset = df_links[[C.ITEM_ID_COL, C.TMDB_ID_COL]].copy()
                df_links_subset[C.TMDB_ID_COL] = pd.to_numeric(df_links_subset[C.TMDB_ID_COL], errors='coerce').astype('Int64')
                
                if C.TMDB_ID_COL in df_items_rich.columns: # Si TMDB_ID_COL existe déjà (ex: de tmdb_full_features)
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

        # Assurer l'unicité des colonnes si des fusions ont créé des doublons (ex: _x, _y non gérés)
        df_items_rich = df_items_rich.loc[:,~df_items_rich.columns.duplicated()]

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

if __name__ == '__main__':
    print("Test de chargement des items...")
    try:
        df_i = load_items()
        print("\nExtrait de df_items_global (après load_items):")
        print(df_i[[C.ITEM_ID_COL, C.LABEL_COL, C.GENRES_COL]].head())
        if C.GENRES_COL in df_i.columns:
            print("\nExemples de chaînes de genres nettoyées:")
            print(df_i[C.GENRES_COL].dropna().sample(5)) # Échantillon de 5 genres non NaN
            
            # Test pour voir les genres uniques qui seraient générés par app.py
            genres_series_test = df_i[C.GENRES_COL].fillna('').astype(str)
            s_genres_test = genres_series_test.str.split('|').explode()
            unique_sidebar_genres_test = sorted([
                g.strip() for g in s_genres_test.unique() if g.strip() and g.strip().lower() != '(no genres listed)'
            ])
            print("\nGenres uniques qui seraient générés pour la sidebar (test):")
            print(unique_sidebar_genres_test[:20]) # Afficher les 20 premiers
            
    except Exception as e_test:
        print(f"Erreur pendant le test de load_items: {e_test}")

    print("\nTest de chargement des ratings...")
    try:
        df_r = load_ratings()
        print("\nExtrait de df_ratings_global (après load_ratings):")
        print(df_r.head())
    except Exception as e_test_r:
        print(f"Erreur pendant le test de load_ratings: {e_test_r}")