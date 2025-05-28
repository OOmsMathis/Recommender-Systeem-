# loaders.py

import pandas as pd
import re
from pathlib import Path
import ast # Pour parser les chaînes de listes/dictionnaires
import constants as C_module
C = C_module.Constant() # Instancier la classe Constant

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

def extract_names_from_list_of_dicts(data_list, key='name', separator='|'):
    """
    Helper function to extract 'name' from a list of dicts and join them with a separator.
    """
    if not isinstance(data_list, list): # Si ce n'est pas une liste (ex: parsing a échoué)
        return '' 
    names = [item[key] for item in data_list if isinstance(item, dict) and key in item and item[key]]
    # filter(None, names) enlève les chaînes vides si un nom était vide.
    # '|'.join([]) donne '', ce qui est correct.
    return separator.join(filter(None, names))

# --- Fonctions de Chargement ---
def load_ratings():
    ratings_filepath = C.EVIDENCE_PATH / C.RATINGS_FILENAME
    try:
        df_ratings = pd.read_csv(ratings_filepath)
        expected_cols = [C.USER_ID_COL, C.ITEM_ID_COL, C.RATING_COL]
        if not all(col in df_ratings.columns for col in expected_cols):
            print(f"AVERTISSEMENT (load_ratings): Colonnes attendues {expected_cols} non toutes trouvées dans {ratings_filepath}. Colonnes présentes: {df_ratings.columns.tolist()}")
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
        
        if C.ITEM_ID_COL not in df_movies.columns:
            raise KeyError(f"Colonne '{C.ITEM_ID_COL}' non trouvée dans {movies_filepath}.")
        
        if C.LABEL_COL in df_movies.columns and C.RELEASE_YEAR_COL not in df_movies.columns:
            df_movies[C.RELEASE_YEAR_COL] = df_movies[C.LABEL_COL].str.extract(r'\((\d{4})\)\s*$', expand=False).astype(float)
        elif C.RELEASE_YEAR_COL not in df_movies.columns:
            df_movies[C.RELEASE_YEAR_COL] = pd.NA
        
        # Nettoyer les genres de movies.csv (pipe-separated) et garder une sauvegarde
        if C.GENRES_COL in df_movies.columns:
            df_movies[C.GENRES_COL] = df_movies[C.GENRES_COL].astype(str).str.replace(r'\s*,\s*', '|', regex=True)
            df_movies[C.GENRES_COL] = df_movies[C.GENRES_COL].str.replace(r'\|\|+', '|', regex=True)
            df_movies[C.GENRES_COL] = df_movies[C.GENRES_COL].str.strip('|')
            # Sauvegarde des genres de movies.csv avant une potentielle fusion avec TMDB
            df_movies_genres_backup = df_movies[[C.ITEM_ID_COL, C.GENRES_COL]].copy()
            df_movies_genres_backup.rename(columns={C.GENRES_COL: f"{C.GENRES_COL}_movies_csv"}, inplace=True)
        else:
            df_movies_genres_backup = pd.DataFrame(columns=[C.ITEM_ID_COL, f"{C.GENRES_COL}_movies_csv"])


        df_items_rich = df_movies.copy()

        if tmdb_filepath.exists():
            try:
                df_tmdb = pd.read_csv(tmdb_filepath, low_memory=False)

                # Assumer que df_tmdb peut être fusionné sur C.ITEM_ID_COL (movieId)
                # Si df_tmdb utilise 'id' (tmdbId) comme clé, la logique de fusion doit être adaptée
                # pour utiliser links.csv en premier. Pour cet exemple, on part du principe que la fusion directe est possible.
                if C.ITEM_ID_COL not in df_tmdb.columns and 'id' in df_tmdb.columns:
                     # Ce cas est délicat: 'id' dans TMDB est tmdbId, pas movieId.
                     # On ne renomme pas directement, la fusion doit être plus intelligente via links.csv
                     print(f"INFO: Fichier TMDB '{tmdb_filepath}' a 'id' mais pas '{C.ITEM_ID_COL}'. La fusion TMDB directe sur movieId ne sera pas possible sans links.csv d'abord.")
                     pass # On ne fusionne pas TMDB si la clé ITEM_ID_COL n'est pas là
                
                elif C.ITEM_ID_COL in df_tmdb.columns:
                    tmdb_cols_to_select = [C.ITEM_ID_COL]
                    defined_tmdb_constants_attributes = [
                        'GENRES_COL', 'RUNTIME_COL', 'CAST_COL', 'DIRECTORS_COL',
                        'VOTE_COUNT_COL', 'VOTE_AVERAGE_COL', 'POPULARITY_COL', 
                        'BUDGET_COL', 'REVENUE_COL', 'ORIGINAL_LANGUAGE_COL',
                        'RELEASE_YEAR_COL' 
                    ]
                    
                    for attr_name in defined_tmdb_constants_attributes:
                        if hasattr(C, attr_name): 
                            col_name = getattr(C, attr_name) 
                            if col_name in df_tmdb.columns and col_name not in tmdb_cols_to_select:
                                tmdb_cols_to_select.append(col_name)
                    
                    df_tmdb_subset = df_tmdb[tmdb_cols_to_select].copy()

                    if C.GENRES_COL in df_tmdb_subset.columns:
                        df_tmdb_subset[C.GENRES_COL] = parse_literal_eval_column(df_tmdb_subset[C.GENRES_COL])
                        df_tmdb_subset[C.GENRES_COL] = df_tmdb_subset[C.GENRES_COL].apply(
                            lambda x: extract_names_from_list_of_dicts(x, key='name', separator='|')
                        )
                    
                    if C.CAST_COL in df_tmdb_subset.columns:
                        df_tmdb_subset[C.CAST_COL] = parse_literal_eval_column(df_tmdb_subset[C.CAST_COL])
                        df_tmdb_subset[C.CAST_COL] = df_tmdb_subset[C.CAST_COL].apply(lambda x: extract_names_from_list_of_dicts(x, key='name', separator=', '))
                    
                    # Fusion
                    cols_to_drop_from_main = [col for col in df_tmdb_subset.columns if col in df_items_rich.columns and col != C.ITEM_ID_COL]
                    df_items_rich = pd.merge(
                        df_items_rich.drop(columns=cols_to_drop_from_main, errors='ignore'), 
                        df_tmdb_subset, 
                        on=C.ITEM_ID_COL, 
                        how='left'
                    )
            except Exception as e:
                print(f"AVERTISSEMENT (load_items): Erreur lors du traitement de TMDB: {e}.")
        else:
            print(f"INFO (load_items): Fichier TMDB '{tmdb_filepath}' non trouvé.")
        
        # Fusionner avec la sauvegarde des genres de movies.csv pour combler les vides potentiels de TMDB
        if not df_movies_genres_backup.empty:
            df_items_rich = pd.merge(df_items_rich, df_movies_genres_backup, on=C.ITEM_ID_COL, how='left')
            
            # Si C.GENRES_COL (potentiellement de TMDB) est vide, et que la sauvegarde ne l'est pas, utiliser la sauvegarde.
            if C.GENRES_COL in df_items_rich.columns and f"{C.GENRES_COL}_movies_csv" in df_items_rich.columns:
                def fill_empty_genres_from_backup(row):
                    tmdb_genre_val = row[C.GENRES_COL]
                    movies_csv_genre_val = row[f"{C.GENRES_COL}_movies_csv"]
                    
                    # Vérifier si tmdb_genre_val est NaN, None ou une chaîne vide
                    is_tmdb_genre_empty = pd.isna(tmdb_genre_val) or (isinstance(tmdb_genre_val, str) and not tmdb_genre_val.strip())
                    
                    # Vérifier si movies_csv_genre_val est valide
                    is_movies_csv_genre_valid = pd.notna(movies_csv_genre_val) and \
                                                isinstance(movies_csv_genre_val, str) and \
                                                movies_csv_genre_val.strip() and \
                                                movies_csv_genre_val.strip().lower() != "(no genres listed)"

                    if is_tmdb_genre_empty and is_movies_csv_genre_valid:
                        return movies_csv_genre_val
                    return tmdb_genre_val

                df_items_rich[C.GENRES_COL] = df_items_rich.apply(fill_empty_genres_from_backup, axis=1)
                df_items_rich.drop(columns=[f"{C.GENRES_COL}_movies_csv"], inplace=True, errors='ignore')
            elif f"{C.GENRES_COL}_movies_csv" in df_items_rich.columns and C.GENRES_COL not in df_items_rich.columns:
                # Si la colonne GENRES_COL a été complètement supprimée (ne devrait pas arriver avec how='left' ci-dessus)
                # ou si TMDB n'avait pas de colonne GENRES_COL
                df_items_rich.rename(columns={f"{C.GENRES_COL}_movies_csv": C.GENRES_COL}, inplace=True)


        # S'assurer que GENRES_COL est de type string à la fin, et remplacer les NaN/None par ""
        if C.GENRES_COL in df_items_rich.columns:
            df_items_rich[C.GENRES_COL] = df_items_rich[C.GENRES_COL].fillna('').astype(str)
        else: # Si la colonne n'existe toujours pas (cas improbable)
            df_items_rich[C.GENRES_COL] = ''


        if links_filepath.exists() and C.TMDB_ID_COL not in df_items_rich.columns:
            try:
                df_links = pd.read_csv(links_filepath)
                if C.ITEM_ID_COL in df_links.columns and C.TMDB_ID_COL in df_links.columns:
                    df_links_subset = df_links[[C.ITEM_ID_COL, C.TMDB_ID_COL]].copy()
                    df_links_subset[C.TMDB_ID_COL] = pd.to_numeric(df_links_subset[C.TMDB_ID_COL], errors='coerce').astype('Int64')
                    if C.TMDB_ID_COL not in df_items_rich.columns:
                        df_items_rich = pd.merge(df_items_rich, df_links_subset, on=C.ITEM_ID_COL, how='left')
            except Exception as e:
                print(f"AVERTISSEMENT (load_items): Erreur lors du chargement/fusion de links: {e}.")
        elif C.TMDB_ID_COL not in df_items_rich.columns:
             print(f"INFO (load_items): Fichier links '{links_filepath}' non trouvé.")
                
        if C.RELEASE_YEAR_COL in df_items_rich.columns:
            df_items_rich[C.RELEASE_YEAR_COL] = pd.to_numeric(df_items_rich[C.RELEASE_YEAR_COL], errors='coerce').fillna(0).astype(int)

        return df_items_rich

    except FileNotFoundError: 
        print(f"ERREUR Critique (load_items): Fichier movies.csv '{movies_filepath}' non trouvé.")
        raise
    except KeyError as e: 
        print(f"ERREUR Critique de clé (load_items) : {e}.")
        raise
    except Exception as e:
        print(f"Une erreur inattendue est survenue dans load_items: {e}")
        raise

if __name__ == '__main__':
    print("Test de load_items():")
    df_i_test = load_items()
    print("\n premières lignes de df_items:")
    print(df_i_test.head())
    if C.GENRES_COL in df_i_test.columns:
        print(f"\nExemple de valeurs dans la colonne '{C.GENRES_COL}' (5 premières non-NaN):")
        print(df_i_test[C.GENRES_COL].dropna().head())
        print(f"\nNombre de valeurs uniques (incluant vide) pour '{C.GENRES_COL}': {df_i_test[C.GENRES_COL].nunique()}")
        print(f"Nombre de NaN/None pour '{C.GENRES_COL}': {df_i_test[C.GENRES_COL].isnull().sum()}")
        print(f"Nombre de chaînes vides pour '{C.GENRES_COL}': {(df_i_test[C.GENRES_COL] == '').sum()}")

        try:
            # Test d'explosion des genres
            # S'assurer que la colonne est de type string avant de faire .str.split
            genres_series_for_test = df_i_test[C.GENRES_COL].fillna('').astype(str)
            exploded_genres_test = genres_series_for_test.str.split('|').explode()
            unique_exploded_genres = sorted(list(exploded_genres_test.str.strip().unique()))
            # Filtrer les chaînes vides de la liste unique si elles ne sont pas souhaitées
            unique_exploded_genres_filtered = [g for g in unique_exploded_genres if g and g.lower() != "(no genres listed)"]
            print("\nGenres uniques (test d'explosion, filtré):")
            print(unique_exploded_genres_filtered[:20]) # Afficher les 20 premiers pour ne pas surcharger
            if not unique_exploded_genres_filtered or unique_exploded_genres_filtered == ['']:
                print("ATTENTION: Le test d'explosion des genres n'a pas produit de genres uniques valides.")

        except Exception as e_test:
            print(f"Erreur lors du test d'explosion des genres: {e_test}")
    else:
        print(f"Colonne {C.GENRES_COL} non trouvée dans df_items après chargement.")

    print("\nTest de load_ratings():")
    df_r_test = load_ratings()
    print("\n premières lignes de df_ratings:")
    print(df_r_test.head())