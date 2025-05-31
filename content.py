# content.py (extrait pertinent)

import pandas as pd
import constants as C_module
C = C_module.Constant()

try:
    from models import df_items_global 
    if df_items_global.empty:
        print("content.py: AVERTISSEMENT - df_items_global importé de models.py est vide.")
        raise ImportError 
except ImportError:
    print("content.py: Tentative de chargement direct de df_items_global via loaders...")
    try:
        from loaders import load_items
        df_items_global = load_items()
        if df_items_global.empty:
             raise Exception("Chargement direct via loaders a aussi résulté en df_items_global vide.")
    except Exception as e:
        print(f"content.py: ERREUR FATALE - Échec du chargement de df_items_global: {e}")
        _cols = [getattr(C, col_attr, col_attr.lower()) for col_attr in ['ITEM_ID_COL', 'LABEL_COL', 'GENRES_COL', 'RELEASE_YEAR_COL', 'VOTE_AVERAGE_COL', 'VOTE_COUNT_COL', 'TMDB_ID_COL'] if hasattr(C, col_attr)]
        df_items_global = pd.DataFrame(columns=_cols)


if not df_items_global.empty and C.ITEM_ID_COL in df_items_global.columns:
    try:
        df_movies_indexed = df_items_global.set_index(C.ITEM_ID_COL)
    except Exception as e:
        print(f"content.py: Erreur indexation df_items_global: {e}")
        df_movies_indexed = pd.DataFrame(index=pd.Index([], name=C.ITEM_ID_COL))
else:
    df_movies_indexed = pd.DataFrame(index=pd.Index([], name=C.ITEM_ID_COL))


def get_movie_title(movie_id):
    if C.LABEL_COL not in df_movies_indexed.columns: return "Titre Indisponible"
    try: return df_movies_indexed.loc[movie_id, C.LABEL_COL]
    except KeyError: return f"Film ID {movie_id} Non Trouvé"
    except: return "Erreur Titre"

def get_movie_genres(movie_id):
    if C.GENRES_COL not in df_movies_indexed.columns: return "Genres Indisponibles"
    try:
        genres = df_movies_indexed.loc[movie_id, C.GENRES_COL]
        return genres if pd.notna(genres) else "Genres Non Spécifiés"
    except KeyError: return f"Film ID {movie_id} Non Trouvé"
    except: return "Erreur Genres"

def get_movie_release_year(movie_id):
    if not hasattr(C, 'RELEASE_YEAR_COL') or C.RELEASE_YEAR_COL not in df_movies_indexed.columns: return "Année Indisponible"
    try:
        year = df_movies_indexed.loc[movie_id, C.RELEASE_YEAR_COL]
        return int(year) if pd.notna(year) and year != 0 else "N/A"
    except KeyError: return f"Film ID {movie_id} Non Trouvé"
    except: return "Année Invalide"

def get_movie_tmdb_vote_average(movie_id):
    if not hasattr(C, 'VOTE_AVERAGE_COL') or C.VOTE_AVERAGE_COL not in df_movies_indexed.columns:
        return None 
    try:
        vote_avg = df_movies_indexed.loc[movie_id, C.VOTE_AVERAGE_COL]
        return float(vote_avg) if pd.notna(vote_avg) else None
    except KeyError: return None 
    except: return None 

def get_movie_tmdb_vote_count(movie_id):
    if not hasattr(C, 'VOTE_COUNT_COL') or C.VOTE_COUNT_COL not in df_movies_indexed.columns:
        return None
    try:
        vote_count = df_movies_indexed.loc[movie_id, C.VOTE_COUNT_COL]
        return int(vote_count) if pd.notna(vote_count) else None
    except KeyError: return None
    except: return None

def get_movie_details_list(movie_id_list):
    details = []
    if df_movies_indexed.empty:
        for movie_id in movie_id_list: 
            details.append({ 
                C.ITEM_ID_COL: movie_id, 
                C.LABEL_COL: "Info Indisponible", 
                C.GENRES_COL: "", 
                C.RELEASE_YEAR_COL: "",
                C.VOTE_AVERAGE_COL: None, 
                C.VOTE_COUNT_COL: None,
                C.TMDB_ID_COL: None # MODIFIED: Ensure placeholder includes TMDB_ID_COL
            }) 
        return details

    cols_to_fetch = [C.LABEL_COL, C.GENRES_COL]
    if hasattr(C, 'RELEASE_YEAR_COL') and C.RELEASE_YEAR_COL in df_movies_indexed.columns: 
        cols_to_fetch.append(C.RELEASE_YEAR_COL)
    if hasattr(C, 'VOTE_AVERAGE_COL') and C.VOTE_AVERAGE_COL in df_movies_indexed.columns: 
        cols_to_fetch.append(C.VOTE_AVERAGE_COL)
    if hasattr(C, 'VOTE_COUNT_COL') and C.VOTE_COUNT_COL in df_movies_indexed.columns: 
        cols_to_fetch.append(C.VOTE_COUNT_COL)
    
    # MODIFIED: Explicitly add TMDB_ID_COL to cols_to_fetch if available
    if hasattr(C, 'TMDB_ID_COL') and C.TMDB_ID_COL in df_movies_indexed.columns:
        cols_to_fetch.append(C.TMDB_ID_COL)
    
    # S'assurer de ne prendre que les colonnes qui existent vraiment
    existing_cols_to_fetch = [col for col in cols_to_fetch if col in df_movies_indexed.columns]
    
    valid_ids_in_list = [mid for mid in movie_id_list if mid in df_movies_indexed.index]
    
    if valid_ids_in_list:
        # Ensure ITEM_ID_COL is fetched if it's the index, by resetting index before loc then setting it back,
        # or by ensuring it's part of existing_cols_to_fetch if not the index (though it is).
        movies_data = df_movies_indexed.loc[valid_ids_in_list, existing_cols_to_fetch].reset_index()
        for _, row in movies_data.iterrows():
            movie_info = {C.ITEM_ID_COL: row[C.ITEM_ID_COL]} # ITEM_ID_COL comes from reset_index()
            for col in existing_cols_to_fetch: # Iterate through specifically fetched columns
                movie_info[col] = row.get(col) 
            details.append(movie_info)
    
    ids_not_found = set(movie_id_list) - set(valid_ids_in_list)
    for movie_id_nf in ids_not_found:
        placeholder = {C.ITEM_ID_COL: movie_id_nf, C.LABEL_COL: f"Film ID {movie_id_nf} Non Trouvé"}
        # Ensure all expected columns (including TMDB_ID_COL) have a placeholder value
        all_expected_display_cols = [
            C.GENRES_COL, C.RELEASE_YEAR_COL, C.VOTE_AVERAGE_COL, C.VOTE_COUNT_COL, C.TMDB_ID_COL
        ]
        for col in all_expected_display_cols:
            if col not in placeholder: # Only add if not already ITEM_ID or LABEL
                 placeholder[getattr(C, col, col) if hasattr(C,col) else col] = None # Use constant value if available
        details.append(placeholder)
    return details

def get_all_movies_for_selection():
    if df_items_global.empty or C.ITEM_ID_COL not in df_items_global or C.LABEL_COL not in df_items_global:
        return pd.DataFrame({C.ITEM_ID_COL: [], C.LABEL_COL: []})
    return df_items_global[[C.ITEM_ID_COL, C.LABEL_COL]].copy()

if __name__ == '__main__':
    if not df_movies_indexed.empty and not df_movies_indexed.index.empty:
        test_id = df_movies_indexed.index[0]
        print(f"TMDB Vote Avg for {test_id}: {get_movie_tmdb_vote_average(test_id)}")
        print(f"TMDB Vote Count for {test_id}: {get_movie_tmdb_vote_count(test_id)}")
        
        # Test get_movie_details_list
        test_ids_list = []
        if len(df_movies_indexed.index) > 5:
            test_ids_list = df_movies_indexed.index[:5].tolist()
        if test_ids_list:
            print(f"\nTesting get_movie_details_list for IDs: {test_ids_list}")
            details_output = get_movie_details_list(test_ids_list)
            for detail in details_output:
                print(detail)
        # Test with a non-existent ID
        print("\nTesting get_movie_details_list with a non-existent ID (e.g., -1):")
        details_non_existent = get_movie_details_list([-1])
        for detail in details_non_existent:
            print(detail)