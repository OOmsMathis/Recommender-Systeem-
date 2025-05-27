# recommender.py (Version complète)

import os
import pandas as pd
from surprise import dump
import re # Ajouté pour re.escape

import constants as C_module
C = C_module.Constant()
import content # Notre module content.py

try:
    from models import df_ratings_global # Nécessaire pour get_movies_watched_by_user
    if df_ratings_global.empty:
        from loaders import load_ratings
        df_ratings_global_direct = load_ratings()
        if not df_ratings_global_direct.empty: df_ratings_global = df_ratings_global_direct
        elif df_ratings_global.empty: print("recommender.py: ERREUR - df_ratings_global vide.")
except ImportError:
    try:
        from loaders import load_ratings
        df_ratings_global = load_ratings()
        if df_ratings_global.empty: raise Exception("df_ratings_global vide.")
    except Exception as e:
        print(f"recommender.py: ERREUR FATALE - Échec chargement df_ratings_global: {e}")
        df_ratings_global = pd.DataFrame()

MODELS_DIR = str(C.DATA_PATH / 'recs')
_loaded_models_cache = {}

def load_model(model_filename):
    if model_filename in _loaded_models_cache: return _loaded_models_cache[model_filename]
    file_path = os.path.join(MODELS_DIR, model_filename)
    if not os.path.exists(file_path): print(f"ERREUR: Modèle '{file_path}' non trouvé."); return None
    try:
        _, loaded_algo = dump.load(file_path)
        _loaded_models_cache[model_filename] = loaded_algo
        print(f"Modèle '{model_filename}' chargé.")
        return loaded_algo
    except Exception as e:
        print(f"ERREUR chargement '{model_filename}': {e}")
        if model_filename in _loaded_models_cache: del _loaded_models_cache[model_filename]
        return None

def get_movies_watched_by_user(user_id):
    if df_ratings_global.empty or C.USER_ID_COL not in df_ratings_global.columns or C.ITEM_ID_COL not in df_ratings_global.columns: return set()
    try: user_id_typed = type(df_ratings_global[C.USER_ID_COL].iloc[0])(user_id) if not df_ratings_global.empty else user_id
    except: user_id_typed = user_id
    return set(df_ratings_global[df_ratings_global[C.USER_ID_COL] == user_id_typed][C.ITEM_ID_COL])

def get_top_n_recommendations(user_id, model_filename, n=15, exclude_watched=True, 
                              filter_genre=None, filter_year_range=None):
    print(f"\nRecos user '{user_id}', modèle '{model_filename}', top {n}, genre: {filter_genre}, années: {filter_year_range}...")
    model = load_model(model_filename)
    if model is None: return pd.DataFrame()

    # Utiliser df_movies_indexed de content.py qui devrait avoir toutes les colonnes nécessaires
    all_movie_details_df = content.df_movies_indexed.reset_index() 
    if all_movie_details_df.empty: print("  ERREUR: Détails films de content.py vides."); return pd.DataFrame()

    items_to_consider = all_movie_details_df.copy()
    
    movies_to_predict_for_ids = items_to_consider[C.ITEM_ID_COL].tolist()
    if exclude_watched:
        movies_watched = get_movies_watched_by_user(user_id)
        movies_to_predict_for_ids = [mid for mid in movies_to_predict_for_ids if mid not in movies_watched]
    
    items_to_consider = items_to_consider[items_to_consider[C.ITEM_ID_COL].isin(movies_to_predict_for_ids)]
    if items_to_consider.empty: print("  Aucun film non vu à considérer."); return pd.DataFrame()

    if filter_genre and hasattr(C, 'GENRES_COL') and C.GENRES_COL in items_to_consider.columns:
        items_to_consider = items_to_consider[items_to_consider[C.GENRES_COL].str.contains(re.escape(filter_genre), case=False, na=False, regex=True)]
    if filter_year_range and hasattr(C, 'RELEASE_YEAR_COL') and C.RELEASE_YEAR_COL in items_to_consider.columns:
        year_min, year_max = filter_year_range
        items_to_consider[C.RELEASE_YEAR_COL] = pd.to_numeric(items_to_consider[C.RELEASE_YEAR_COL], errors='coerce').fillna(0)
        items_to_consider = items_to_consider[
            (items_to_consider[C.RELEASE_YEAR_COL] >= year_min) & 
            (items_to_consider[C.RELEASE_YEAR_COL] <= year_max)
        ]
    if items_to_consider.empty: print(f"  Aucun film ne correspond aux filtres."); return pd.DataFrame()

    movies_to_predict_ids_filtered = items_to_consider[C.ITEM_ID_COL].tolist()
    print(f"  Prédiction pour {len(movies_to_predict_ids_filtered)} films...")
    predictions_list = []
    for movie_id_to_predict in movies_to_predict_ids_filtered:
        pred_obj = model.predict(uid=user_id, iid=movie_id_to_predict)
        predictions_list.append({C.ITEM_ID_COL: pred_obj.iid, 'estimated_score': pred_obj.est})
    
    if not predictions_list: print("  Aucune prédiction générée."); return pd.DataFrame()

    recs_df = pd.DataFrame(predictions_list)
    recs_df = recs_df.sort_values(by='estimated_score', ascending=False).head(n)
    if recs_df.empty: print("  Aucun film dans le top N."); return pd.DataFrame()

    # Enrichir avec les détails de content.py (qui inclut maintenant les notes TMDB)
    recommended_movie_ids = recs_df[C.ITEM_ID_COL].tolist()
    movie_details_list_of_dicts = content.get_movie_details_list(recommended_movie_ids)
    details_df = pd.DataFrame(movie_details_list_of_dicts)

    if details_df.empty or C.ITEM_ID_COL not in details_df.columns:
        print("  AVERTISSEMENT: Aucun détail de film de content.py pour les recos.")
        final_recs_df = recs_df # Retourner au moins les IDs et scores prédits
        # Ajouter des colonnes placeholder si elles manquent pour l'affichage
        for col_attr in ['LABEL_COL', 'GENRES_COL', 'RELEASE_YEAR_COL', 'VOTE_AVERAGE_COL', 'VOTE_COUNT_COL']:
            col_name = getattr(C, col_attr, None)
            if col_name and col_name not in final_recs_df.columns:
                final_recs_df[col_name] = pd.NA
    else:
        final_recs_df = pd.merge(recs_df, details_df, on=C.ITEM_ID_COL, how='left')
        
    # S'assurer que les colonnes de notes TMDB sont bien nommées pour l'affichage
    if hasattr(C, 'VOTE_AVERAGE_COL') and C.VOTE_AVERAGE_COL in final_recs_df.columns:
        final_recs_df = final_recs_df.rename(columns={C.VOTE_AVERAGE_COL: 'tmdb_vote_average'})
    if hasattr(C, 'VOTE_COUNT_COL') and C.VOTE_COUNT_COL in final_recs_df.columns:
        final_recs_df = final_recs_df.rename(columns={C.VOTE_COUNT_COL: 'tmdb_vote_count'})
        
    print(f"  Top {len(final_recs_df)} recommandations finales générées.")
    return final_recs_df

# ... (section if __name__ == '__main__' comme avant) ...