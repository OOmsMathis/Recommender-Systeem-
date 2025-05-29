# recommender.py

import os
import pandas as pd
import re
from surprise import dump, Dataset, Reader

import constants as C_module
C = C_module.Constant()
import content
import explanations

try:
    from models import df_ratings_global, df_items_global
    if df_ratings_global.empty or df_items_global.empty:
        print("recommender.py: AVERTISSEMENT - df_ratings_global ou df_items_global importé de models.py est vide.")
        from loaders import load_ratings, load_items
        if df_ratings_global.empty:
            df_ratings_global_direct = load_ratings()
            if not df_ratings_global_direct.empty: df_ratings_global = df_ratings_global_direct
        if df_items_global.empty:
            df_items_global_direct = load_items()
            if not df_items_global_direct.empty: df_items_global = df_items_global_direct
        if df_ratings_global.empty or df_items_global.empty:
             print("recommender.py: ERREUR CRITIQUE - df_ratings_global ou df_items_global toujours vide.")
except ImportError:
    try:
        from loaders import load_ratings, load_items
        df_ratings_global = load_ratings()
        df_items_global = load_items()
        if df_ratings_global.empty or df_items_global.empty:
            raise Exception("df_ratings_global ou df_items_global vide après chargement direct.")
    except Exception as e:
        print(f"recommender.py: ERREUR FATALE - Échec chargement df_ratings_global/df_items_global: {e}")
        df_ratings_global = pd.DataFrame()
        df_items_global = pd.DataFrame()

MODELS_DIR = str(C.DATA_PATH / 'recs')
_loaded_models_cache = {}
_loaded_trainset_cache = None

def get_full_trainset():
    global _loaded_trainset_cache
    if _loaded_trainset_cache is not None:
        return _loaded_trainset_cache
    if df_ratings_global.empty:
        print("Recommender (get_full_trainset): df_ratings_global est vide.")
        return None
    try:
        reader = Reader(rating_scale=C.RATINGS_SCALE)
        cols_for_surprise = [C.USER_ID_COL, C.ITEM_ID_COL, C.RATING_COL]
        if not all(col in df_ratings_global.columns for col in cols_for_surprise):
            print(f"Recommender (get_full_trainset): Colonnes manquantes.")
            return None
        data = Dataset.load_from_df(df_ratings_global[cols_for_surprise], reader)
        _loaded_trainset_cache = data.build_full_trainset()
        print("Recommender: Trainset Surprise complet construit et mis en cache.")
        return _loaded_trainset_cache
    except Exception as e:
        print(f"Recommender: Erreur construction trainset Surprise: {e}")
        return None

def load_model(model_filename):
    if model_filename in _loaded_models_cache:
        return _loaded_models_cache[model_filename]
    file_path = os.path.join(MODELS_DIR, model_filename)
    if not os.path.exists(file_path):
        print(f"ERREUR: Modèle '{file_path}' non trouvé.")
        return None
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
    if df_ratings_global.empty or C.USER_ID_COL not in df_ratings_global.columns or C.ITEM_ID_COL not in df_ratings_global.columns:
        return set()
    try:
        user_id_typed = type(df_ratings_global[C.USER_ID_COL].iloc[0])(user_id)
    except (ValueError, IndexError):
        user_id_typed = user_id
    return set(df_ratings_global[df_ratings_global[C.USER_ID_COL] == user_id_typed][C.ITEM_ID_COL])


def get_top_n_recommendations(user_id, model_filename, n=15, exclude_watched=True,
                              filter_genre=None, filter_year_range=None):
    print(f"\nRecos user '{user_id}', modèle '{model_filename}', top {n}, genre: {filter_genre}, années: {filter_year_range}...")
    
    model = load_model(model_filename)
    if model is None: return pd.DataFrame()

    model_type_key = None
    if 'svd' in model_filename.lower(): model_type_key = 'svd'
    elif 'user_based' in model_filename.lower(): model_type_key = 'user_based'
    elif 'content_based' in model_filename.lower(): model_type_key = 'content_based'

    trainset_for_explanation = None # Pour SVD
    if model_type_key == 'svd':
        trainset_for_explanation = get_full_trainset()
        if trainset_for_explanation is None:
            print("AVERTISSEMENT: Trainset manquant pour explications SVD.")

    all_movie_details_df = content.df_movies_indexed.reset_index() if not content.df_movies_indexed.empty else df_items_global.copy()
    if all_movie_details_df.empty or C.ITEM_ID_COL not in all_movie_details_df.columns:
        print("  ERREUR: Détails films (all_movie_details_df) vides ou ITEM_ID_COL manquant.")
        return pd.DataFrame()

    items_to_consider = all_movie_details_df.copy()
    movies_to_predict_for_ids_initial = items_to_consider[C.ITEM_ID_COL].unique().tolist()
    
    if exclude_watched:
        movies_watched = get_movies_watched_by_user(user_id)
        movies_to_predict_for_ids_initial = [mid for mid in movies_to_predict_for_ids_initial if mid not in movies_watched]
    
    items_to_predict_df = items_to_consider[items_to_consider[C.ITEM_ID_COL].isin(movies_to_predict_for_ids_initial)]

    if filter_genre and hasattr(C, 'GENRES_COL') and C.GENRES_COL in items_to_predict_df.columns:
        try:
            items_to_predict_df = items_to_predict_df[items_to_predict_df[C.GENRES_COL].str.contains(re.escape(filter_genre), case=False, na=False, regex=True)]
        except Exception as e_genre_filter: print(f"  Erreur filtre genre: {e_genre_filter}")

    if filter_year_range and hasattr(C, 'RELEASE_YEAR_COL') and C.RELEASE_YEAR_COL in items_to_predict_df.columns:
        year_min, year_max = filter_year_range
        items_to_predict_df[C.RELEASE_YEAR_COL] = pd.to_numeric(items_to_predict_df[C.RELEASE_YEAR_COL], errors='coerce').fillna(0)
        items_to_predict_df = items_to_predict_df[
            (items_to_predict_df[C.RELEASE_YEAR_COL] >= year_min) & 
            (items_to_predict_df[C.RELEASE_YEAR_COL] <= year_max)
        ]
    
    if items_to_predict_df.empty:
        print(f"  Aucun film ne correspond aux filtres pour la prédiction.")
        return pd.DataFrame()

    final_movies_to_predict_ids = items_to_predict_df[C.ITEM_ID_COL].tolist()
    
    # --- ÉTAPE 1: Prédire les scores pour tous les films candidats ---
    print(f"  Prédiction des scores pour {len(final_movies_to_predict_ids)} films après filtrage...")
    predictions_scores_list = []
    for movie_id_to_predict in final_movies_to_predict_ids:
        try:
            pred_obj = model.predict(uid=user_id, iid=movie_id_to_predict)
            predictions_scores_list.append({
                C.ITEM_ID_COL: pred_obj.iid,
                'estimated_score': pred_obj.est
            })
        except Exception: # Ignorer les erreurs de prédiction de score pour un item
            continue 
    
    if not predictions_scores_list:
        print("  Aucun score de prédiction généré.")
        return pd.DataFrame()

    # --- ÉTAPE 2: Trier et sélectionner le Top N (le 'n' passé à la fonction) ---
    raw_recs_df = pd.DataFrame(predictions_scores_list)
    # S'assurer que 'estimated_score' est numérique pour le tri
    raw_recs_df['estimated_score'] = pd.to_numeric(raw_recs_df['estimated_score'], errors='coerce')
    raw_recs_df = raw_recs_df.sort_values(by='estimated_score', ascending=False).head(n)

    if raw_recs_df.empty:
        print(f"  Aucun film dans le top {n} après tri des scores.")
        return pd.DataFrame()

    # --- ÉTAPE 3: Générer les explications SEULEMENT pour ces Top N films ---
    print(f"  Génération des explications pour les {len(raw_recs_df)} meilleurs films...")
    explained_recommendations_list = []
    
    for _, row in raw_recs_df.iterrows():
        movie_id_to_explain = row[C.ITEM_ID_COL]
        estimated_score = row['estimated_score']
        explanation_text = "Explication non disponible." # Default

        if model_type_key:
            current_trainset_for_expl = trainset_for_explanation if model_type_key == 'svd' else getattr(model, 'trainset', None)
            
            # Gérer les cas où le trainset pourrait manquer pour certains modèles
            if model_type_key == 'svd' and current_trainset_for_expl is None:
                 explanation_text = "Profil de goût général (SVD)." # Explication par défaut si trainset SVD manque
            elif current_trainset_for_expl is None and model_type_key not in ['content_based']:
                 explanation_text = f"Suggestion basée sur le modèle {model_type_key}."
            else:
                try:
                    explanation_text = explanations.get_explanation_for_recommendation(
                        user_id, movie_id_to_explain, model, model_type_key, current_trainset_for_expl
                    )
                except Exception as e_expl:
                    print(f"  Erreur génération explication pour movie_id {movie_id_to_explain}: {e_expl}")
                    explanation_text = "Explication temporairement indisponible."
        
        explained_recommendations_list.append({
            C.ITEM_ID_COL: movie_id_to_explain,
            'estimated_score': estimated_score,
            'explanation': explanation_text
        })
    
    recs_df_with_explanations = pd.DataFrame(explained_recommendations_list)

    # --- ÉTAPE 4: Enrichir avec les détails des films ---
    if recs_df_with_explanations.empty:
        print("  Aucune recommandation après génération des explications.")
        return pd.DataFrame()

    recommended_movie_ids = recs_df_with_explanations[C.ITEM_ID_COL].tolist()
    # Utiliser content.get_movie_details_list qui est optimisé
    movie_details_list_of_dicts = content.get_movie_details_list(recommended_movie_ids)
    details_df = pd.DataFrame(movie_details_list_of_dicts)

    final_recs_df = pd.DataFrame() # Initialiser au cas où le merge échoue ou details_df est vide
    if not details_df.empty and C.ITEM_ID_COL in details_df.columns:
        final_recs_df = pd.merge(recs_df_with_explanations, details_df, on=C.ITEM_ID_COL, how='left')
    else: # Fallback si les détails ne peuvent être mergés
        print("  AVERTISSEMENT: Aucun détail de film de content.py pour les recos finales. Utilisation des données brutes.")
        final_recs_df = recs_df_with_explanations.copy()
        # S'assurer que les colonnes minimales pour l'affichage existent
        cols_to_ensure = [C.LABEL_COL, C.GENRES_COL, C.RELEASE_YEAR_COL, C.VOTE_AVERAGE_COL, C.TMDB_ID_COL]
        for col_ensure in cols_to_ensure:
            if col_ensure not in final_recs_df.columns:
                final_recs_df[col_ensure] = "N/A" if col_ensure != C.VOTE_AVERAGE_COL else pd.NA


    # Standardisation des noms de colonnes pour l'affichage (si TMDB est la source principale des votes)
    if hasattr(C, 'VOTE_AVERAGE_COL') and C.VOTE_AVERAGE_COL in final_recs_df.columns and 'tmdb_vote_average' not in final_recs_df.columns:
        final_recs_df = final_recs_df.rename(columns={C.VOTE_AVERAGE_COL: 'tmdb_vote_average'})
    elif 'vote_average' in final_recs_df.columns and 'tmdb_vote_average' not in final_recs_df.columns:
        final_recs_df = final_recs_df.rename(columns={'vote_average': 'tmdb_vote_average'})

    if hasattr(C, 'VOTE_COUNT_COL') and C.VOTE_COUNT_COL in final_recs_df.columns and 'tmdb_vote_count' not in final_recs_df.columns:
        final_recs_df = final_recs_df.rename(columns={C.VOTE_COUNT_COL: 'tmdb_vote_count'})
    elif 'vote_count' in final_recs_df.columns and 'tmdb_vote_count' not in final_recs_df.columns:
        final_recs_df = final_recs_df.rename(columns={'vote_count': 'tmdb_vote_count'})
        
    print(f"  Top {len(final_recs_df)} recommandations finales générées avec explications (optimisé).")
    return final_recs_df

if __name__ == '__main__':
    if not df_ratings_global.empty and not df_items_global.empty:
        test_user_id = 1
        available_models = [f for f in os.listdir(MODELS_DIR) if f.endswith('.p')]
        if not available_models:
            print(f"Aucun modèle trouvé dans {MODELS_DIR}.")
        else:
            print(f"Modèles disponibles pour test: {available_models}")
            svd_model_file = next((m for m in available_models if 'svd' in m.lower()), None)
            if svd_model_file:
                print(f"\n--- Test avec SVD ({svd_model_file}) ---")
                recs_svd = get_top_n_recommendations(test_user_id, svd_model_file, n=5)
                if not recs_svd.empty: print(recs_svd[[C.ITEM_ID_COL, C.LABEL_COL, 'estimated_score', 'explanation']].head())
                else: print("Aucune recommandation SVD générée.")
    else:
        print("Données (ratings ou items) non chargées. Impossible de lancer les tests de recommender.py.")