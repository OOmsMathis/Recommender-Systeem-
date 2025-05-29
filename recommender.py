# recommender.py (Mis à Jour)

import os
import pandas as pd
import re
from surprise import dump, Dataset, Reader # Ajout de Dataset, Reader

import constants as C_module
C = C_module.Constant()
import content # Notre module content.py

# Importation du nouveau module d'explications
import explanations

# Accès à df_ratings_global (pour get_movies_watched_by_user et pour construire le trainset pour SVD)
try:
    from models import df_ratings_global, df_items_global # df_items_global pour le trainset aussi
    if df_ratings_global.empty or df_items_global.empty:
        print("recommender.py: AVERTISSEMENT - df_ratings_global ou df_items_global importé de models.py est vide.")
        # Tentative de chargement direct si vide
        from loaders import load_ratings, load_items
        if df_ratings_global.empty:
            df_ratings_global_direct = load_ratings()
            if not df_ratings_global_direct.empty: df_ratings_global = df_ratings_global_direct
        if df_items_global.empty: # Nécessaire pour construire le trainset
            df_items_global_direct = load_items()
            if not df_items_global_direct.empty: df_items_global = df_items_global_direct
        
        if df_ratings_global.empty or df_items_global.empty:
             print("recommender.py: ERREUR CRITIQUE - df_ratings_global ou df_items_global toujours vide après tentative de chargement direct.")

except ImportError:
    try:
        from loaders import load_ratings, load_items
        df_ratings_global = load_ratings()
        df_items_global = load_items() # Nécessaire pour construire le trainset
        if df_ratings_global.empty or df_items_global.empty:
            raise Exception("df_ratings_global ou df_items_global vide après chargement direct.")
    except Exception as e:
        print(f"recommender.py: ERREUR FATALE - Échec chargement df_ratings_global/df_items_global: {e}")
        df_ratings_global = pd.DataFrame()
        df_items_global = pd.DataFrame()


MODELS_DIR = str(C.DATA_PATH / 'recs')
_loaded_models_cache = {}
_loaded_trainset_cache = None # Cache pour le trainset complet

def get_full_trainset():
    """
    Charge et retourne le trainset complet de Surprise.
    Utilisé principalement pour les explications SVD.
    """
    global _loaded_trainset_cache
    if _loaded_trainset_cache is not None:
        return _loaded_trainset_cache

    if df_ratings_global.empty:
        print("Recommender (get_full_trainset): df_ratings_global est vide. Impossible de construire le trainset.")
        return None
    try:
        reader = Reader(rating_scale=C.RATINGS_SCALE)
        # S'assurer que les colonnes USER_ID_COL, ITEM_ID_COL, RATING_COL existent
        cols_for_surprise = [C.USER_ID_COL, C.ITEM_ID_COL, C.RATING_COL]
        if not all(col in df_ratings_global.columns for col in cols_for_surprise):
            print(f"Recommender (get_full_trainset): Colonnes manquantes dans df_ratings_global pour Surprise: {cols_for_surprise}")
            return None
            
        data = Dataset.load_from_df(df_ratings_global[cols_for_surprise], reader)
        _loaded_trainset_cache = data.build_full_trainset()
        print("Recommender: Trainset Surprise complet construit et mis en cache.")
        return _loaded_trainset_cache
    except Exception as e:
        print(f"Recommender: Erreur lors de la construction du trainset Surprise complet: {e}")
        return None


def load_model(model_filename):
    if model_filename in _loaded_models_cache:
        return _loaded_models_cache[model_filename]
    
    file_path = os.path.join(MODELS_DIR, model_filename)
    if not os.path.exists(file_path):
        print(f"ERREUR: Modèle '{file_path}' non trouvé.")
        return None
    try:
        # dump.load retourne une liste de prédictions (si sauvegardées) ET l'algorithme.
        # On ne s'intéresse qu'à l'algorithme ici.
        _, loaded_algo = dump.load(file_path)
        _loaded_models_cache[model_filename] = loaded_algo
        print(f"Modèle '{model_filename}' chargé.")
        return loaded_algo
    except Exception as e:
        print(f"ERREUR chargement '{model_filename}': {e}")
        if model_filename in _loaded_models_cache:
            del _loaded_models_cache[model_filename] # Retirer du cache en cas d'erreur
        return None

def get_movies_watched_by_user(user_id):
    if df_ratings_global.empty or C.USER_ID_COL not in df_ratings_global.columns or C.ITEM_ID_COL not in df_ratings_global.columns:
        return set()
    try:
        # Convertir user_id au type de la colonne USER_ID_COL pour une comparaison correcte
        user_id_typed = type(df_ratings_global[C.USER_ID_COL].iloc[0])(user_id)
    except (ValueError, IndexError): # IndexError si df_ratings_global est vide après tout
        user_id_typed = user_id # Fallback

    return set(df_ratings_global[df_ratings_global[C.USER_ID_COL] == user_id_typed][C.ITEM_ID_COL])


def get_top_n_recommendations(user_id, model_filename, n=15, exclude_watched=True,
                              filter_genre=None, filter_year_range=None):
    print(f"\nRecos user '{user_id}', modèle '{model_filename}', top {n}, genre: {filter_genre}, années: {filter_year_range}...")
    
    model = load_model(model_filename)
    if model is None:
        return pd.DataFrame()

    # Déterminer le type de modèle pour l'explication
    model_type_key = None
    if 'svd' in model_filename.lower():
        model_type_key = 'svd'
    elif 'user_based' in model_filename.lower():
        model_type_key = 'user_based'
    elif 'content_based' in model_filename.lower():
        model_type_key = 'content_based'

    # Charger le trainset complet si nécessaire (pour SVD principalement)
    trainset_for_explanation = None
    if model_type_key == 'svd':
        trainset_for_explanation = get_full_trainset()
        if trainset_for_explanation is None and model_type_key == 'svd':
            print("AVERTISSEMENT: Impossible de charger le trainset pour les explications SVD. Les explications SVD seront génériques.")
            # On pourrait décider de ne pas fournir d'explication SVD ou une explication par défaut.


    # Utiliser df_movies_indexed de content.py qui devrait avoir toutes les colonnes nécessaires
    # Assurez-vous que content.df_movies_indexed est bien peuplé.
    # Si df_items_global est utilisé ici, il doit être le même que celui utilisé par content.py
    all_movie_details_df = content.df_movies_indexed.reset_index() if not content.df_movies_indexed.empty else df_items_global.copy()

    if all_movie_details_df.empty or C.ITEM_ID_COL not in all_movie_details_df.columns:
        print("  ERREUR: Détails films (all_movie_details_df) vides ou colonne ITEM_ID_COL manquante.")
        return pd.DataFrame()

    items_to_consider = all_movie_details_df.copy()
    
    movies_to_predict_for_ids = items_to_consider[C.ITEM_ID_COL].unique().tolist() # Utiliser unique() pour éviter doublons
    
    if exclude_watched:
        movies_watched = get_movies_watched_by_user(user_id)
        movies_to_predict_for_ids = [mid for mid in movies_to_predict_for_ids if mid not in movies_watched]
    
    # Appliquer les filtres de genre et d'année AVANT la prédiction pour optimiser
    items_to_predict_df = items_to_consider[items_to_consider[C.ITEM_ID_COL].isin(movies_to_predict_for_ids)]

    if filter_genre and hasattr(C, 'GENRES_COL') and C.GENRES_COL in items_to_predict_df.columns:
        # Utiliser re.escape pour les caractères spéciaux dans les noms de genre
        try:
            items_to_predict_df = items_to_predict_df[items_to_predict_df[C.GENRES_COL].str.contains(re.escape(filter_genre), case=False, na=False, regex=True)]
        except Exception as e_genre_filter:
            print(f"  Erreur filtre genre: {e_genre_filter}")


    if filter_year_range and hasattr(C, 'RELEASE_YEAR_COL') and C.RELEASE_YEAR_COL in items_to_predict_df.columns:
        year_min, year_max = filter_year_range
        # S'assurer que la colonne est numérique pour la comparaison
        items_to_predict_df[C.RELEASE_YEAR_COL] = pd.to_numeric(items_to_predict_df[C.RELEASE_YEAR_COL], errors='coerce').fillna(0)
        items_to_predict_df = items_to_predict_df[
            (items_to_predict_df[C.RELEASE_YEAR_COL] >= year_min) & 
            (items_to_predict_df[C.RELEASE_YEAR_COL] <= year_max)
        ]
    
    if items_to_predict_df.empty:
        print(f"  Aucun film ne correspond aux filtres pour la prédiction.")
        return pd.DataFrame()

    final_movies_to_predict_ids = items_to_predict_df[C.ITEM_ID_COL].tolist()
    print(f"  Prédiction pour {len(final_movies_to_predict_ids)} films après filtrage...")
    
    predictions_list = []
    for movie_id_to_predict in final_movies_to_predict_ids:
        try:
            # L'ID utilisateur pour model.predict doit être celui que le modèle connaît (brut ou interne)
            # Surprise s'attend généralement à des ID bruts pour uid et iid dans predict()
            pred_obj = model.predict(uid=user_id, iid=movie_id_to_predict)
            
            explanation_text = "Explication non disponible."
            if model_type_key:
                # Pour SVD, le trainset est passé directement.
                # Pour les autres, model.trainset devrait être accessible si le modèle a été fitté.
                current_trainset = trainset_for_explanation if model_type_key == 'svd' else getattr(model, 'trainset', None)
                if model_type_key == 'svd' and current_trainset is None:
                     explanation_text = "Explication SVD non générée (trainset manquant)."
                elif current_trainset is None and model_type_key != 'content_based': # CB n'utilise pas le trainset pour l'explication ici
                     explanation_text = f"Explication {model_type_key} non générée (trainset du modèle manquant)."
                else:
                    explanation_text = explanations.get_explanation_for_recommendation(
                        user_id, movie_id_to_predict, model, model_type_key, current_trainset
                    )

            predictions_list.append({
                C.ITEM_ID_COL: pred_obj.iid,
                'estimated_score': pred_obj.est,
                'explanation': explanation_text # Ajout de l'explication
            })
        except Exception as e_pred:
            # print(f"  Erreur de prédiction pour movie_id {movie_id_to_predict}: {e_pred}")
            continue # Passer au suivant en cas d'erreur de prédiction pour un item
    
    if not predictions_list:
        print("  Aucune prédiction générée.")
        return pd.DataFrame()

    recs_df = pd.DataFrame(predictions_list)
    recs_df = recs_df.sort_values(by='estimated_score', ascending=False).head(n)
    
    if recs_df.empty:
        print("  Aucun film dans le top N après tri.")
        return pd.DataFrame()

    # Enrichir avec les détails de content.py (qui inclut maintenant les notes TMDB)
    recommended_movie_ids = recs_df[C.ITEM_ID_COL].tolist()
    
    # Utiliser la fonction de content.py pour obtenir les détails
    # S'assurer que get_movie_details_list retourne bien les colonnes attendues
    # (LABEL_COL, GENRES_COL, RELEASE_YEAR_COL, VOTE_AVERAGE_COL, VOTE_COUNT_COL, etc.)
    movie_details_list_of_dicts = content.get_movie_details_list(recommended_movie_ids)
    details_df = pd.DataFrame(movie_details_list_of_dicts)

    if details_df.empty or C.ITEM_ID_COL not in details_df.columns:
        print("  AVERTISSEMENT: Aucun détail de film de content.py pour les recos.")
        final_recs_df = recs_df # Retourner au moins les IDs, scores et explications
        # Ajouter des colonnes placeholder si elles manquent pour l'affichage
        cols_to_ensure = [C.LABEL_COL, C.GENRES_COL, C.RELEASE_YEAR_COL, C.VOTE_AVERAGE_COL, C.VOTE_COUNT_COL]
        for col_name in cols_to_ensure:
            if col_name not in final_recs_df.columns:
                final_recs_df[col_name] = pd.NA # ou une valeur par défaut appropriée
    else:
        final_recs_df = pd.merge(recs_df, details_df, on=C.ITEM_ID_COL, how='left')
        
    # S'assurer que les colonnes de notes TMDB sont bien nommées pour l'affichage si elles existent
    # et qu'elles ne sont pas écrasées par le merge si les noms sont identiques.
    # Le DataFrame `details_df` devrait déjà avoir les bons noms grâce à content.py
    # Si C.VOTE_AVERAGE_COL est 'vote_average', il sera déjà correct.
    # Renommer uniquement si nécessaire pour éviter confusion ou standardiser.
    if hasattr(C, 'VOTE_AVERAGE_COL') and C.VOTE_AVERAGE_COL in final_recs_df.columns and 'tmdb_vote_average' not in final_recs_df.columns:
        final_recs_df = final_recs_df.rename(columns={C.VOTE_AVERAGE_COL: 'tmdb_vote_average'})
    elif 'vote_average' in final_recs_df.columns and 'tmdb_vote_average' not in final_recs_df.columns: # Fallback si C.VOTE_AVERAGE_COL n'est pas 'vote_average'
        final_recs_df = final_recs_df.rename(columns={'vote_average': 'tmdb_vote_average'})


    if hasattr(C, 'VOTE_COUNT_COL') and C.VOTE_COUNT_COL in final_recs_df.columns and 'tmdb_vote_count' not in final_recs_df.columns:
        final_recs_df = final_recs_df.rename(columns={C.VOTE_COUNT_COL: 'tmdb_vote_count'})
    elif 'vote_count' in final_recs_df.columns and 'tmdb_vote_count' not in final_recs_df.columns:
        final_recs_df = final_recs_df.rename(columns={'vote_count': 'tmdb_vote_count'})
        
    print(f"  Top {len(final_recs_df)} recommandations finales générées avec explications.")
    return final_recs_df


if __name__ == '__main__':
    # Section de test
    # Assurez-vous que des modèles sont entraînés et présents dans MODELS_DIR
    # et que les données (ratings, items) sont chargées.
    
    # Charger manuellement df_ratings_global et df_items_global si ce n'est pas fait au niveau du module
    if df_ratings_global.empty or df_items_global.empty:
        print("Chargement des données pour les tests de recommender.py...")
        from loaders import load_ratings, load_items
        df_ratings_global = load_ratings()
        df_items_global = load_items()
        # Il faut aussi que content.py ait accès à df_items_global
        content.df_items_global = df_items_global 
        content.df_movies_indexed = df_items_global.set_index(C.ITEM_ID_COL)


    if not df_ratings_global.empty and not df_items_global.empty:
        test_user_id = 1  # Un ID utilisateur de votre jeu de données
        
        # Lister les modèles disponibles pour le test
        available_models = [f for f in os.listdir(MODELS_DIR) if f.endswith('.p')]
        if not available_models:
            print(f"Aucun modèle trouvé dans {MODELS_DIR}. Veuillez entraîner des modèles d'abord.")
        else:
            print(f"Modèles disponibles pour test: {available_models}")
            
            # Test avec le premier modèle SVD trouvé (ou un nom spécifique)
            svd_model_file = next((m for m in available_models if 'svd' in m.lower()), None)
            if svd_model_file:
                print(f"\n--- Test avec SVD ({svd_model_file}) ---")
                recs_svd = get_top_n_recommendations(test_user_id, svd_model_file, n=5)
                if not recs_svd.empty:
                    print(recs_svd[[C.ITEM_ID_COL, C.LABEL_COL, 'estimated_score', 'explanation']].head())
                else:
                    print("Aucune recommandation SVD générée.")
            else:
                print("Aucun modèle SVD trouvé pour le test.")

            # Test avec le premier modèle User-Based trouvé
            user_based_model_file = next((m for m in available_models if 'user_based' in m.lower()), None)
            if user_based_model_file:
                print(f"\n--- Test avec User-Based ({user_based_model_file}) ---")
                recs_ub = get_top_n_recommendations(test_user_id, user_based_model_file, n=5)
                if not recs_ub.empty:
                    print(recs_ub[[C.ITEM_ID_COL, C.LABEL_COL, 'estimated_score', 'explanation']].head())
                else:
                    print("Aucune recommandation User-Based générée.")
            else:
                print("Aucun modèle User-Based trouvé pour le test.")
            
            # Test avec le premier modèle Content-Based trouvé
            content_based_model_file = next((m for m in available_models if 'content_based' in m.lower()), None)
            if content_based_model_file:
                print(f"\n--- Test avec Content-Based ({content_based_model_file}) ---")
                recs_cb = get_top_n_recommendations(test_user_id, content_based_model_file, n=5)
                if not recs_cb.empty:
                    print(recs_cb[[C.ITEM_ID_COL, C.LABEL_COL, 'estimated_score', 'explanation']].head())
                else:
                    print("Aucune recommandation Content-Based générée.")
            else:
                print("Aucun modèle Content-Based trouvé pour le test.")
    else:
        print("Données (ratings ou items) non chargées. Impossible de lancer les tests de recommender.py.")

