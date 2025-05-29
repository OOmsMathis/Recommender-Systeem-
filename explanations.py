# explanations.py

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    import constants as C_module
    C = C_module.Constant()
    from models import df_items_global, df_ratings_global
    import content
except ImportError as e:
    print(f"Erreur d'importation dans explanations.py: {e}. Vérifiez les chemins.")
    df_items_global = pd.DataFrame()
    df_ratings_global = pd.DataFrame()


def get_user_profile_for_explanation(user_id, top_n_movies=3, min_rating=3.5): # min_rating ajusté pour plus de flexibilité
    """
    Récupère les N films les mieux notés par un utilisateur.
    Retourne une liste de dictionnaires [{'movieId': id, 'title': titre, 'rating': note, 'genres': genres}].
    """
    if df_ratings_global.empty or df_items_global.empty:
        return []

    try:
        if not df_ratings_global.empty and C.USER_ID_COL in df_ratings_global.columns:
             user_id_typed = type(df_ratings_global[C.USER_ID_COL].iloc[0])(user_id)
        else: user_id_typed = user_id
    except ValueError:
        user_id_typed = user_id

    user_ratings = df_ratings_global[df_ratings_global[C.USER_ID_COL] == user_id_typed]
    if user_ratings.empty:
        return []

    top_rated = user_ratings[user_ratings[C.RATING_COL] >= min_rating].sort_values(by=C.RATING_COL, ascending=False).head(top_n_movies)
    if top_rated.empty:
        return []

    profile_movies = []
    for _, row in top_rated.iterrows():
        movie_id = row[C.ITEM_ID_COL]
        profile_movies.append({
            'movieId': movie_id,
            'title': content.get_movie_title(movie_id),
            'rating': row[C.RATING_COL],
            'genres': content.get_movie_genres(movie_id)
        })
    return profile_movies


def explain_for_content_based(user_id, recommended_item_id, user_profile_movies, model_instance=None):
    """
    Explication Content-Based.
    Ex: "Genre 'Action' : comme dans 'Die Hard' (4.5/5) que vous avez aimé."
    """
    rec_item_title = content.get_movie_title(recommended_item_id) # Peut être utile si pas de profil
    if not user_profile_movies:
        return f"Style similaire aux films que vous appréciez habituellement."

    rec_item_genres_str = content.get_movie_genres(recommended_item_id)
    rec_item_genres_set = set(rec_item_genres_str.split('|')) if pd.notna(rec_item_genres_str) and rec_item_genres_str.lower() not in ["genres non spécifiés", "genres indisponibles"] else set()

    for profile_movie in user_profile_movies:
        profile_movie_genres_str = profile_movie.get('genres', "")
        profile_movie_genres_set = set(profile_movie_genres_str.split('|')) if pd.notna(profile_movie_genres_str) and profile_movie_genres_str.lower() not in ["genres non spécifiés", "genres indisponibles"] else set()
        
        common_genres = rec_item_genres_set.intersection(profile_movie_genres_set)
        
        if common_genres:
            shared_genre = list(common_genres)[0] # Prend le premier genre commun
            return f"Genre '{shared_genre}' : comme dans '{profile_movie['title']}' ({profile_movie['rating']:.1f}/5)."
            
    # Fallback si aucun genre commun n'est trouvé avec les films du profil
    return f"Dans le même style que '{user_profile_movies[0]['title']}' ({user_profile_movies[0]['rating']:.1f}/5)."


def explain_for_user_based(user_id, recommended_item_id, user_profile_movies, model_instance=None):
    """
    Explication User-Based.
    Ex: "Les fans de 'Matrix' (5.0/5) et 'Inception' (4.5/5) ont aussi aimé ce film."
    """
    rec_item_title = content.get_movie_title(recommended_item_id) # Peut être utile

    if not user_profile_movies:
        return "Populaire parmi les utilisateurs aux goûts similaires."

    if len(user_profile_movies) >= 2:
        movie1 = user_profile_movies[0]
        movie2 = user_profile_movies[1]
        return f"Les fans de '{movie1['title']}' et '{movie2['title']}' ont aussi aimé '{rec_item_title}'."
    elif len(user_profile_movies) == 1:
        movie1 = user_profile_movies[0]
        return f"Les fans de '{movie1['title']}' ({movie1['rating']:.1f}/5) ont aussi aimé '{rec_item_title}'."
    
    return f"Apprécié par des utilisateurs aux goûts similaires aux vôtres."


def explain_for_svd(user_id, recommended_item_id, user_profile_movies, model_instance, trainset):
    """
    Explication SVD.
    Ex: "Dans la lignée de 'Inception' (5.0/5) que vous avez aimé."
    """
    rec_item_title = content.get_movie_title(recommended_item_id) # Peut être utile

    if not user_profile_movies or model_instance is None or trainset is None:
        return f"Correspond à votre profil de goût général."

    try:
        # Gestion des ID internes/bruts pour l'item recommandé
        inner_rec_iid = None
        if isinstance(recommended_item_id, (int, np.integer)) and recommended_item_id < trainset.n_items: # Supposons que c'est un ID interne
             if trainset.to_raw_iid(recommended_item_id) is not None: # Vérifie si c'est un ID interne valide
                inner_rec_iid = recommended_item_id
        if inner_rec_iid is None: # Sinon, essayer de convertir l'ID brut
            inner_rec_iid = trainset.to_inner_iid(recommended_item_id)

        if inner_rec_iid >= model_instance.svd_model.qi.shape[0]: # Vérification supplémentaire de la taille
             return f"'{rec_item_title}' : un style qui pourrait vous plaire."
        rec_item_vector = model_instance.svd_model.qi[inner_rec_iid]

    except ValueError:
        return f"'{rec_item_title}' : un style qui pourrait vous plaire."
    except Exception:
        return f"'{rec_item_title}' : un style qui pourrait vous plaire."

    best_match_movie = None
    max_similarity = -1

    for profile_movie in user_profile_movies:
        try:
            profile_movie_raw_id = profile_movie['movieId']
            # Gestion des ID internes/bruts pour l'item du profil
            inner_profile_iid = None
            if isinstance(profile_movie_raw_id, (int, np.integer)) and profile_movie_raw_id < trainset.n_items:
                if trainset.to_raw_iid(profile_movie_raw_id) is not None:
                    inner_profile_iid = profile_movie_raw_id
            if inner_profile_iid is None:
                inner_profile_iid = trainset.to_inner_iid(profile_movie_raw_id)
            
            if inner_profile_iid >= model_instance.svd_model.qi.shape[0]:
                continue
            profile_item_vector = model_instance.svd_model.qi[inner_profile_iid]
            
            similarity = cosine_similarity(rec_item_vector.reshape(1, -1), profile_item_vector.reshape(1, -1))[0][0]
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_match_movie = profile_movie
        except ValueError:
            continue
        except Exception:
            continue

    if best_match_movie and max_similarity > 0.3: # Seuil un peu abaissé pour plus de chances d'avoir une explication
        return f"Dans la lignée de '{best_match_movie['title']}' ({best_match_movie['rating']:.1f}/5)."
    
    return f"'{rec_item_title}' : un style qui correspond à vos goûts."


MODEL_EXPLAINERS = {
    'content_based': explain_for_content_based,
    'user_based': explain_for_user_based,
    'svd': explain_for_svd,
}

def get_explanation_for_recommendation(user_id, recommended_item_id, model_instance, model_type_key, trainset=None):
    if model_type_key not in MODEL_EXPLAINERS:
        return "Explication non disponible."

    user_profile_movies = get_user_profile_for_explanation(user_id, top_n_movies=2, min_rating=3.5) # Récupère 2 films pour UserBased

    explainer_func = MODEL_EXPLAINERS[model_type_key]
    
    args_for_explainer = [user_id, recommended_item_id, user_profile_movies, model_instance]
    if model_type_key == 'svd':
        if trainset is None:
            return "Profil de goût général." # Explication très générique si trainset manque pour SVD
        args_for_explainer.append(trainset)
    
    return explainer_func(*args_for_explainer)


if __name__ == '__main__':
    if not df_ratings_global.empty and not df_items_global.empty:
        print("--- Test du module d'explications (Phrases Concises) ---")
        test_user_id = 1 
        
        profile = get_user_profile_for_explanation(test_user_id, top_n_movies=3, min_rating=3.5)
        print(f"\nProfil pour l'utilisateur {test_user_id} (films préférés) :")
        if profile:
            for movie in profile:
                print(f"  - {movie['title']} (ID: {movie['movieId']}, Note: {movie['rating']:.1f}, Genres: {movie['genres']})")
        else:
            print("  Profil vide ou utilisateur non trouvé.")

        # Simuler des recommandations (adaptez les IDs à vos données)
        example_rec_item_id_1 = 2 # Jumanji
        example_rec_item_id_2 = 3 # Grumpier Old Men
        example_rec_item_id_3 = 6 # Heat

        print(f"\n--- Explications pour User {test_user_id} ---")
        
        print(f"\nContent-Based pour '{content.get_movie_title(example_rec_item_id_1)}':")
        explanation_cb = explain_for_content_based(test_user_id, example_rec_item_id_1, profile)
        print(f"  -> {explanation_cb}")

        print(f"\nUser-Based pour '{content.get_movie_title(example_rec_item_id_2)}':")
        explanation_ub = explain_for_user_based(test_user_id, example_rec_item_id_2, profile)
        print(f"  -> {explanation_ub}")
        
        # Pour SVD, il faudrait un modèle et un trainset chargés
        # print(f"\nSVD pour '{content.get_movie_title(example_rec_item_id_3)}':")
        # svd_explanation = explain_for_svd(test_user_id, example_rec_item_id_3, profile, loaded_svd_model, loaded_trainset)
        # print(f"  -> {svd_explanation}")
    else:
        print("Impossible de lancer les tests de explanations.py : df_items_global ou df_ratings_global est vide.")

