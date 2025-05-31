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
    print(f"Import error in explanations.py: {e}. Check paths.") # MODIFIED
    df_items_global = pd.DataFrame()
    df_ratings_global = pd.DataFrame()


def get_user_profile_for_explanation(user_id, top_n_movies=3, min_rating=3.5):
    """
    Retrieves the N highest-rated movies by a user. # MODIFIED
    Returns a list of dictionaries [{'movieId': id, 'title': title, 'rating': rating, 'genres': genres}]. # MODIFIED
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
    Content-Based Explanation. # MODIFIED
    Ex: "Genre 'Action': as in 'Die Hard' (4.5/5) which you liked." # MODIFIED
    """
    rec_item_title = content.get_movie_title(recommended_item_id) 
    if not user_profile_movies:
        return f"Similar style to movies you usually enjoy." # MODIFIED

    rec_item_genres_str = content.get_movie_genres(recommended_item_id)
    # The check below uses French literals as content.py might still return them. The output explanation will be English.
    rec_item_genres_set = set(rec_item_genres_str.split('|')) if pd.notna(rec_item_genres_str) and rec_item_genres_str.lower() not in ["genres non spécifiés", "genres indisponibles"] else set()

    for profile_movie in user_profile_movies:
        profile_movie_genres_str = profile_movie.get('genres', "")
        # The check below uses French literals.
        profile_movie_genres_set = set(profile_movie_genres_str.split('|')) if pd.notna(profile_movie_genres_str) and profile_movie_genres_str.lower() not in ["genres non spécifiés", "genres indisponibles"] else set()
        
        common_genres = rec_item_genres_set.intersection(profile_movie_genres_set)
        
        if common_genres:
            shared_genre = list(common_genres)[0] 
            return f"Genre '{shared_genre}': as in '{profile_movie['title']}' ({profile_movie['rating']:.1f}/5)." # MODIFIED
            
    return f"In the same style as '{user_profile_movies[0]['title']}' ({user_profile_movies[0]['rating']:.1f}/5)." # MODIFIED


def explain_for_user_based(user_id, recommended_item_id, user_profile_movies, model_instance=None):
    """
    User-Based Explanation. # MODIFIED
    Ex: "Fans of 'Matrix' (5.0/5) and 'Inception' (4.5/5) also liked this movie." # MODIFIED
    """
    rec_item_title = content.get_movie_title(recommended_item_id)

    if not user_profile_movies:
        return "Popular among users with similar tastes." # MODIFIED

    if len(user_profile_movies) >= 2:
        movie1 = user_profile_movies[0]
        movie2 = user_profile_movies[1]
        return f"Fans of '{movie1['title']}' and '{movie2['title']}' also liked '{rec_item_title}'." # MODIFIED
    elif len(user_profile_movies) == 1:
        movie1 = user_profile_movies[0]
        return f"Fans of '{movie1['title']}' ({movie1['rating']:.1f}/5) also liked '{rec_item_title}'." # MODIFIED
    
    return f"Enjoyed by users with tastes similar to yours." # MODIFIED


def explain_for_svd(user_id, recommended_item_id, user_profile_movies, model_instance, trainset):
    """
    SVD Explanation. # MODIFIED
    Ex: "Along the lines of 'Inception' (5.0/5) which you liked." # MODIFIED
    """
    rec_item_title = content.get_movie_title(recommended_item_id)

    if not user_profile_movies or model_instance is None or trainset is None:
        return f"Matches your general taste profile." # MODIFIED

    try:
        inner_rec_iid = None
        if isinstance(recommended_item_id, (int, np.integer)) and recommended_item_id < trainset.n_items: 
             if trainset.to_raw_iid(recommended_item_id) is not None: 
                inner_rec_iid = recommended_item_id
        if inner_rec_iid is None: 
            inner_rec_iid = trainset.to_inner_iid(recommended_item_id)

        if inner_rec_iid >= model_instance.svd_model.qi.shape[0]: 
             return f"'{rec_item_title}': a style you might like." # MODIFIED
        rec_item_vector = model_instance.svd_model.qi[inner_rec_iid]

    except ValueError:
        return f"'{rec_item_title}': a style you might like." # MODIFIED
    except Exception:
        return f"'{rec_item_title}': a style you might like." # MODIFIED

    best_match_movie = None
    max_similarity = -1

    for profile_movie in user_profile_movies:
        try:
            profile_movie_raw_id = profile_movie['movieId']
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

    if best_match_movie and max_similarity > 0.3: 
        return f"Along the lines of '{best_match_movie['title']}' ({best_match_movie['rating']:.1f}/5)." # MODIFIED
    
    return f"'{rec_item_title}': a style that matches your tastes." # MODIFIED


MODEL_EXPLAINERS = {
    'content_based': explain_for_content_based,
    'user_based': explain_for_user_based,
    'svd': explain_for_svd,
}

def get_explanation_for_recommendation(user_id, recommended_item_id, model_instance, model_type_key, trainset=None):
    if model_type_key not in MODEL_EXPLAINERS:
        return "Explanation not available." # MODIFIED

    user_profile_movies = get_user_profile_for_explanation(user_id, top_n_movies=2, min_rating=3.5) 

    explainer_func = MODEL_EXPLAINERS[model_type_key]
    
    args_for_explainer = [user_id, recommended_item_id, user_profile_movies, model_instance]
    if model_type_key == 'svd':
        if trainset is None:
            return "General taste profile." # MODIFIED 
        args_for_explainer.append(trainset)
    
    return explainer_func(*args_for_explainer)


if __name__ == '__main__':
    if not df_ratings_global.empty and not df_items_global.empty:
        print("--- Testing explanations module (Concise Phrases) ---") # MODIFIED
        test_user_id = 1 
        
        profile = get_user_profile_for_explanation(test_user_id, top_n_movies=3, min_rating=3.5)
        print(f"\nProfile for user {test_user_id} (favorite movies):") # MODIFIED
        if profile:
            for movie in profile:
                print(f"  - {movie['title']} (ID: {movie['movieId']}, Rating: {movie['rating']:.1f}, Genres: {movie['genres']})") # MODIFIED
        else:
            print("  Empty profile or user not found.") # MODIFIED

        example_rec_item_id_1 = 2 
        example_rec_item_id_2 = 3 
        example_rec_item_id_3 = 6 

        print(f"\n--- Explanations for User {test_user_id} ---") # MODIFIED
        
        print(f"\nContent-Based for '{content.get_movie_title(example_rec_item_id_1)}':") # MODIFIED
        explanation_cb = explain_for_content_based(test_user_id, example_rec_item_id_1, profile)
        print(f"  -> {explanation_cb}")

        print(f"\nUser-Based for '{content.get_movie_title(example_rec_item_id_2)}':") # MODIFIED
        explanation_ub = explain_for_user_based(test_user_id, example_rec_item_id_2, profile)
        print(f"  -> {explanation_ub}")
        
        # print(f"\nSVD for '{content.get_movie_title(example_rec_item_id_3)}':")
        # svd_explanation = explain_for_svd(test_user_id, example_rec_item_id_3, profile, loaded_svd_model, loaded_trainset)
        # print(f"  -> {svd_explanation}")
    else:
        print("Cannot run tests for explanations.py: df_items_global or df_ratings_global is empty.") # MODIFIED