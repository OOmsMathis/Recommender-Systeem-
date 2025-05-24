import pandas as pd
from pathlib import Path
from surprise import dump, Dataset, Reader # Importer Dataset et Reader pour un trainset général "à la volée"

import constants as C
# Supposons que content.py est au même niveau ou dans le PYTHONPATH
import content # Pour obtenir la liste de tous les movie_ids et les données des films les plus populaires

# --- Fonctions Utilitaires pour le Chargement ---

def load_model_and_trainset(model_filename, trainset_filename=None):
    """
    Charge un algorithme Surprise (modèle) et optionnellement son trainset associé.
    Les fichiers sont attendus dans C.Constant.MODELS_STORAGE_PATH.
    """
    model_path = C.Constant.MODELS_STORAGE_PATH / model_filename
    algo = None
    trainset = None

    try:
        # La fonction dump.load retourne un tuple (predictions, algo) ou (trainset, algo) ou juste algo
        # Nous avons sauvegardé 'algo' pour le modèle et 'trainset' pour le trainset séparément.
        _predictions, loaded_algo = dump.load(str(model_path)) # _predictions est souvent None si on a dumpé que l'algo
        algo = loaded_algo
        print(f"Modèle {model_filename} chargé avec succès.")
    except FileNotFoundError:
        print(f"ERREUR : Fichier modèle {model_path} non trouvé.")
        return None, None
    except Exception as e:
        print(f"ERREUR lors du chargement du modèle {model_path}: {e}")
        return None, None

    if trainset_filename:
        trainset_path = C.Constant.MODELS_STORAGE_PATH / trainset_filename
        try:
            _predictions_ts, loaded_trainset = dump.load(str(trainset_path))
            trainset = loaded_trainset
            print(f"Trainset {trainset_filename} chargé avec succès.")
        except FileNotFoundError:
            print(f"AVERTISSEMENT : Fichier trainset {trainset_path} non trouvé. Le modèle sera utilisé sans trainset spécifique.")
        except Exception as e:
            print(f"AVERTISSEMENT : Erreur lors du chargement du trainset {trainset_path}: {e}. Le modèle sera utilisé sans trainset spécifique.")
            
    # Si le trainset a été chargé avec l'algo (ancienne méthode de dump), on l'extrait
    if algo and not trainset and hasattr(algo, 'trainset'):
        trainset = algo.trainset

    return algo, trainset

def get_movies_rated_by_user(user_id, trainset):
    """
    Retourne un ensemble de movie_ids (raw ids) notés par l'utilisateur dans le trainset donné.
    """
    rated_movie_ids = set()
    if trainset is None:
        print("AVERTISSEMENT: Aucun trainset fourni pour get_movies_rated_by_user.")
        return rated_movie_ids
    try:
        inner_user_id = trainset.to_inner_uid(user_id)
        for inner_item_id, _rating in trainset.ur[inner_user_id]:
            rated_movie_ids.add(trainset.to_raw_iid(inner_item_id))
    except ValueError: # Utilisateur inconnu dans ce trainset
        print(f"Utilisateur {user_id} non trouvé dans le trainset fourni.")
    return rated_movie_ids

# --- Fonctions de Recommandation Principales ---

def get_popular_movies_recommendations(n=10, all_movie_features_df=None, ratings_df=None):
    """
    Recommande les films les plus populaires (plus grand nombre de ratings, puis meilleure moyenne).
    Nécessite le DataFrame des ratings et optionnellement celui des features pour les titres.
    """
    if ratings_df is None:
        try:
            ratings_file_path = C.Constant.EVIDENCE_PATH / C.Constant.RATINGS_FILENAME
            ratings_df = pd.read_csv(ratings_file_path)
        except Exception as e:
            print(f"Erreur lors du chargement des ratings pour les films populaires: {e}")
            return []

    if ratings_df.empty:
        return []

    # Compter le nombre de ratings et la moyenne par film
    movie_stats = ratings_df.groupby(C.Constant.MOVIE_ID_COL).agg(
        num_ratings=(C.Constant.RATING_COL, 'count'),
        avg_rating=(C.Constant.RATING_COL, 'mean')
    ).reset_index()

    # Trier par nombre de ratings (décroissant) puis par note moyenne (décroissant)
    # On peut appliquer un seuil minimum de ratings pour la popularité (ex: au moins 5 ratings)
    min_ratings_threshold = 5 
    popular_movies = movie_stats[movie_stats['num_ratings'] >= min_ratings_threshold]
    popular_movies = popular_movies.sort_values(by=['num_ratings', 'avg_rating'], ascending=[False, False])
    
    recommended_ids = popular_movies[C.Constant.MOVIE_ID_COL].head(n).tolist()
    
    # Si on a les features, on peut retourner plus d'infos (titres)
    if all_movie_features_df is not None and not all_movie_features_df.empty:
        titles = [content.get_movie_title(mid, all_movie_features_df) for mid in recommended_ids]
        return list(zip(recommended_ids, titles))
        
    return recommended_ids


def generate_recommendations_for_user(
    user_id,
    n=10,
    all_movie_ids=None, # Liste de tous les movieIds possibles
    user_name_for_profile=None, # Ex: "alice" pour charger model_alice_...
    model_config_name_suffix=None, # Ex: "svd_implicit"
    ratings_df_path=None # Chemin vers le fichier ratings.csv global (pour utilisateurs non personnalisés)
):
    """
    Génère des recommandations pour un utilisateur donné.
    Charge le modèle approprié (personnalisé ou général).
    """
    algo = None
    trainset_associated = None # Le trainset sur lequel le modèle a été entraîné

    if all_movie_ids is None:
        print("ERREUR: La liste de tous les movie_ids est nécessaire pour générer des recommandations.")
        return []

    if user_name_for_profile and model_config_name_suffix:
        # Profil personnalisé
        model_filename = C.Constant.MODEL_STORAGE_FILE_TEMPLATE_NAMED.format(
            user_name=user_name_for_profile, model_name=model_config_name_suffix
        )
        trainset_filename = C.Constant.TRAINSET_STORAGE_FILE_TEMPLATE_NAMED.format(
            user_name=user_name_for_profile
        )
        algo, trainset_associated = load_model_and_trainset(model_filename, trainset_filename)
    else:
        # Modèle général
        if not hasattr(C.Constant, 'GENERAL_MODEL_NAME'):
            print("ERREUR: C.Constant.GENERAL_MODEL_NAME non défini pour le modèle général.")
            return get_popular_movies_recommendations(n=n) # Fallback

        general_model_filename = C.Constant.GENERAL_MODEL_NAME
        # Pour le modèle général, on s'attend à un trainset général aussi.
        # Supposons qu'il s'appelle "general_trainset.pkl" ou doit être construit.
        general_trainset_filename = "general_trainset.pkl" # À créer et sauvegarder une fois
        
        algo, trainset_associated = load_model_and_trainset(general_model_filename, general_trainset_filename)

        # Si le trainset général n'a pas été chargé avec le modèle, on le construit à la volée
        # C'est moins efficace, il vaut mieux le pré-calculer et le sauvegarder.
        if algo and not trainset_associated:
            print("Construction du trainset général à la volée (il est préférable de le pré-calculer)...")
            if ratings_df_path is None:
                ratings_df_path = C.Constant.EVIDENCE_PATH / C.Constant.RATINGS_FILENAME
            try:
                temp_ratings_df = pd.read_csv(ratings_df_path)
                reader = Reader(rating_scale=(0.5, 5.0)) # ou C.Constant.RATINGS_SCALE
                data = Dataset.load_from_df(temp_ratings_df[[C.Constant.USER_ID_COL, C.Constant.MOVIE_ID_COL, C.Constant.RATING_COL]], reader)
                trainset_associated = data.build_full_trainset()
                print("Trainset général construit.")
            except Exception as e:
                print(f"Erreur lors de la construction du trainset général à la volée: {e}")


    if not algo:
        print(f"Aucun modèle n'a pu être chargé pour l'utilisateur {user_id if not user_name_for_profile else user_name_for_profile}. "
              "Retour des recommandations populaires.")
        # Charger ratings_df pour les films populaires
        ratings_df_for_popular = None
        if ratings_df_path:
             ratings_df_for_popular = pd.read_csv(ratings_df_path)
        return get_popular_movies_recommendations(n=n, ratings_df=ratings_df_for_popular)

    # Obtenir les films déjà notés par l'utilisateur DANS LE CONTEXTE DU TRAINSET DU MODÈLE
    movies_to_exclude = get_movies_rated_by_user(user_id, trainset_associated if trainset_associated else algo.trainset)
    
    # Prédire les scores pour les films non encore notés par l'utilisateur
    predictions = []
    for movie_id in all_movie_ids:
        if movie_id not in movies_to_exclude:
            # `predict` prend les IDs bruts (raw IDs)
            prediction = algo.predict(uid=user_id, iid=movie_id)
            predictions.append((movie_id, prediction.est))

    # Trier les prédictions par score estimé (décroissant)
    predictions.sort(key=lambda x: x[1], reverse=True)

    # Retourner les N meilleurs movie_ids
    recommended_movie_ids = [movie_id for movie_id, score in predictions[:n]]
    
    return recommended_movie_ids


# --- Exemple d'utilisation (pour tests) ---
if __name__ == '__main__':
    print("Test du module Recommender...")

    # 0. Prérequis : Charger la liste de tous les films
    # (dans une vraie app, all_movie_features_df serait chargé une fois)
    all_movie_features_df = content.get_all_movie_features()
    if all_movie_features_df.empty:
        print("Impossible de charger les features des films depuis content.py. Arrêt des tests.")
        exit()
    all_movie_ids_list = all_movie_features_df.index.tolist()
    
    # S'assurer que le dossier des modèles est là
    if not C.Constant.MODELS_STORAGE_PATH.exists():
        print(f"Le dossier des modèles {C.Constant.MODELS_STORAGE_PATH} n'existe pas. Créez-le et placez-y des modèles.")
        exit()

    # Test 1: Recommandations populaires (cold start)
    print("\n--- Test Recommandations Populaires ---")
    # Pour ce test, on s'assure que ratings.csv est accessible
    ratings_csv_path = C.Constant.EVIDENCE_PATH / C.Constant.RATINGS_FILENAME
    if not ratings_csv_path.exists():
        print(f"Fichier {ratings_csv_path} non trouvé pour les recommandations populaires.")
    else:
        popular_recs = get_popular_movies_recommendations(n=5, all_movie_features_df=all_movie_features_df)
        if popular_recs:
            print("Top 5 films populaires :")
            for mid, title in popular_recs:
                print(f"  ID: {mid}, Titre: {title}")
        else:
            print("Impossible de générer des recommandations populaires.")

    # Test 2: Recommandations pour un profil personnalisé (ex: "alice", ID -1)
    # Assurez-vous qu'un modèle pour "alice" (ou le nom que vous testez) existe !
    # Ex: model_alice_svd_implicit.pkl et trainset_alice.pkl dans data/small/recs/
    USER_PROFILE_NAME = "alice" # Doit correspondre à un profil créé par recommender_building.py
    USER_PROFILE_ID = -1
    MODEL_CONFIG_NAME = "svd_implicit" # Doit correspondre au suffixe utilisé lors de la création

    print(f"\n--- Test Recommandations pour Profil Personnalisé: {USER_PROFILE_NAME} (ID: {USER_PROFILE_ID}) ---")
    # Vérifier si le modèle personnalisé existe avant de tester
    expected_model_file = C.Constant.MODELS_STORAGE_PATH / C.Constant.MODEL_STORAGE_FILE_TEMPLATE_NAMED.format(
        user_name=USER_PROFILE_NAME, model_name=MODEL_CONFIG_NAME
    )
    if not expected_model_file.exists():
        print(f"Modèle personnalisé {expected_model_file} non trouvé. Veuillez le générer avec recommender_building.py.")
    else:
        personalized_recs = generate_recommendations_for_user(
            user_id=USER_PROFILE_ID,
            n=5,
            all_movie_ids=all_movie_ids_list,
            user_name_for_profile=USER_PROFILE_NAME,
            model_config_name_suffix=MODEL_CONFIG_NAME
        )
        if personalized_recs:
            print(f"Top 5 recommandations pour {USER_PROFILE_NAME}:")
            for movie_id_rec in personalized_recs:
                title = content.get_movie_title(movie_id_rec, all_movie_features_df)
                print(f"  ID: {movie_id_rec}, Titre: {title}")
        else:
            print(f"Aucune recommandation personnalisée générée pour {USER_PROFILE_NAME}.")


    # Test 3: Recommandations pour un utilisateur MovieLens standard (ID > 0) avec le modèle général
    # Assurez-vous qu'un modèle général (ex: svd_general_model.pkl et general_trainset.pkl) existe !
    # Vous devez créer et sauvegarder "general_trainset.pkl" manuellement ou via un script.
    # Contenu de general_trainset.pkl:
    #   df_ratings = pd.read_csv(C.Constant.EVIDENCE_PATH / C.Constant.RATINGS_FILENAME)
    #   reader = Reader(rating_scale=(0.5, 5.0)) # Ou (1,5) si que MovieLens original
    #   data = Dataset.load_from_df(df_ratings[['userId', 'movieId', 'rating']], reader)
    #   full_trainset = data.build_full_trainset()
    #   dump.dump(str(C.Constant.MODELS_STORAGE_PATH / "general_trainset.pkl"), trainset=full_trainset)
    #   (Entraînez et sauvegardez aussi le modèle général: svd_general_model.pkl)

    MOVIELENS_USER_ID = 1 # Un ID utilisateur de MovieLens
    print(f"\n--- Test Recommandations pour Utilisateur MovieLens ID: {MOVIELENS_USER_ID} (Modèle Général) ---")
    if not hasattr(C.Constant, 'GENERAL_MODEL_NAME') or \
       not (C.Constant.MODELS_STORAGE_PATH / C.Constant.GENERAL_MODEL_NAME).exists() or \
       not (C.Constant.MODELS_STORAGE_PATH / "general_trainset.pkl").exists():
        print(f"Modèle général ({C.Constant.GENERAL_MODEL_NAME}) ou trainset général (general_trainset.pkl) non trouvé(s).")
        print("Veuillez les générer et les placer dans ", C.Constant.MODELS_STORAGE_PATH)
    else:
        general_user_recs = generate_recommendations_for_user(
            user_id=MOVIELENS_USER_ID,
            n=5,
            all_movie_ids=all_movie_ids_list,
            ratings_df_path=C.Constant.EVIDENCE_PATH / C.Constant.RATINGS_FILENAME # Pour construire le trainset général si non chargé
        )
        if general_user_recs:
            print(f"Top 5 recommandations pour l'utilisateur MovieLens {MOVIELENS_USER_ID}:")
            for movie_id_rec in general_user_recs:
                title = content.get_movie_title(movie_id_rec, all_movie_features_df)
                print(f"  ID: {movie_id_rec}, Titre: {title}")
        else:
            print(f"Aucune recommandation générée pour l'utilisateur MovieLens {MOVIELENS_USER_ID} avec le modèle général.")