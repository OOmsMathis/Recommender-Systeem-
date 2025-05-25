import pandas as pd
from pathlib import Path
from surprise import dump, Dataset, Reader
import pickle

import constants as C
import content # Pour obtenir la liste de tous les movie_ids et les données des films les plus populaires

# --- Fonctions Utilitaires pour le Chargement (inchangées) ---

# Dans recommender.py
def load_model_and_trainset(model_filename, trainset_filename=None):
    """
    Charge un algorithme Surprise (modèle) et optionnellement son trainset associé.
    Les modèles sont chargés avec surprise.dump.load().
    Les trainsets sont chargés avec pickle.load() (car ils ont probablement été sauvegardés avec pickle.dump).
    """
    model_path = C.Constant.MODELS_STORAGE_PATH / model_filename
    algo = None
    trainset = None

    try:
        # Charger le modèle (algorithme)
        _predictions, loaded_algo = dump.load(str(model_path)) # dump.load de Surprise pour l'algo
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
        if trainset_path.exists(): # Vérifier si le fichier existe avant de tenter de le charger
            try:
                # Utiliser pickle.load() pour le trainset
                with open(str(trainset_path), 'rb') as f_pickle:
                    loaded_trainset = pickle.load(f_pickle)
                trainset = loaded_trainset
                print(f"Trainset {trainset_filename} chargé avec succès (via pickle).")
            except FileNotFoundError: # Devrait être couvert par .exists() mais par sécurité
                print(f"AVERTISSEMENT : Fichier trainset {trainset_path} non trouvé (vérification post-existence).")
            except Exception as e:
                print(f"AVERTISSEMENT : Erreur lors du chargement du trainset {trainset_path} avec pickle: {e}.")
        else:
            print(f"AVERTISSEMENT : Fichier trainset {trainset_path} non trouvé (vérification pré-existence).")
            
    # Si le trainset a été chargé avec l'algo (ancienne méthode de dump non utilisée ici pour le trainset)
    # On peut tenter de récupérer le trainset interne à l'algo si le chargement externe a échoué
    if algo and not trainset and hasattr(algo, 'trainset'):
        print(f"Utilisation du trainset interne au modèle {model_filename} car le chargement externe a échoué ou le fichier n'existe pas.")
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
        # S'assurer que user_id est du type attendu par to_inner_uid (souvent int ou str)
        inner_user_id = trainset.to_inner_uid(user_id)
        for inner_item_id, _rating in trainset.ur[inner_user_id]:
            rated_movie_ids.add(trainset.to_raw_iid(inner_item_id))
    except ValueError: 
        print(f"Utilisateur {user_id} (type: {type(user_id)}) non trouvé dans le trainset fourni.")
    except Exception as e:
        print(f"Erreur inattendue dans get_movies_rated_by_user pour user_id {user_id}: {e}")
    return rated_movie_ids

# --- Fonctions de Recommandation Principales (inchangées) ---

def get_popular_movies_recommendations(n=10, all_movie_features_df=None, ratings_df=None):
    """
    Recommande les films les plus populaires.
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

    movie_stats = ratings_df.groupby(C.Constant.MOVIE_ID_COL).agg(
        num_ratings=(C.Constant.RATING_COL, 'count'),
        avg_rating=(C.Constant.RATING_COL, 'mean')
    ).reset_index()

    min_ratings_threshold = 5 
    popular_movies = movie_stats[movie_stats['num_ratings'] >= min_ratings_threshold]
    popular_movies = popular_movies.sort_values(by=['num_ratings', 'avg_rating'], ascending=[False, False])
    
    recommended_ids = popular_movies[C.Constant.MOVIE_ID_COL].head(n).tolist()
    
    if all_movie_features_df is not None and not all_movie_features_df.empty:
        titles = [content.get_movie_title(mid, all_movie_features_df) for mid in recommended_ids]
        return list(zip(recommended_ids, titles))
        
    return recommended_ids


def generate_recommendations_for_user(
    user_id, # ID numérique
    n=10,
    all_movie_ids=None, 
    user_name_for_profile=None, # Nom du profil (ex: "alice", "testuser") pour trouver le fichier modèle
    model_config_name_suffix=None, # Suffixe du modèle (ex: "svd_implicit")
    ratings_df_path_for_general_trainset=None 
):
    """
    Génère des recommandations pour un utilisateur donné.
    """
    algo = None
    trainset_associated = None

    if all_movie_ids is None:
        print("ERREUR: La liste de tous les movie_ids est nécessaire.")
        return []

    if user_name_for_profile and model_config_name_suffix:
        # Profil personnalisé
        model_filename = C.Constant.MODEL_STORAGE_FILE_TEMPLATE_NAMED.format(
            user_name=user_name_for_profile, model_name=model_config_name_suffix
        )
        trainset_filename = C.Constant.TRAINSET_STORAGE_FILE_TEMPLATE_NAMED.format(
            user_name=user_name_for_profile
        )
        print(f"Tentative de chargement du modèle personnalisé: {model_filename} et trainset: {trainset_filename}")
        algo, trainset_associated = load_model_and_trainset(model_filename, trainset_filename)
    else:
        # Modèle général
        if not hasattr(C.Constant, 'GENERAL_MODEL_NAME'):
            print("ERREUR: C.Constant.GENERAL_MODEL_NAME non défini.")
            return get_popular_movies_recommendations(n=n) 

        general_model_filename = C.Constant.GENERAL_MODEL_NAME
        general_trainset_filename = "general_trainset.pkl" # Nom conventionnel
        
        print(f"Tentative de chargement du modèle général: {general_model_filename} et trainset: {general_trainset_filename}")
        algo, trainset_associated = load_model_and_trainset(general_model_filename, general_trainset_filename)

        if algo and not trainset_associated:
            print("Construction du trainset général à la volée...")
            if ratings_df_path_for_general_trainset is None:
                ratings_df_path_for_general_trainset = C.Constant.EVIDENCE_PATH / C.Constant.RATINGS_FILENAME
            try:
                temp_ratings_df = pd.read_csv(ratings_df_path_for_general_trainset)
                reader = Reader(rating_scale=(0.5, 5.0)) 
                data = Dataset.load_from_df(temp_ratings_df[[C.Constant.USER_ID_COL, C.Constant.MOVIE_ID_COL, C.Constant.RATING_COL]], reader)
                trainset_associated = data.build_full_trainset()
                print("Trainset général construit.")
            except Exception as e:
                print(f"Erreur construction trainset général à la volée: {e}")


    if not algo:
        print(f"Aucun modèle chargé pour l'utilisateur {user_id if not user_name_for_profile else user_name_for_profile}. Recommandations populaires.")
        ratings_df_for_popular = None
        if ratings_df_path_for_general_trainset and Path(ratings_df_path_for_general_trainset).exists():
             ratings_df_for_popular = pd.read_csv(ratings_df_path_for_general_trainset)
        return get_popular_movies_recommendations(n=n, ratings_df=ratings_df_for_popular)

    # Utiliser user_id (l'ID numérique) pour les opérations internes au trainset
    movies_to_exclude = get_movies_rated_by_user(user_id, trainset_associated if trainset_associated else algo.trainset)
    
    predictions = []
    for movie_id_to_predict in all_movie_ids:
        if movie_id_to_predict not in movies_to_exclude:
            prediction = algo.predict(uid=user_id, iid=movie_id_to_predict) # uid est l'ID numérique
            predictions.append((movie_id_to_predict, prediction.est))

    predictions.sort(key=lambda x: x[1], reverse=True)
    recommended_movie_ids = [movie_id for movie_id, score in predictions[:n]]
    
    return recommended_movie_ids


# --- Section de Test (if __name__ == '__main__') ---
if __name__ == '__main__':
    print("--- Test du Module Recommender ---")

    # --- CONFIGURATION POUR LES TESTS DE CE SCRIPT ---
    # Modifiez ces valeurs pour tester différents scénarios
    TEST_USER_PROFILE_NAME = "testuser"  # Le nom utilisé pour les fichiers (ex: "testuser", "alice")
    TEST_USER_PROFILE_ID = -1          # L'ID numérique correspondant (ex: -1)
    TEST_MODEL_CONFIG_SUFFIX = "svd_implicit" # Le suffixe du modèle (ex: "svd_implicit")
    
    TEST_MOVIELENS_USER_ID = 1         # Un ID utilisateur MovieLens pour tester le modèle général
    NUM_RECS_TO_TEST = 5
    # --- FIN DE LA CONFIGURATION DES TESTS ---

    all_movie_features_df = content.get_all_movie_features()
    if all_movie_features_df.empty:
        print("ERREUR CRITIQUE: Impossible de charger les features des films via content.py. Tests interrompus.")
        exit()
    all_movie_ids_list = all_movie_features_df.index.tolist()
    
    if not C.Constant.MODELS_STORAGE_PATH.exists():
        print(f"ERREUR: Le dossier des modèles {C.Constant.MODELS_STORAGE_PATH} n'existe pas.")
        exit()

    # Test 1: Recommandations populaires
    print("\n--- Test 1: Recommandations Populaires ---")
    ratings_csv_path = C.Constant.EVIDENCE_PATH / C.Constant.RATINGS_FILENAME
    ratings_df_for_test_popular = None
    if ratings_csv_path.exists():
        ratings_df_for_test_popular = pd.read_csv(ratings_csv_path)
        popular_recs = get_popular_movies_recommendations(n=NUM_RECS_TO_TEST, all_movie_features_df=all_movie_features_df, ratings_df=ratings_df_for_test_popular)
        if popular_recs:
            print(f"Top {NUM_RECS_TO_TEST} films populaires :")
            for mid, title in popular_recs: print(f"  ID: {mid}, Titre: {title}")
        else: print("Impossible de générer des recommandations populaires.")
    else:
        print(f"Fichier {ratings_csv_path} non trouvé pour les recommandations populaires.")

    # Test 2: Recommandations pour le profil personnalisé configuré ci-dessus
    print(f"\n--- Test 2: Recommandations pour Profil Personnalisé: {TEST_USER_PROFILE_NAME} (ID: {TEST_USER_PROFILE_ID}) ---")
    expected_model_file = C.Constant.MODELS_STORAGE_PATH / C.Constant.MODEL_STORAGE_FILE_TEMPLATE_NAMED.format(
        user_name=TEST_USER_PROFILE_NAME, model_name=TEST_MODEL_CONFIG_SUFFIX
    )
    if not expected_model_file.exists():
        print(f"AVERTISSEMENT: Modèle personnalisé {expected_model_file} non trouvé. Veuillez le générer avec recommender_building.py.")
    else:
        personalized_recs = generate_recommendations_for_user(
            user_id=TEST_USER_PROFILE_ID, # L'ID numérique
            n=NUM_RECS_TO_TEST,
            all_movie_ids=all_movie_ids_list,
            user_name_for_profile=TEST_USER_PROFILE_NAME, # Le nom pour trouver le fichier
            model_config_name_suffix=TEST_MODEL_CONFIG_SUFFIX
        )
        if personalized_recs:
            print(f"Top {NUM_RECS_TO_TEST} recommandations pour {TEST_USER_PROFILE_NAME}:")
            for movie_id_rec in personalized_recs:
                title = content.get_movie_title(movie_id_rec, all_movie_features_df)
                print(f"  ID: {movie_id_rec}, Titre: {title}")
        else:
            print(f"Aucune recommandation personnalisée générée pour {TEST_USER_PROFILE_NAME}.")

    # Test 3: Recommandations pour un utilisateur MovieLens standard avec le modèle général
    print(f"\n--- Test 3: Recommandations pour Utilisateur MovieLens ID: {TEST_MOVIELENS_USER_ID} (Modèle Général) ---")
    general_model_path_check = C.Constant.MODELS_STORAGE_PATH / C.Constant.GENERAL_MODEL_NAME
    general_trainset_path_check = C.Constant.MODELS_STORAGE_PATH / "general_trainset.pkl"

    if not hasattr(C.Constant, 'GENERAL_MODEL_NAME') or not general_model_path_check.exists() or not general_trainset_path_check.exists():
        print(f"AVERTISSEMENT: Modèle général ({C.Constant.GENERAL_MODEL_NAME}) ou trainset ({general_trainset_path_check.name}) non trouvé(s).")
        print(f"Veuillez les générer et les placer dans {C.Constant.MODELS_STORAGE_PATH}.")
    else:
        general_user_recs = generate_recommendations_for_user(
            user_id=TEST_MOVIELENS_USER_ID, # L'ID numérique
            n=NUM_RECS_TO_TEST,
            all_movie_ids=all_movie_ids_list,
            # user_name_for_profile et model_config_name_suffix sont None pour utiliser le modèle général
            ratings_df_path_for_general_trainset=ratings_csv_path 
        )
        if general_user_recs:
            print(f"Top {NUM_RECS_TO_TEST} recommandations pour l'utilisateur MovieLens {TEST_MOVIELENS_USER_ID}:")
            for movie_id_rec in general_user_recs:
                title = content.get_movie_title(movie_id_rec, all_movie_features_df)
                print(f"  ID: {movie_id_rec}, Titre: {title}")
        else:
            print(f"Aucune recommandation générée pour l'utilisateur MovieLens {TEST_MOVIELENS_USER_ID}.")
    
    print("\n--- Fin des Tests du Module Recommender ---")
