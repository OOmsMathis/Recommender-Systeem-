import pickle # Modifié depuis surprise.dump pour cohérence avec la sauvegarde du trainset
from surprise import dump # Pour charger les modèles surprise
import streamlit as st
import numpy as np
import pandas as pd

# Vos modules existants
import constants as C
import models # Pour les types et PredictionImpossible
import loaders # Pourrait être utilisé pour charger le trainset brut si non sauvegardé avec le modèle

# Assurez-vous que ces constantes sont définies dans constants.py et pointent vers les bons fichiers/dossiers
# Exemple:
# PERSONAL_USER_ID = -1 (doit correspondre à celui dans recommender_building.py)
# MODELS_STORAGE_PATH = Path(...)
# TRAINSET_STORAGE_FILE = MODELS_STORAGE_PATH / "augmented_trainset_user{}.pkl"
# MODEL_STORAGE_FILE_TEMPLATE = MODELS_STORAGE_PATH / "user{}_{}_model.pkl"

DF_ITEMS_GLOBAL = loaders.load_items() # Pour avoir la liste de tous les movie_ids
ALL_MOVIE_IDS_RAW_GLOBAL = DF_ITEMS_GLOBAL.index.tolist()


@st.cache_resource # Cache la ressource (modèle chargé, trainset)
def load_model_and_trainset(user_id_for_model, model_type_str_simple):
    """
    Charge un modèle pré-entraîné et son trainset associé.
    model_type_str_simple: 'SVD', 'UserBased', 'ContentBased'
    """
    model_filename = C.Constant.MODEL_STORAGE_FILE_TEMPLATE.format(user_id_for_model, model_type_str_simple.lower())
    trainset_filename = C.Constant.TRAINSET_STORAGE_FILE.format(user_id_for_model)
    
    try:
        print(f"Chargement du modèle depuis : {model_filename}")
        # Utiliser surprise.dump.load pour charger le modèle
        _, model = dump.load(str(model_filename))
        
        print(f"Chargement du trainset depuis : {trainset_filename}")
        with open(trainset_filename, 'rb') as f:
            trainset = pickle.load(f)
            
        return model, trainset
    except FileNotFoundError as e:
        st.error(f"Fichier modèle ou trainset non trouvé pour user {user_id_for_model}, model {model_type_str_simple}. " \
                 f"Veuillez exécuter recommender_building.py. Détail: {e}")
        return None, None
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle/trainset: {e}")
        return None, None

def get_available_user_ids_for_app():
    """
    Retourne une liste des User IDs pour lesquels des modèles personnalisés sont disponibles.
    Pour l'instant, cela retournera principalement votre ID spécial.
    """
    # Pour ce projet, nous nous concentrons sur l'utilisateur personnalisé
    return [str(C.Constant.PERSONAL_USER_ID)] # Doit être une chaîne pour selectbox

@st.cache_data # Cache le résultat de cette fonction
def get_top_n_recommendations(_model, _trainset, user_id_raw, model_type_str, n=10):
    """
    Génère les top-N recommandations pour un utilisateur.
    _model et _trainset sont préfixés pour indiquer qu'ils sont gérés par le cache de Streamlit.
    user_id_raw est l'ID brut (celui de vos fichiers).
    """
    if _model is None or _trainset is None:
        return []

    predictions = []
    
    # Convertir l'user_id (raw) en inner_id de Surprise, s'il existe dans ce trainset
    try:
        user_inner_id = _trainset.to_inner_uid(user_id_raw) # Doit être du même type que dans le trainset (int si PERSONAL_USER_ID est int)
        items_rated_by_user_inner_ids = {item_inner_id for (item_inner_id, _) in _trainset.ur[user_inner_id]}
    except ValueError: # Utilisateur non trouvé dans ce trainset (ne devrait pas arriver si le trainset est le bon)
        st.warning(f"L'utilisateur {user_id_raw} n'a pas été trouvé dans le trainset chargé. Aucune recommandation personnelle possible.")
        return [] # Ou recommander des items populaires en fallback

    candidate_items_count = 0
    for movie_id_raw_candidate in ALL_MOVIE_IDS_RAW_GLOBAL:
        try:
            movie_inner_id_candidate = _trainset.to_inner_iid(movie_id_raw_candidate)
            if movie_inner_id_candidate in items_rated_by_user_inner_ids:
                continue # L'utilisateur a déjà noté ce film
            candidate_items_count +=1
            
            # Prédiction (les modèles AlgoBase s'attendent à des inner_ids)
            pred_obj = _model.predict(user_inner_id, movie_inner_id_candidate)
            predictions.append({'movie_id': movie_id_raw_candidate, 'score': pred_obj.est})

        except models.PredictionImpossible: # Gérer si un modèle custom lève cette exception
            pass 
        except ValueError: # movie_id_raw_candidate n'est pas dans le trainset
            # st.write(f"Film ID {movie_id_raw_candidate} non trouvé dans le trainset pour la prédiction.")
            pass
        except Exception as e:
            # print(f"Erreur de prédiction pour user {user_id_raw}, item {movie_id_raw_candidate}: {e}")
            pass
            
    # Trier les prédictions et prendre le top-N
    predictions.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"Nombre de films candidats pour les recommandations: {candidate_items_count}")
    print(f"Nombre de prédictions réussies: {len(predictions)}")
    
    # Préparer la liste finale avec les explications
    final_recommendations = []
    for pred_info in predictions[:n]:
        explanation = get_recommendation_explanation(
            user_id_raw, 
            pred_info['movie_id'], 
            model_type_str, 
            _model, # L'instance du modèle chargé
            _trainset
        )
        final_recommendations.append((pred_info['movie_id'], pred_info['score'], explanation))
        
    return final_recommendations


def get_recommendation_explanation(user_id_raw, movie_id_raw, model_type_str, model_instance, trainset):
    """
    Génère une explication pour une recommandation.
    """
    # Utiliser content.py pour obtenir les détails du film
    import content # Importation locale pour éviter les dépendances circulaires au niveau global si content.py chargeait des choses lourdes
    movie_details = content.get_movie_details(movie_id_raw)
    movie_title = movie_details['title'] if movie_details else f"Film ID {movie_id_raw}"

    # Convertir les IDs raw en inner IDs si nécessaire par le modèle pour l'explication
    try:
        user_inner_id = trainset.to_inner_uid(user_id_raw)
    except ValueError: # Utilisateur non dans le trainset (ne devrait pas arriver ici)
        user_inner_id = None 

    if model_type_str == "UserBased":
        # L'explication pour UserBased est complexe sans recalculer les voisins pour cet item.
        # Pour une démo, une explication générique peut suffire.
        return f"Recommandé car des utilisateurs ayant des profils de notation similaires au vôtre ont apprécié '{movie_title}'. Notre algorithme User-Based a identifié une forte affinité."

    elif model_type_str == "ContentBased":
        if hasattr(model_instance, 'user_profile') and hasattr(model_instance, 'content_features') and user_inner_id is not None:
            user_specific_model = model_instance.user_profile.get(user_inner_id)
            if user_specific_model and movie_id_raw in model_instance.content_features.index:
                # Expliquer en se basant sur les features du film.
                # Exemple simple:
                genres = movie_details['genres'] if movie_details else "N/A"
                # features_principales = # Pourrait être basé sur les coefs du modèle linéaire de l'utilisateur, ou les règles d'un arbre...
                # C'est la partie la plus difficile à rendre générique et pertinente.
                return f"'{movie_title}' (genres: {genres}) semble bien correspondre à vos préférences basées sur le contenu, d'après l'analyse des caractéristiques des films que vous avez déjà notés."
        return f"'{movie_title}' vous est suggéré sur la base de ses caractéristiques et de votre profil de contenu appris."

    elif model_type_str == "SVD":
        return f"Notre algorithme SVD a analysé les interactions complexes entre utilisateurs et films. '{movie_title}' a été identifié comme un film que vous pourriez apprécier en fonction des préférences et caractéristiques latentes (cachées) découvertes."
    
    return "Explication non disponible."


if __name__ == '__main__':
    # Zone de test pour recommender.py (peut être utile pendant le développement)
    print("Test du module recommender.py...")
    test_user_id = C.Constant.PERSONAL_USER_ID # Assurez-vous que c'est défini
    test_model_type = 'SVD' # Ou 'UserBased', 'ContentBased'

    print(f"Test pour l'utilisateur: {test_user_id}, Modèle: {test_model_type}")
    
    model, trainset = load_model_and_trainset(test_user_id, test_model_type)
    
    if model and trainset:
        print(f"Modèle et trainset chargés pour user {test_user_id}, type {test_model_type}.")
        recommendations = get_top_n_recommendations(model, trainset, test_user_id, test_model_type, n=5)
        if recommendations:
            print(f"\nTop 5 recommandations pour l'utilisateur {test_user_id} avec {test_model_type}:")
            for movie_id, score, explanation in recommendations:
                details = DF_ITEMS_GLOBAL.loc[movie_id]
                print(f"  - {details[C.Constant.LABEL_COL]}: Score {score:.2f} (Genres: {details[C.Constant.GENRES_COL]})")
                print(f"    Explication: {explanation}")
        else:
            print("Aucune recommandation générée.")
    else:
        print("Échec du chargement du modèle ou du trainset.")