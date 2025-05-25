import pandas as pd
import time
from pathlib import Path
from surprise import Dataset, Reader, dump, SVD # SVD est un exemple
import pickle # <--- AJOUT IMPORTANT ICI

# Votre module constants
import constants as C 
# import loaders # Si vous avez un loader spécifique pour les ratings
# import models # Si vous utilisez vos propres wrappers de modèles

def calculate_implicit_ratings_from_library(library_df, current_user_id):
    """
    Calcule les ratings implicites à partir d'un DataFrame de bibliothèque utilisateur
    pour un `current_user_id` donné.
    """
    implicit_ratings = []
    # Colonnes attendues dans le fichier CSV de la bibliothèque de l'utilisateur
    # C.Constant.MOVIE_ID_COL (qui est 'movieId') doit être présent.
    required_cols = [C.Constant.MOVIE_ID_COL, 'watched_count', 'last_watched_months_ago', 'is_favorite', 'on_wishlist']
    if not all(col in library_df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in library_df.columns]
        raise ValueError(f"Colonnes manquantes dans le fichier de bibliothèque : {missing_cols}. "
                         f"Colonnes attendues (au minimum) : {required_cols}")

    for _, row in library_df.iterrows():
        rating = 2.5  # Note de base

        # Exemple de logique de conversion (à adapter / affiner)
        if row.get('watched_count', 0) > 1:
            rating += min(row['watched_count'] * 0.25, 1.0)
        if row.get('last_watched_months_ago', 99) <= 6: # Vu récemment
            rating += 0.5
        elif row.get('last_watched_months_ago', 99) > 24: # Vu il y a longtemps
            rating -= 0.5
        if row.get('is_favorite', 0) == 1:
            rating += 1.0
        if row.get('on_wishlist', 0) == 1 and row.get('is_favorite', 0) == 0:
            rating += 0.25

        # Borner le rating à l'échelle [0.5, 5.0]
        rating = max(0.5, min(5.0, rating))
        
        implicit_ratings.append({
            C.Constant.USER_ID_COL: current_user_id,
            C.Constant.MOVIE_ID_COL: int(row[C.Constant.MOVIE_ID_COL]),
            C.Constant.RATING_COL: rating,
            C.Constant.TIMESTAMP_COL: int(time.time())
        })
    return pd.DataFrame(implicit_ratings)

def append_implicit_ratings_and_build_trainset(implicit_ratings_df, movielens_ratings_df, current_user_id):
    """
    Ajoute les ratings implicites au dataset MovieLens et construit un trainset Surprise.
    """
    print(f"Nombre de ratings MovieLens originaux : {len(movielens_ratings_df)}")
    print(f"Nombre de ratings implicites ajoutés pour l'utilisateur {current_user_id} : {len(implicit_ratings_df)}")
    
    cols_to_keep = [C.Constant.USER_ID_COL, C.Constant.MOVIE_ID_COL, C.Constant.RATING_COL, C.Constant.TIMESTAMP_COL]
    
    movielens_ratings_df = movielens_ratings_df[cols_to_keep].copy()
    if not all(col in implicit_ratings_df.columns for col in cols_to_keep):
        raise ValueError(f"Les colonnes de implicit_ratings_df ne correspondent pas. Attendu: {cols_to_keep}, Obtenu: {implicit_ratings_df.columns.tolist()}")
    implicit_ratings_df = implicit_ratings_df[cols_to_keep].copy()

    for df_part in [movielens_ratings_df, implicit_ratings_df]:
        for col in [C.Constant.USER_ID_COL, C.Constant.MOVIE_ID_COL]:
            df_part[col] = pd.to_numeric(df_part[col])
        df_part[C.Constant.RATING_COL] = pd.to_numeric(df_part[C.Constant.RATING_COL], errors='coerce')
        df_part[C.Constant.TIMESTAMP_COL] = pd.to_numeric(df_part[C.Constant.TIMESTAMP_COL], errors='coerce').fillna(0).astype(int)

    augmented_ratings_df = pd.concat([movielens_ratings_df, implicit_ratings_df], ignore_index=True)
    print(f"Nombre total de ratings après ajout : {len(augmented_ratings_df)}")
    
    if current_user_id not in augmented_ratings_df[C.Constant.USER_ID_COL].unique():
        print(f"ATTENTION : USER_ID ({current_user_id}) non trouvé dans le DataFrame augmenté !")
    else:
        print(f"Ratings pour USER_ID ({current_user_id}) trouvés et intégrés.")

    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(augmented_ratings_df[[C.Constant.USER_ID_COL, C.Constant.MOVIE_ID_COL, C.Constant.RATING_COL]], reader)
    augmented_trainset = data.build_full_trainset() 
    
    print(f"Trainset augmenté construit avec {augmented_trainset.n_users} utilisateurs et {augmented_trainset.n_items} items.")
    return augmented_trainset # Retourner le trainset pour la sauvegarde

def train_and_save_personalized_model(trainset_to_save, model_name_prefix, user_name_for_filename, model_type="SVD", model_config=None): # Renommé pour clarté
    """
    Entraîne un modèle sur le trainset fourni et le sauvegarde, ainsi que le trainset.
    """
    # L'ID utilisateur numérique est déjà dans le trainset, on utilise user_name_for_filename pour les noms de fichiers.
    # Pour l'affichage, on peut essayer de récupérer l'ID interne si le trainset est déjà construit.
    # Ici, trainset_to_save est l'objet trainset complet.
    
    # Tentative d'affichage d'un ID interne (peut être complexe si l'ID n'est pas le dernier ajouté)
    # Pour simplifier, on se fie au user_name_for_filename pour l'identification du profil.
    print(f"Entraînement du modèle {model_type} pour le profil '{user_name_for_filename}'...")
    
    if model_type == "SVD":
        algo = SVD(**(model_config if model_config else {}))
    else:
        raise ValueError(f"Type de modèle inconnu : {model_type}")

    algo.fit(trainset_to_save) # Entraîner sur le trainset passé en argument
    
    # Sauvegarde du modèle
    model_filename = C.Constant.MODEL_STORAGE_FILE_TEMPLATE_NAMED.format(user_name=user_name_for_filename, model_name=model_name_prefix)
    model_path = C.Constant.MODELS_STORAGE_PATH / model_filename
    print(f"Sauvegarde du modèle entraîné sous : {model_path}")
    dump.dump(str(model_path), algo=algo, verbose=1)
    
    # Sauvegarde du trainset
    trainset_filename = C.Constant.TRAINSET_STORAGE_FILE_TEMPLATE_NAMED.format(user_name=user_name_for_filename)
    trainset_path = C.Constant.MODELS_STORAGE_PATH / trainset_filename
    
    try:
        print(f"Sauvegarde du trainset augmenté sous : {trainset_path}")
        # Utiliser la variable correcte du trainset ici (celle passée à la fonction)
        dump.dump(file_name=str(trainset_path), trainset=trainset_to_save, verbose=1)
        print(f"Trainset pour {user_name_for_filename} sauvegardé avec surprise.dump.")
    except TypeError as te:
        if "unexpected keyword argument 'trainset'" in str(te) or "dump() got an unexpected keyword argument 'trainset'" in str(te): # Gérer les deux messages d'erreur possibles
            print(f"AVERTISSEMENT: surprise.dump a échoué pour le trainset de {user_name_for_filename}. Tentative avec pickle.")
            try:
                with open(str(trainset_path), 'wb') as f_pickle:
                    pickle.dump(trainset_to_save, f_pickle) # Utiliser la variable correcte
                print(f"Trainset pour {user_name_for_filename} sauvegardé avec pickle.")
            except Exception as pickle_e:
                print(f"ERREUR lors de la sauvegarde du trainset pour {user_name_for_filename} avec pickle : {pickle_e}")
        else:
            print(f"ERREUR (TypeError) lors de la sauvegarde du trainset pour {user_name_for_filename} : {te}")
    except Exception as e:
        print(f"ERREUR générale lors de la sauvegarde du trainset pour {user_name_for_filename} : {e}")


def main_recommender_building_for_user(user_name, 
                                       target_user_id, 
                                       model_prefix_name="svd_implicit", 
                                       model_type="SVD", 
                                       model_config_params=None):
    """
    Fonction principale pour un utilisateur donné par son nom et son ID cible.
    """
    if not hasattr(C.Constant, 'IMPLICIT_LIBRARY_FILENAME_TEMPLATE'):
        raise AttributeError("C.Constant.IMPLICIT_LIBRARY_FILENAME_TEMPLATE n'est pas défini dans constants.py")
    
    personal_library_filename = C.Constant.IMPLICIT_LIBRARY_FILENAME_TEMPLATE.format(user_name=user_name)
    personal_library_file_path = C.Constant.IMPLICIT_LIBRARIES_PATH / personal_library_filename
    
    movielens_ratings_file_path = C.Constant.EVIDENCE_PATH / C.Constant.RATINGS_FILENAME

    try:
        personal_library_df = pd.read_csv(personal_library_file_path)
        print(f"Bibliothèque personnelle pour '{user_name}' chargée depuis : {personal_library_file_path}")
    except FileNotFoundError:
        print(f"ERREUR : Fichier de bibliothèque personnelle '{personal_library_filename}' non trouvé à : {C.Constant.IMPLICIT_LIBRARIES_PATH}")
        return
    except Exception as e:
        print(f"ERREUR lors du chargement de la bibliothèque pour '{user_name}' : {e}")
        return

    try:
        implicit_ratings_df = calculate_implicit_ratings_from_library(personal_library_df, target_user_id)
    except ValueError as e:
        print(f"ERREUR lors du calcul des ratings implicites pour '{user_name}' : {e}")
        return
    if implicit_ratings_df.empty:
        print(f"Aucun rating implicite n'a été généré pour '{user_name}'.")
        return
    print(f"Ratings implicites calculés pour '{user_name}' (ID: {target_user_id}) :\n{implicit_ratings_df.head()}")

    try:
        movielens_ratings_df = pd.read_csv(movielens_ratings_file_path)
        if C.Constant.ITEM_ID_COL in movielens_ratings_df.columns and C.Constant.MOVIE_ID_COL != C.Constant.ITEM_ID_COL :
             movielens_ratings_df.rename(columns={C.Constant.ITEM_ID_COL: C.Constant.MOVIE_ID_COL}, inplace=True)
        elif C.Constant.MOVIE_ID_COL not in movielens_ratings_df.columns:
             raise ValueError(f"La colonne Movie ID ({C.Constant.MOVIE_ID_COL}) est introuvable dans {movielens_ratings_file_path}")
        print(f"Ratings MovieLens chargés depuis : {movielens_ratings_file_path}")
    except FileNotFoundError:
        print(f"ERREUR : Fichier de ratings MovieLens non trouvé à : {movielens_ratings_file_path}")
        return
    except Exception as e:
        print(f"ERREUR lors du chargement des ratings MovieLens : {e}")
        return

    # augmented_trainset est créé ici
    augmented_trainset = append_implicit_ratings_and_build_trainset(implicit_ratings_df, movielens_ratings_df, target_user_id)

    # Passer augmented_trainset à la fonction de sauvegarde
    train_and_save_personalized_model(
        trainset_to_save=augmented_trainset, # Passer le trainset ici
        model_name_prefix=model_prefix_name,
        user_name_for_filename=user_name, 
        model_type=model_type,
        model_config=model_config_params
    )
    
    print(f"\nProcessus de construction des modèles pour '{user_name}' (ID: {target_user_id}) terminé.")
    print(f"Les modèles et le trainset sont sauvegardés dans : {C.Constant.MODELS_STORAGE_PATH}")

if __name__ == '__main__':
    USER_NAME_TO_BUILD = "testuser"
    USER_ID_FOR_DATASET = -1     
    MODEL_TYPE_FOR_USER = "SVD" 
    MODEL_PREFIX_FOR_FILENAME = "svd_implicit"
    svd_config = {'n_factors': 50, 'n_epochs': 25, 'lr_all': 0.005, 'reg_all': 0.04, 'verbose': True}
    MODEL_PARAMS = svd_config 

    try:
        required_attrs = [
            'IMPLICIT_LIBRARIES_PATH', 'IMPLICIT_LIBRARY_FILENAME_TEMPLATE',
            'EVIDENCE_PATH', 'RATINGS_FILENAME', # Pour charger les ratings MovieLens
            'MODELS_STORAGE_PATH', 
            'MODEL_STORAGE_FILE_TEMPLATE_NAMED', 
            'TRAINSET_STORAGE_FILE_TEMPLATE_NAMED',
            'USER_ID_COL', 'MOVIE_ID_COL', 'RATING_COL', 'TIMESTAMP_COL'
        ]
        if not all(hasattr(C.Constant, attr) for attr in required_attrs):
            missing = [attr for attr in required_attrs if not hasattr(C.Constant, attr)]
            raise AttributeError(f"Constantes essentielles manquantes dans constants.py: {missing}")

        C.Constant.IMPLICIT_LIBRARIES_PATH.mkdir(parents=True, exist_ok=True)
        C.Constant.MODELS_STORAGE_PATH.mkdir(parents=True, exist_ok=True)

        movielens_ratings_path_check = C.Constant.EVIDENCE_PATH / C.Constant.RATINGS_FILENAME
        if not movielens_ratings_path_check.exists():
            raise FileNotFoundError(f"Le fichier de ratings MovieLens {movielens_ratings_path_check} n'existe pas.")

    except AttributeError as e:
        print(f"ERREUR DE CONFIGURATION (constants.py) : {e}")
        exit()
    except FileNotFoundError as e:
        print(f"ERREUR DE FICHIER : {e}")
        exit()
    except Exception as e:
        print(f"Une erreur inattendue est survenue lors de l'initialisation : {e}")
        exit()

    print(f"--- Lancement de Recommender Building pour l'utilisateur: {USER_NAME_TO_BUILD} (ID interne: {USER_ID_FOR_DATASET}) ---")
    
    main_recommender_building_for_user(
        user_name=USER_NAME_TO_BUILD,
        target_user_id=USER_ID_FOR_DATASET,
        model_prefix_name=MODEL_PREFIX_FOR_FILENAME,
        model_type=MODEL_TYPE_FOR_USER,
        model_config_params=MODEL_PARAMS
    )
