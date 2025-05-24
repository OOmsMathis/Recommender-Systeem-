import pandas as pd
import time
from pathlib import Path
from surprise import Dataset, Reader, dump, SVD # SVD est un exemple

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

        # Borner le rating à l'échelle [0.5, 5.0] comme suggéré par Workshop 2 pour les ratings implicites
        rating = max(0.5, min(5.0, rating))
        
        implicit_ratings.append({
            C.Constant.USER_ID_COL: current_user_id,
            C.Constant.MOVIE_ID_COL: int(row[C.Constant.MOVIE_ID_COL]), # Utilise MOVIE_ID_COL de constants.py
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
    
    # Utilisation des noms de colonnes définis dans constants.py
    cols_to_keep = [C.Constant.USER_ID_COL, C.Constant.MOVIE_ID_COL, C.Constant.RATING_COL, C.Constant.TIMESTAMP_COL]
    
    movielens_ratings_df = movielens_ratings_df[cols_to_keep].copy()
    if not all(col in implicit_ratings_df.columns for col in cols_to_keep):
        raise ValueError(f"Les colonnes de implicit_ratings_df ne correspondent pas. Attendu: {cols_to_keep}, Obtenu: {implicit_ratings_df.columns.tolist()}")
    implicit_ratings_df = implicit_ratings_df[cols_to_keep].copy()

    # Conversion types numériques
    for df_part in [movielens_ratings_df, implicit_ratings_df]:
        for col in [C.Constant.USER_ID_COL, C.Constant.MOVIE_ID_COL]: # MOVIE_ID_COL ici
            df_part[col] = pd.to_numeric(df_part[col])
        df_part[C.Constant.RATING_COL] = pd.to_numeric(df_part[C.Constant.RATING_COL], errors='coerce')
        df_part[C.Constant.TIMESTAMP_COL] = pd.to_numeric(df_part[C.Constant.TIMESTAMP_COL], errors='coerce').fillna(0).astype(int)

    augmented_ratings_df = pd.concat([movielens_ratings_df, implicit_ratings_df], ignore_index=True)
    print(f"Nombre total de ratings après ajout : {len(augmented_ratings_df)}")
    
    if current_user_id not in augmented_ratings_df[C.Constant.USER_ID_COL].unique():
        print(f"ATTENTION : USER_ID ({current_user_id}) non trouvé dans le DataFrame augmenté !")
    else:
        print(f"Ratings pour USER_ID ({current_user_id}) trouvés et intégrés.")

    # L'échelle de rating pour le Reader doit couvrir 0.5-5.0 pour les ratings implicites (Workshop 2)
    # même si C.Constant.RATINGS_SCALE est (1,5) pour les ratings MovieLens originaux.
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(augmented_ratings_df[[C.Constant.USER_ID_COL, C.Constant.MOVIE_ID_COL, C.Constant.RATING_COL]], reader)
    augmented_trainset = data.build_full_trainset() 
    
    print(f"Trainset augmenté construit avec {augmented_trainset.n_users} utilisateurs et {augmented_trainset.n_items} items.")
    return augmented_trainset

def train_and_save_personalized_model(trainset, model_name_prefix, user_name_for_filename, model_type="SVD", model_config=None):
    """
    Entraîne un modèle sur le trainset fourni et le sauvegarde.
    Le nom du modèle contiendra le user_name_for_filename.
    """
    # Pour l'affichage, essayons de trouver l'ID numérique interne si possible.
    # Cela suppose que l'ID numérique est le dernier ajouté ou qu'il est connu.
    # Si `user_name_for_filename` est l'ID numérique, cela fonctionnera. Sinon, c'est juste pour l'affichage.
    try:
        internal_user_id_display = trainset.to_raw_uid(trainset.n_users -1) # Hypothèse: dernier utilisateur ajouté
    except: # Au cas où l'ID n'est pas directement le dernier ou n'est pas un int.
        internal_user_id_display = user_name_for_filename

    print(f"Entraînement du modèle {model_type} pour le profil '{user_name_for_filename}' (ID interne approx.: {internal_user_id_display})...")
    
    if model_type == "SVD":
        algo = SVD(**(model_config if model_config else {}))
    # elif model_type == "MonModeleCustom":
    #     algo = models.MonModeleCustomWrapper(**(model_config if model_config else {}))
    else:
        raise ValueError(f"Type de modèle inconnu : {model_type}")

    algo.fit(trainset)
    
    model_filename = C.Constant.MODEL_STORAGE_FILE_TEMPLATE_NAMED.format(user_name=user_name_for_filename, model_name=model_name_prefix)
    model_path = C.Constant.MODELS_STORAGE_PATH / model_filename
    
    print(f"Sauvegarde du modèle entraîné sous : {model_path}")
    dump.dump(str(model_path), algo=algo, verbose=1)
    
    trainset_filename = C.Constant.TRAINSET_STORAGE_FILE_TEMPLATE_NAMED.format(user_name=user_name_for_filename)
    trainset_path = C.Constant.MODELS_STORAGE_PATH / trainset_filename
    print(f"Sauvegarde du trainset augmenté sous : {trainset_path}")
    dump.dump(str(trainset_path), trainset=trainset, verbose=1)

def main_recommender_building_for_user(user_name, 
                                       target_user_id, 
                                       model_prefix_name="svd_implicit", 
                                       model_type="SVD", 
                                       model_config_params=None):
    """
    Fonction principale pour un utilisateur donné par son nom et son ID cible.
    Le fichier de bibliothèque sera recherché en utilisant C.Constant.IMPLICIT_LIBRARY_FILENAME_TEMPLATE.
    Les ratings MovieLens sont chargés en utilisant C.Constant.EVIDENCE_PATH et C.Constant.RATINGS_FILENAME.
    """
    personal_library_filename = C.Constant.IMPLICIT_LIBRARY_FILENAME_TEMPLATE.format(user_name=user_name)
    personal_library_file_path = C.Constant.IMPLICIT_LIBRARIES_PATH / personal_library_filename
    
    movielens_ratings_file_path = C.Constant.EVIDENCE_PATH / C.Constant.RATINGS_FILENAME

    # 1. Charger la bibliothèque personnelle
    try:
        personal_library_df = pd.read_csv(personal_library_file_path)
        print(f"Bibliothèque personnelle pour '{user_name}' chargée depuis : {personal_library_file_path}")
    except FileNotFoundError:
        print(f"ERREUR : Fichier de bibliothèque personnelle '{personal_library_filename}' non trouvé à : {C.Constant.IMPLICIT_LIBRARIES_PATH}")
        return
    except Exception as e:
        print(f"ERREUR lors du chargement de la bibliothèque pour '{user_name}' : {e}")
        return

    # 2. Calculer les ratings implicites
    try:
        implicit_ratings_df = calculate_implicit_ratings_from_library(personal_library_df, target_user_id)
    except ValueError as e:
        print(f"ERREUR lors du calcul des ratings implicites pour '{user_name}' : {e}")
        return
    if implicit_ratings_df.empty:
        print(f"Aucun rating implicite n'a été généré pour '{user_name}'.")
        return
    print(f"Ratings implicites calculés pour '{user_name}' (ID: {target_user_id}) :\n{implicit_ratings_df.head()}")

    # 3. Charger les ratings MovieLens
    try:
        movielens_ratings_df = pd.read_csv(movielens_ratings_file_path)
        # S'assurer que la colonne movie ID est bien nommée C.Constant.MOVIE_ID_COL ('movieId')
        # Votre constants.py utilise ITEM_ID_COL = 'movieId' et plus loin MOVIE_ID_COL = 'movieId'.
        # Je suppose que la colonne dans ratings.csv est bien 'movieId'.
        if C.Constant.ITEM_ID_COL not in movielens_ratings_df.columns and C.Constant.MOVIE_ID_COL in movielens_ratings_df.columns:
            pass # C'est bon si MOVIE_ID_COL ('movieId') est là
        elif C.Constant.ITEM_ID_COL in movielens_ratings_df.columns and C.Constant.MOVIE_ID_COL != C.Constant.ITEM_ID_COL :
             movielens_ratings_df.rename(columns={C.Constant.ITEM_ID_COL: C.Constant.MOVIE_ID_COL}, inplace=True)
        elif C.Constant.MOVIE_ID_COL not in movielens_ratings_df.columns:
             raise ValueError(f"La colonne Movie ID ({C.Constant.MOVIE_ID_COL} ou {C.Constant.ITEM_ID_COL}) est introuvable dans {movielens_ratings_file_path}")

        print(f"Ratings MovieLens chargés depuis : {movielens_ratings_file_path}")
    except FileNotFoundError:
        print(f"ERREUR : Fichier de ratings MovieLens non trouvé à : {movielens_ratings_file_path}")
        return
    except Exception as e:
        print(f"ERREUR lors du chargement des ratings MovieLens : {e}")
        return

    # 4. Ajouter les ratings implicites et construire le trainset
    augmented_trainset = append_implicit_ratings_and_build_trainset(implicit_ratings_df, movielens_ratings_df, target_user_id)

    # 5. Entraîner et sauvegarder le modèle personnalisé
    train_and_save_personalized_model(
        trainset=augmented_trainset,
        model_name_prefix=model_prefix_name,
        user_name_for_filename=user_name, 
        model_type=model_type,
        model_config=model_config_params
    )
    
    print(f"\nProcessus de construction des modèles pour '{user_name}' (ID: {target_user_id}) terminé.")
    print(f"Les modèles et le trainset sont sauvegardés dans : {C.Constant.MODELS_STORAGE_PATH}")

if __name__ == '__main__':
    # --- CONFIGURATION POUR L'EXÉCUTION SPÉCIFIQUE D'UN PROFIL ---
    USER_NAME_TO_BUILD = "testuser"  # Nom de la personne (utilisé pour library_testuser.csv et model_testuser_...)
    USER_ID_FOR_DATASET = -1     # ID numérique unique pour cette personne dans le dataset Surprise
    
    MODEL_TYPE_FOR_USER = "SVD" 
    MODEL_PREFIX_FOR_FILENAME = "svd_implicit" # Ex: "model_alice_svd_implicit.pkl"
    
    svd_config = {'n_factors': 50, 'n_epochs': 25, 'lr_all': 0.005, 'reg_all': 0.04, 'verbose': True}
    MODEL_PARAMS = svd_config 

    # --- Vérification des constantes et des chemins ---
    try:
        required_paths = [
            C.Constant.IMPLICIT_LIBRARIES_PATH,
            C.Constant.EVIDENCE_PATH,
            C.Constant.MODELS_STORAGE_PATH
        ]
        for p in required_paths: 
            if not isinstance(p, Path):
                 raise AttributeError(f"La constante {p} devrait être un objet Path.")
        
        required_files_templates = [
            'IMPLICIT_LIBRARY_FILENAME_TEMPLATE',
            'MODEL_STORAGE_FILE_TEMPLATE_NAMED',
            'TRAINSET_STORAGE_FILE_TEMPLATE_NAMED'
        ]
        for tmpl in required_files_templates:
            if not hasattr(C.Constant, tmpl):
                raise AttributeError(f"Constante de template manquante dans constants.py: {tmpl}. Veuillez l'ajouter comme suggéré.")

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

    # --- Lancement du processus de construction ---
    print(f"--- Lancement de Recommender Building pour l'utilisateur: {USER_NAME_TO_BUILD} (ID interne: {USER_ID_FOR_DATASET}) ---")
    
    main_recommender_building_for_user(
        user_name=USER_NAME_TO_BUILD,
        target_user_id=USER_ID_FOR_DATASET,
        model_prefix_name=MODEL_PREFIX_FOR_FILENAME,
        model_type=MODEL_TYPE_FOR_USER,
        model_config_params=MODEL_PARAMS
    )