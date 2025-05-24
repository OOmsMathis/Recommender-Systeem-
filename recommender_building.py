import pandas as pd
import time
from surprise import Dataset, Reader, dump

# Vos modules existants
import constants as C
import loaders
import models # Assurez-vous que models.py est accessible et correct

# --- Configuration pour ce script ---
# Assurez-vous que ces chemins et noms de colonnes sont corrects et définis dans constants.py
# ou définissez-les ici si ce n'est pas le cas.
# Exemple :
# PERSONAL_LIBRARY_PATH = C.Constant.DATA_PATH.parent / "implicit_libraries" / "library_votrenom.csv"
# MOVIELENS_RATINGS_PATH = C.Constant.EVIDENCE_PATH / C.Constant.RATINGS_FILENAME
# MODELS_STORAGE_PATH = C.Constant.DATA_PATH.parent / "models_streamlit_recs" # Doit correspondre à ce que recommender.py utilise
# MODELS_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
# TRAINSET_STORAGE_FILE = MODELS_STORAGE_PATH / "augmented_trainset_user{}.pkl"
# MODEL_STORAGE_FILE_TEMPLATE = MODELS_STORAGE_PATH / "user{}_{}_model.pkl"


# Votre ID utilisateur spécial pour la bibliothèque implicite
# (doit être un ID non utilisé dans MovieLens) [cite: 19]
PERSONAL_USER_ID = -1 # Ou max(userId) + 1, etc. [cite: 19, 20]

def calculate_implicit_ratings_from_library(library_df):
    """
    Calcule les notes implicites à partir du DataFrame de la bibliothèque personnelle.
    Basé sur la section 4.5.1 de "Practical Recommender Systems". [cite: 5, 17]
    La note doit être dans l'échelle MovieLens [0.5, 5]. [cite: 17]
    """
    print("Calcul des notes implicites à partir de la bibliothèque...")
    implicit_ratings = []

    # DÉFINISSEZ VOTRE LOGIQUE DE CALCUL ICI
    # C'est la partie la plus subjective et personnelle.
    # Exemple de structure (à adapter impérativement) :
    # Colonnes attendues dans library_df : 'movieId', 'n_watched', 'wishlist', 'recent', 'top10'
    # (celles que vous avez définies dans votre fichier CSV)
    for _, row in library_df.iterrows():
        movie_id = row['movieId']
        # Exemple de calcul simple (À PERSONNALISER FORTEMENT)
        score = 2.5 # Note de base
        if 'n_watched' in row and pd.notna(row['n_watched']):
            score += min(float(row['n_watched']), 5) * 0.2 # Max 1 pt pour n_watched
        if 'wishlist' in row and pd.notna(row['wishlist']) and int(row['wishlist']) == 1:
            score += 0.75
        if 'recent' in row and pd.notna(row['recent']) and int(row['recent']) == 1:
            score += 0.5
        if 'top10' in row and pd.notna(row['top10']) and int(row['top10']) == 1:
            score += 1.0

        # Assurer que le score est dans l'échelle [0.5, 5]
        final_score = max(0.5, min(5.0, score))
        implicit_ratings.append({
            C.Constant.USER_ID_COL: PERSONAL_USER_ID,
            C.Constant.ITEM_ID_COL: movie_id,
            C.Constant.RATING_COL: final_score,
            C.Constant.TIMESTAMP_COL: int(time.time()) # Timestamp actuel
        })
    
    print(f"{len(implicit_ratings)} notes implicites calculées.")
    return pd.DataFrame(implicit_ratings)

def augment_ratings_and_build_trainset(movielens_ratings_df, implicit_ratings_df):
    """
    Ajoute les notes implicites au dataset MovieLens et construit un trainset Surprise. [cite: 18]
    """
    print("Augmentation du dataset MovieLens avec les notes implicites...")
    augmented_ratings_df = pd.concat([movielens_ratings_df, implicit_ratings_df], ignore_index=True)
    
    reader = Reader(rating_scale=C.Constant.RATINGS_SCALE)
    data = Dataset.load_from_df(
        augmented_ratings_df[[C.Constant.USER_ID_COL, C.Constant.ITEM_ID_COL, C.Constant.RATING_COL]],
        reader
    )
    full_trainset = data.build_full_trainset()
    print("Trainset Surprise augmenté créé.")
    return full_trainset, augmented_ratings_df

def train_and_save_model(trainset, model_type, model_config, user_id_for_filename):
    """
    Entraîne un modèle sur le trainset et le sauvegarde. [cite: 21]
    """
    print(f"Entraînement du modèle {model_type} pour l'utilisateur {user_id_for_filename}...")
    
    if model_type == 'SVD':
        model = models.ModelBaseline4(**model_config) # Assurez-vous que ModelBaseline4 accepte les kwargs
    elif model_type == 'UserBased':
        model = models.UserBased(**model_config)
    elif model_type == 'ContentBased':
        # ContentBased a une initialisation différente, il prend features_methods et regressor_method
        model = models.ContentBased(
            features_methods=model_config.get('features_methods'),
            regressor_method=model_config.get('regressor_method')
        )
    else:
        raise ValueError(f"Type de modèle inconnu: {model_type}")

    model.fit(trainset)
    
    # Sauvegarde du modèle
    model_filename = C.Constant.MODEL_STORAGE_FILE_TEMPLATE.format(user_id_for_filename, model_type.lower())
    print(f"Sauvegarde du modèle dans {model_filename}...")
    dump.dump(str(model_filename), algo=model, verbose=1) # Utiliser surprise.dump [cite: 21, 22]

    # Sauvegarde du trainset associé (important pour les IDs internes de Surprise)
    trainset_filename = C.Constant.TRAINSET_STORAGE_FILE.format(user_id_for_filename)
    with open(trainset_filename, 'wb') as f:
        pickle.dump(trainset, f)
    print(f"Trainset associé sauvegardé dans {trainset_filename}")


def main():
    # 1. Charger la bibliothèque personnelle
    try:
        personal_library_df = pd.read_csv(C.Constant.PERSONAL_LIBRARY_PATH)
        # Assurez-vous que les colonnes movieId, et vos colonnes d'événements (n_watched, etc.) sont présentes.
        # Exemple : Vérifier si 'movieId' est présent
        if C.Constant.ITEM_ID_COL not in personal_library_df.columns:
            raise ValueError(f"La colonne '{C.Constant.ITEM_ID_COL}' est manquante dans la bibliothèque personnelle.")

    except FileNotFoundError:
        print(f"ERREUR: Fichier de bibliothèque personnelle non trouvé à {C.Constant.PERSONAL_LIBRARY_PATH}")
        print("Veuillez créer votre fichier library_votrenom.csv et le placer correctement.")
        return
    except ValueError as e:
        print(f"ERREUR dans le fichier de bibliothèque personnelle : {e}")
        return

    # 2. Calculer les notes implicites
    implicit_ratings_df = calculate_implicit_ratings_from_library(personal_library_df)

    # 3. Charger les notes MovieLens
    movielens_ratings_df = loaders.load_ratings(surprise_format=False)

    # 4. Augmenter le dataset et construire le trainset
    augmented_trainset, _ = augment_ratings_and_build_trainset(movielens_ratings_df, implicit_ratings_df)

    # 5. Entraîner et sauvegarder les modèles souhaités
    # Adaptez les configurations selon vos meilleurs hyperparamètres/choix
    models_to_train = {
        'SVD': {'n_factors': 50, 'n_epochs': 20, 'lr_all':0.005, 'reg_all':0.02, 'random_state': 42},
        'UserBased': {'k': 40, 'min_k': 2, 'sim_options': {'name': 'jaccard', 'min_support': 1, 'user_based': True}},
        'ContentBased': {'features_methods': ["Genre_tfidf", "Year_of_release"], 'regressor_method': 'ridge'}
    }

    for model_type, config in models_to_train.items():
        train_and_save_model(augmented_trainset, model_type, config, PERSONAL_USER_ID)
    
    print(f"Processus de construction des modèles pour l'utilisateur {PERSONAL_USER_ID} terminé.")
    print(f"Les modèles et le trainset sont sauvegardés dans : {C.Constant.MODELS_STORAGE_PATH}")

if __name__ == '__main__':
    # Avant de lancer, assurez-vous que C.Constant.PERSONAL_LIBRARY_PATH,
    # C.Constant.MODELS_STORAGE_PATH, C.Constant.TRAINSET_STORAGE_FILE,
    # et C.Constant.MODEL_STORAGE_FILE_TEMPLATE sont bien définis dans votre constants.py
    # et que le dossier C.Constant.MODELS_STORAGE_PATH existe.
    # Exemple de définitions à ajouter dans constants.py si elles n'y sont pas :
    # from pathlib import Path
    # class Constant:
    #     # ... vos constantes existantes ...
    #     DATA_ROOT = Path(__file__).parent.parent # Suppose que constants.py est dans un sous-dossier du projet
    #     PERSONAL_LIBRARY_PATH = DATA_ROOT / 'data' / 'implicit_libraries' / 'library_votrenom.csv' # Adaptez "votrenom"
    #     MODELS_STORAGE_PATH = DATA_ROOT / 'data' / 'small' / 'recs_personalized' # Ou le chemin de Workshop 2 "data/small/recs" [cite: 22]
    #     MODELS_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
    #     TRAINSET_STORAGE_FILE = MODELS_STORAGE_PATH / "augmented_trainset_user{}.pkl"
    #     MODEL_STORAGE_FILE_TEMPLATE = MODELS_STORAGE_PATH / "user{}_{}_model.pkl"

    import pickle # Ajout de l'import pickle ici pour la sauvegarde du trainset
    main()