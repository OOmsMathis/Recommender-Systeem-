# train_models_script.py

import pandas as pd
from surprise import Dataset, Reader, dump
import os
import constants as C_module
C = C_module.Constant() 
from models import ContentBased, UserBased, SVDAlgo, df_ratings_global

# --- Configuration ---
# Le chemin des ratings est utilisé par Surprise pour créer le dataset.
# df_ratings_global est déjà chargé dans models.py et sera utilisé par ContentBased.create_content_features
PATH_TO_RATINGS_FOR_SURPRISE = str(C.EVIDENCE_PATH / C.RATINGS_FILENAME) 
OUTPUT_MODELS_DIR = str(C.DATA_PATH / 'recs') # Assumant que tu veux les sauvegarder dans data/small/recs 
os.makedirs(OUTPUT_MODELS_DIR, exist_ok=True)

# --- Préparation des données Surprise ---
# Cette partie est surtout pour UserBased et SVDAlgo, et pour fournir le 'trainset' à ContentBased.fit()
print(f"Chargement des ratings pour Surprise depuis: {PATH_TO_RATINGS_FOR_SURPRISE}")
try:
    # Utilise df_ratings_global qui est déjà chargé et vérifié dans models.py
    # Cependant, Surprise a besoin de son propre chargement pour créer son trainset.
    # On s'assure que le fichier lu par Surprise est le même que celui utilisé pour df_ratings_global.
    reader = Reader(rating_scale=C.RATINGS_SCALE)
    data = Dataset.load_from_df(df_ratings_global[[C.USER_ID_COL, C.ITEM_ID_COL, C.RATING_COL]], reader)
    trainset_full = data.build_full_trainset()
    print("Trainset Surprise complet construit.")
except FileNotFoundError:
    print(f"ERREUR: Fichier de ratings non trouvé à {PATH_TO_RATINGS_FOR_SURPRISE} pour Surprise. Vérifie C.RATINGS_FILENAME.")
    exit()
except KeyError as e:
    print(f"ERREUR de clé lors de la préparation des données Surprise: {e}. Vérifie les noms de colonnes dans df_ratings_global et tes constantes.")
    exit()
except Exception as e:
    print(f"ERREUR inattendue lors de la préparation des données Surprise: {e}")
    exit()


# --- Entraînement et Sauvegarde ---

# 1. SVDAlgo
print("\n--- SVDAlgo ---")
svd_model = SVDAlgo(n_factors=100, n_epochs=20, random_state=42, verbose=True) # Hyperparamètres par défaut
svd_model.fit(trainset_full)
dump.dump(os.path.join(OUTPUT_MODELS_DIR, 'svd_model_final.p'), algo=svd_model)
print("SVDAlgo entraîné et sauvegardé.")

# 2. UserBased
print("\n--- UserBased ---")
sim_method = 'msd'  # Change ici si tu veux tester d'autres méthodes de similarité
user_based_model = UserBased(k=40, min_k=2, sim_options={'name': sim_method, 'user_based': True}, verbose=True)
user_based_model.fit(trainset_full)
user_based_filename = f"user_based_model_{sim_method}_final.p"
dump.dump(os.path.join(OUTPUT_MODELS_DIR, user_based_filename), algo=user_based_model)
print(f"UserBased entraîné et sauvegardé sous {user_based_filename}.")

# 3. ContentBased
#    Définis ici la liste des features que tu veux que ton ContentBased utilise.

features = [
    "title_length", 
    #"Year_of_release", 
    #"average_ratings", 
    #"count_ratings", 
    #"Genre_binary", 
    #"Genre_tfidf", 
    #"Tags_tfidf", 
    #"tmdb_vote_average", 
    #"title_tfidf", 
]
regressor_method = 'ridge'  # Change ici pour tester d'autres régressseurs si besoin

cb_model_all_features = ContentBased(
    features_methods=features,
    regressor_method=regressor_method
)
cb_model_all_features.fit(trainset_full)

# Génère un nom de fichier basé sur les features sélectionnées
features_str = "_".join([f for f in features if not f.startswith("#")]).replace(" ", "")
if not features_str:
    features_str = "nofeatures"
filename = f"content_based_{regressor_method}_{features_str}.p"

dump.dump(os.path.join(OUTPUT_MODELS_DIR, filename), algo=cb_model_all_features)
print(f"ContentBased ({regressor_method} avec {len(features)} types de features) entraîné et sauvegardé sous {filename}.")


print(f"\n\nTous les modèles sélectionnés ont été entraînés et sauvegardés dans '{OUTPUT_MODELS_DIR}'")