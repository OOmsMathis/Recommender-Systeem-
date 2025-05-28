# recommender_building.py

import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise import dump
import time
import os

# --- Configuration ---

# Demande le nom de la personne au lancement du script
PERSON_NAME = "maxime"
USER_ID = 0 # Choisis un ID non utilisé par MovieLens 

# Adapte ces chemins à ta structure de projet
PATH_TO_MOVIELENS_RATINGS = 'data/small/evidence/ratings.csv' # Chemin vers tes ratings MovieLens
PATH_TO_LIBRARY = f'library_{PERSON_NAME}.csv' # Chemin vers le fichier CSV personnel
OUTPUT_MODEL_PATH = 'data/small/recs/'
PERSONALIZED_MODEL_NAME = f'svd_{PERSON_NAME}_personalized.p'
USER_ID = 0 # Choisis un ID non utilisé par MovieLens (0 ou -1 sont de bonnes options)

# Assure-toi que le dossier de sortie existe
os.makedirs(OUTPUT_MODEL_PATH, exist_ok=True)

def calculate_implicit_rating(row):
    """
    Calcule une note implicite basée sur les colonnes de la bibliothèque personnelle.
    Échelle de 0.5 à 5.0.
    C'est une formule d'exemple, tu DOIS l'adapter à ta logique !
    """
    rating = 2.5 # Note de base neutre

    # Augmentation basée sur le nombre de visionnages
    if row['times_watched'] >= 5:
        rating += 1.5
    elif row['times_watched'] >= 2:
        rating += 1.0
    elif row['times_watched'] >= 1:
        rating += 0.5

    # Bonus pour les favoris
    if row['is_favorite'] == 1:
        rating += 1.0

    # Bonus pour la récence
    if row['recency_score'] >= 4: # Vu récemment ou très récemment
        rating += 0.5
    elif row['recency_score'] <= 2 and row['times_watched'] > 0 : # Vu il y a longtemps
        rating -= 0.25


    # Bonus si "plan to watch" et pas encore vu beaucoup
    if row['plan_to_watch'] == 1 and row['times_watched'] <= 1:
        rating = max(rating, 3.0) # Si planifié, au moins 3.0 s'il n'a pas été beaucoup vu et mal noté

    # Plafonner la note entre 0.5 et 5.0
    return max(0.5, min(5.0, rating))

def build_and_save_personalized_model():
    """
    Fonction principale pour lire la bibliothèque, l'ajouter aux ratings,
    entraîner un modèle et le sauvegarder.
    """
    # 1. Lire la bibliothèque personnelle et calculer les notes implicites
    try:
        df_library = pd.read_csv(PATH_TO_LIBRARY)
    except FileNotFoundError:
        print(f"ERREUR: Le fichier '{PATH_TO_LIBRARY}' n'a pas été trouvé. Crée-le d'abord.")
        return

    if not all(col in df_library.columns for col in ['movieId', 'times_watched', 'is_favorite', 'recency_score', 'plan_to_watch']):
        print("ERREUR: Le fichier CSV personnel ne contient pas toutes les colonnes requises.")
        print("Requis: movieId, times_watched, is_favorite, recency_score, plan_to_watch")
        return

    df_library['implicit_rating'] = df_library.apply(calculate_implicit_rating, axis=1)
    
    # Préparer les ratings de Maxime pour la fusion
    ratings_list = []
    current_timestamp = int(time.time()) # Timestamp actuel
    for _, row in df_library.iterrows():
        ratings_list.append({
            'userId': USER_ID,
            'movieId': row['movieId'],
            'rating': row['implicit_rating'],
            'timestamp': current_timestamp
        })
    df_ratings = pd.DataFrame(ratings_list)
    print(f"Notes implicites pour (userId={USER_ID}) calculées pour {len(df_ratings)} films.")
    print(df_ratings.head())

    # 2. Charger les ratings MovieLens et ajouter les ratings de Maxime
    try:
        df_movielens_ratings = pd.read_csv(PATH_TO_MOVIELENS_RATINGS)
    except FileNotFoundError:
        print(f"ERREUR: Le fichier de ratings MovieLens '{PATH_TO_MOVIELENS_RATINGS}' n'a pas été trouvé.")
        return

    # Vérifier si MAXIME_USER_ID existe déjà dans MovieLens (peu probable avec 0 ou -1)
    if USER_ID in df_movielens_ratings['userId'].unique():
        print(f"ATTENTION: L'userId {USER_ID} choisi pour Maxime existe déjà dans MovieLens. Choisis-en un autre.")
        # Optionnel: tu pourrais choisir de supprimer cet utilisateur existant ou de lever une erreur.
        # Pour cet exemple, on continue mais c'est à noter.

    df_combined_ratings = pd.concat([df_movielens_ratings, df_ratings], ignore_index=True)
    print(f"Taille du dataset MovieLens original: {len(df_movielens_ratings)} ratings.")
    print(f"Taille du dataset combiné (avec Maxime): {len(df_combined_ratings)} ratings.")

    # 3. Construire le jeu d'entraînement Surprise
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(df_combined_ratings[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset() # Utilise toutes les données pour l'entraînement final du modèle

    # 4. Entraîner un modèle (par exemple SVD)
    print("Entraînement du modèle SVD personnalisé...")
    algo = SVD(n_factors=100, n_epochs=20, biased=True, lr_all=0.005, reg_all=0.02, verbose=True) # Tu peux utiliser ton SVDAlgorithm de models.py aussi
    algo.fit(trainset)
    print("Entraînement terminé.")

    # 5. Sauvegarder le modèle entraîné
    model_full_path = os.path.join(OUTPUT_MODEL_PATH, PERSONALIZED_MODEL_NAME)
    print(f"Sauvegarde du modèle personnalisé dans: {model_full_path}")
    dump.dump(model_full_path, algo=algo)
    print("Modèle sauvegardé avec succès!")
    print(f"\nPour utiliser ce modèle, charge-le avec: _, loaded_algo = dump.load('{model_full_path}')")

if __name__ == '__main__':
    build_and_save_personalized_model()