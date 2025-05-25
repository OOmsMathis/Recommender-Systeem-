import pandas as pd
from pathlib import Path
from surprise import Dataset, Reader, SVD, dump
import pickle # Pour une alternative de sauvegarde si surprise.dump pose problème pour le trainset

# Importer vos constantes
import constants as C

def create_and_save_general_model_and_trainset():
    """
    Crée et sauvegarde le trainset général et un modèle SVD général
    basé sur toutes les données de ratings.
    """
    print("Début de la création du modèle et du trainset généraux...")

    # 1. Définir les chemins en utilisant les constantes
    # Assurez-vous que ces constantes sont bien définies dans votre constants.py
    if not hasattr(C.Constant, 'EVIDENCE_PATH') or \
       not hasattr(C.Constant, 'RATINGS_FILENAME') or \
       not hasattr(C.Constant, 'MODELS_STORAGE_PATH') or \
       not hasattr(C.Constant, 'GENERAL_MODEL_NAME'):
        print("ERREUR: Une ou plusieurs constantes nécessaires (EVIDENCE_PATH, RATINGS_FILENAME, "
              "MODELS_STORAGE_PATH, GENERAL_MODEL_NAME) ne sont pas définies dans constants.py.")
        return

    ratings_file_path = C.Constant.EVIDENCE_PATH / C.Constant.RATINGS_FILENAME
    
    # Nom conventionnel pour le trainset général
    general_trainset_filename = "general_trainset.pkl"
    general_trainset_path = C.Constant.MODELS_STORAGE_PATH / general_trainset_filename
    
    general_model_filename = C.Constant.GENERAL_MODEL_NAME
    general_model_path = C.Constant.MODELS_STORAGE_PATH / general_model_filename

    # S'assurer que le dossier de stockage des modèles existe
    try:
        C.Constant.MODELS_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Erreur lors de la création du dossier {C.Constant.MODELS_STORAGE_PATH}: {e}")
        return

    # 2. Charger tous les ratings
    try:
        print(f"Chargement des ratings depuis : {ratings_file_path}")
        ratings_df = pd.read_csv(ratings_file_path)
        # S'assurer que les colonnes nécessaires sont présentes
        required_rating_cols = [C.Constant.USER_ID_COL, C.Constant.MOVIE_ID_COL, C.Constant.RATING_COL]
        if not all(col in ratings_df.columns for col in required_rating_cols):
            missing_cols = [col for col in required_rating_cols if col not in ratings_df.columns]
            raise ValueError(f"Colonnes manquantes dans {ratings_file_path}: {missing_cols}")

    except FileNotFoundError:
        print(f"ERREUR: Fichier de ratings {ratings_file_path} non trouvé.")
        return
    except ValueError as ve:
        print(f"ERREUR de données dans {ratings_file_path}: {ve}")
        return
    except Exception as e:
        print(f"ERREUR inattendue lors du chargement de {ratings_file_path}: {e}")
        return

    # 3. Créer le trainset général
    try:
        # L'échelle de rating pour le Reader. Votre constants.py a RATINGS_SCALE = (1, 5)
        # mais les ratings implicites sont [0.5, 5]. Pour un modèle général basé UNIQUEMENT sur
        # les ratings explicites de MovieLens, C.Constant.RATINGS_SCALE est correct.
        # Si ce modèle général devait un jour inclure des données implicites, (0.5, 5.0) serait plus sûr.
        # Pour l'instant, utilisons ce qui est dans constants.py pour les ratings explicites.
        print(f"Utilisation de l'échelle de rating de constants.py : {C.Constant.RATINGS_SCALE}")
        reader = Reader(rating_scale=C.Constant.RATINGS_SCALE) 
        
        data = Dataset.load_from_df(
            ratings_df[[C.Constant.USER_ID_COL, C.Constant.MOVIE_ID_COL, C.Constant.RATING_COL]],
            reader
        )
        print("Construction du trainset général complet...")
        full_trainset = data.build_full_trainset()
    except Exception as e:
        print(f"ERREUR lors de la création du Dataset/Trainset Surprise : {e}")
        return

    # 4. Sauvegarder le trainset général
    try:
        print(f"Sauvegarde du trainset général sous : {general_trainset_path}")
        # La documentation de Surprise indique que `trainset=` est un argument valide.
        # Si l'erreur persiste, c'est très curieux.
        dump.dump(file_name=str(general_trainset_path), trainset=full_trainset, verbose=1)
        print("Trainset général sauvegardé avec surprise.dump.")
    except TypeError as te: # Attraper spécifiquement le TypeError
        if "unexpected keyword argument 'trainset'" in str(te):
            print("AVERTISSEMENT: surprise.dump a échoué avec 'unexpected keyword argument trainset'. Tentative avec pickle.")
            try:
                with open(str(general_trainset_path), 'wb') as f_pickle:
                    pickle.dump(full_trainset, f_pickle)
                print("Trainset général sauvegardé avec pickle.")
            except Exception as pickle_e:
                print(f"ERREUR lors de la sauvegarde du trainset général avec pickle : {pickle_e}")
                return # Arrêter si la sauvegarde du trainset échoue
        else:
            print(f"ERREUR (TypeError) lors de la sauvegarde du trainset général avec surprise.dump : {te}")
            return
    except Exception as e:
        print(f"ERREUR générale lors de la sauvegarde du trainset général avec surprise.dump : {e}")
        return
            
    # 5. Entraîner le modèle général (exemple avec SVD)
    # Vous pouvez choisir d'autres hyperparamètres ou un autre algorithme.
    print("Entraînement du modèle général (SVD)...")
    # Exemple d'hyperparamètres, à ajuster potentiellement après évaluation
    algo_general = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, verbose=True) 
    try:
        algo_general.fit(full_trainset)
    except Exception as e:
        print(f"ERREUR lors de l'entraînement du modèle général : {e}")
        return

    # 6. Sauvegarder le modèle général
    try:
        print(f"Sauvegarde du modèle général sous : {general_model_path}")
        # La sauvegarde de l'algo avec `algo=` devrait fonctionner sans problème.
        dump.dump(file_name=str(general_model_path), algo=algo_general, verbose=1)
        print("Modèle général sauvegardé avec succès.")
    except Exception as e:
        print(f"ERREUR lors de la sauvegarde du modèle général : {e}")
        return

    print("\nModèle et trainset généraux créés et sauvegardés avec succès.")
    print(f"  Trainset: {general_trainset_path}")
    print(f"  Modèle:   {general_model_path}")

if __name__ == '__main__':
    # S'assurer que les constantes de chemin de base sont disponibles
    if not hasattr(C.Constant, 'DATA_PATH'):
        # Fallback si DATA_PATH n'est pas là mais que les autres le sont (moins probable avec votre constants.py)
        if hasattr(C.Constant, 'EVIDENCE_PATH') and hasattr(C.Constant, 'MODELS_STORAGE_PATH'):
            print("DATA_PATH non trouvé, mais EVIDENCE_PATH et MODELS_STORAGE_PATH existent.")
        else:
            print("ERREUR: Les constantes de chemin de base (DATA_PATH, EVIDENCE_PATH, MODELS_STORAGE_PATH) "
                  "semblent manquantes dans constants.py.")
            exit()
            
    create_and_save_general_model_and_trainset()