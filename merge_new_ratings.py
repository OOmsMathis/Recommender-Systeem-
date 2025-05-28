# merge_new_ratings.py
import pandas as pd
from pathlib import Path
import os
import sys # Utilisé pour ajuster le chemin d'importation si nécessaire

# S'assurer que le script peut trouver le module constants
# Si constants.py est dans le même répertoire ou un répertoire parent simple, cela devrait suffire.
# Sinon, ajustez le chemin. Par exemple, si constants.py est dans le répertoire parent:
# current_dir = Path(__file__).resolve().parent
# project_root = current_dir.parent 
# sys.path.append(str(project_root))

try:
    import constants as C_module
except ImportError:
    print("ERREUR: Impossible d'importer 'constants'. Assurez-vous que constants.py est accessible.")
    print("Vous pourriez avoir besoin d'ajuster sys.path si ce script n'est pas dans le même répertoire que constants.py.")
    sys.exit(1) # Quitter si les constantes ne peuvent pas être chargées

C = C_module.Constant()

def merge_pending_ratings():
    """
    Fusionne les évaluations en attente avec le fichier d'évaluations principal.
    """
    main_ratings_path = C.EVIDENCE_PATH / C.RATINGS_FILENAME
    pending_ratings_path = C.EVIDENCE_PATH / getattr(C, 'NEW_RATINGS_PENDING_FILENAME', 'new_ratings_pending.csv') # Utilise getattr pour fallback

    print(f"--- Fusion des Évaluations en Attente ---")
    print(f"Fichier principal des évaluations : {main_ratings_path}")
    print(f"Fichier des évaluations en attente : {pending_ratings_path}")

    if not main_ratings_path.exists():
        print(f"ERREUR: Le fichier principal des évaluations '{main_ratings_path}' n'a pas été trouvé.")
        print("Veuillez vérifier le chemin et le nom du fichier dans constants.py.")
        return False

    if not pending_ratings_path.exists():
        print(f"INFO: Aucun fichier d'évaluations en attente trouvé à '{pending_ratings_path}'. Aucune action requise.")
        return True # Pas une erreur, juste rien à faire

    try:
        df_main = pd.read_csv(main_ratings_path)
        print(f"Nombre d'évaluations principales chargées : {len(df_main)}")
    except Exception as e:
        print(f"ERREUR lors de la lecture du fichier principal des évaluations '{main_ratings_path}': {e}")
        return False

    try:
        df_pending = pd.read_csv(pending_ratings_path)
        print(f"Nombre d'évaluations en attente chargées : {len(df_pending)}")
    except Exception as e:
        print(f"ERREUR lors de la lecture du fichier des évaluations en attente '{pending_ratings_path}': {e}")
        return False

    if df_pending.empty:
        print("INFO: Le fichier des évaluations en attente est vide. Aucune fusion nécessaire.")
        # Optionnel: supprimer le fichier pending vide s'il existe
        # try:
        #     pending_ratings_path.unlink()
        #     print(f"INFO: Fichier vide '{pending_ratings_path}' supprimé.")
        # except OSError as e_del:
        #     print(f"AVERTISSEMENT: Impossible de supprimer le fichier vide '{pending_ratings_path}': {e_del}")
        return True

    # S'assurer que les colonnes nécessaires sont présentes dans les deux DataFrames
    required_cols = [C.USER_ID_COL, C.ITEM_ID_COL, C.RATING_COL, C.TIMESTAMP_COL]
    for col in required_cols:
        if col not in df_main.columns:
            print(f"ERREUR: Colonne '{col}' manquante dans le fichier principal des évaluations.")
            return False
        if col not in df_pending.columns:
            print(f"ERREUR: Colonne '{col}' manquante dans le fichier des évaluations en attente.")
            return False
            
    # S'assurer que les types de données pour les colonnes clés sont cohérents avant la fusion
    # Ceci est crucial pour éviter les problèmes de types mixtes ou les erreurs de fusion.
    try:
        df_pending[C.USER_ID_COL] = df_pending[C.USER_ID_COL].astype(df_main[C.USER_ID_COL].dtype)
        df_pending[C.ITEM_ID_COL] = df_pending[C.ITEM_ID_COL].astype(df_main[C.ITEM_ID_COL].dtype)
        df_pending[C.RATING_COL] = df_pending[C.RATING_COL].astype(df_main[C.RATING_COL].dtype)
        df_pending[C.TIMESTAMP_COL] = df_pending[C.TIMESTAMP_COL].astype(df_main[C.TIMESTAMP_COL].dtype)
    except Exception as e_type:
        print(f"ERREUR lors de la conversion des types de données pour la fusion : {e_type}")
        print("Vérifiez que les données dans les fichiers CSV sont conformes.")
        return False

    # Concaténer les évaluations
    df_combined = pd.concat([df_main, df_pending], ignore_index=True)
    print(f"Nombre total d'évaluations après concaténation : {len(df_combined)}")

    # Optionnel mais recommandé : Gérer les doublons.
    # Si un utilisateur note à nouveau un film, quelle note conserver ? La plus récente ?
    # Ici, on conserve la dernière entrée (la plus récente si les timestamps sont corrects et uniques,
    # ou la dernière ajoutée au fichier pending).
    # Si vous avez une logique de timestamp fiable, trier par timestamp avant drop_duplicates.
    # df_combined.sort_values(C.TIMESTAMP_COL, ascending=True, inplace=True)
    
    # Supprimer les doublons en gardant la dernière occurrence pour une paire (utilisateur, item)
    # Cela signifie que si un utilisateur a noté un item dans df_main, puis à nouveau dans df_pending,
    # la note de df_pending (qui est à la fin après concat) sera conservée.
    num_before_dedup = len(df_combined)
    df_combined.drop_duplicates(subset=[C.USER_ID_COL, C.ITEM_ID_COL], keep='last', inplace=True)
    num_after_dedup = len(df_combined)
    if num_after_dedup < num_before_dedup:
        print(f"INFO: {num_before_dedup - num_after_dedup} doublons (utilisateur, item) ont été supprimés, en gardant la dernière entrée.")

    # Sauvegarder le fichier principal mis à jour
    try:
        df_combined.to_csv(main_ratings_path, index=False)
        print(f"SUCCÈS: Évaluations fusionnées et sauvegardées dans '{main_ratings_path}'. Total final : {len(df_combined)}")
    except Exception as e_save:
        print(f"ERREUR lors de la sauvegarde du fichier principal mis à jour '{main_ratings_path}': {e_save}")
        return False

    # Optionnel : Supprimer ou archiver le fichier des évaluations en attente après une fusion réussie
    try:
        # Renommer avec un timestamp pour l'archivage au lieu de supprimer directement
        archive_name = pending_ratings_path.stem + "_" + pd.Timestamp.now().strftime('%Y%m%d_%H%M%S') + pending_ratings_path.suffix
        archive_path = pending_ratings_path.with_name(archive_name)
        pending_ratings_path.rename(archive_path)
        print(f"INFO: Fichier des évaluations en attente archivé sous '{archive_path}'.")
        # Ou pour supprimer :
        # pending_ratings_path.unlink()
        # print(f"INFO: Fichier des évaluations en attente '{pending_ratings_path}' supprimé.")
    except OSError as e_archive:
        print(f"AVERTISSEMENT: Impossible d'archiver/supprimer le fichier des évaluations en attente '{pending_ratings_path}': {e_archive}")

    return True

if __name__ == '__main__':
    success = merge_pending_ratings()
    if success:
        print("\nProcessus de fusion terminé avec succès.")
        print(f"N'oubliez pas de ré-entraîner vos modèles avec 'training.py' en utilisant le fichier '{C.RATINGS_FILENAME}' mis à jour.")
    else:
        print("\nLe processus de fusion a rencontré une erreur.")