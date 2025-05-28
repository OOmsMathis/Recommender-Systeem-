# profile_manager.py (version simplifiée)
import json
import os
from pathlib import Path
import pandas as pd # Nécessaire pour lire ratings.csv et trouver max_user_id
import streamlit as st

import constants as C_module
C = C_module.Constant()

C.DATA_PATH.mkdir(parents=True, exist_ok=True) # S'assurer que le dossier data existe

def _load_prenom_id_map():
    if C.USER_PRENOM_TO_ID_MAP_FILE.exists():
        with open(C.USER_PRENOM_TO_ID_MAP_FILE, 'r', encoding='utf-8') as f:
            try:
                return json.load(f) # Format: {"Prenom1": 1001, "Prenom2": 1002}
            except json.JSONDecodeError:
                return {}
    return {}

def _save_prenom_id_map(id_map):
    with open(C.USER_PRENOM_TO_ID_MAP_FILE, 'w', encoding='utf-8') as f:
        json.dump(id_map, f, indent=4, ensure_ascii=False)

def get_next_available_user_id():
    """
    Détermine le prochain UserID disponible en faisant max(existing_ids) + 1.
    Lit directement le fichier ratings.csv pour cela.
    """
    ratings_filepath = C.EVIDENCE_PATH / C.RATINGS_FILENAME
    if not ratings_filepath.exists():
        print(f"Fichier ratings {ratings_filepath} non trouvé pour déterminer le max UserID. Départ à 1.")
        return 1 # Ou une autre valeur de départ si le fichier est vide ou n'existe pas
        
    try:
        df_ratings = pd.read_csv(ratings_filepath)
        if df_ratings.empty or C.USER_ID_COL not in df_ratings.columns:
            return 1 # Si vide ou colonne manquante, commencer à 1
        max_id = df_ratings[C.USER_ID_COL].max()
        return int(max_id + 1)
    except Exception as e:
        print(f"Erreur lors de la lecture de {ratings_filepath} pour max UserID: {e}. Départ à 1.")
        return 1


def add_user_prenom_mapping(prenom, user_id):
    """Ajoute ou met à jour le mappage Prénom -> UserID."""
    if not prenom or not isinstance(user_id, int):
        return
    id_map = _load_prenom_id_map()
    id_map[prenom] = user_id
    _save_prenom_id_map(id_map)
    print(f"Mappage sauvegardé : {prenom} -> UserID {user_id}")

def get_user_id_by_prenom(prenom):
    id_map = _load_prenom_id_map()
    return id_map.get(prenom)

def get_prenom_by_user_id(user_id):
    id_map = _load_prenom_id_map()
    for prenom_map, uid_map in id_map.items():
        if uid_map == user_id:
            return prenom_map
    return None

def get_all_mapped_prenoms():
    """Retourne une liste de prénoms triés qui ont un UserID mappé."""
    id_map = _load_prenom_id_map()
    return sorted(list(id_map.keys()))

def append_ratings_to_csv(new_ratings_df):
    """Ajoute de nouvelles évaluations au fichier ratings.csv principal."""
    ratings_filepath = C.EVIDENCE_PATH / C.RATINGS_FILENAME
    try:
        if ratings_filepath.exists() and os.path.getsize(ratings_filepath) > 0:
            # Lire l'en-tête pour s'assurer qu'on écrit dans le bon format
            header = pd.read_csv(ratings_filepath, nrows=0).columns.tolist()
            new_ratings_df.to_csv(ratings_filepath, mode='a', header=False, index=False, columns=header)
        else: # Le fichier n'existe pas ou est vide, écrire avec en-tête
            new_ratings_df.to_csv(ratings_filepath, mode='w', header=True, index=False)
        print(f"{len(new_ratings_df)} nouvelles évaluations ajoutées à {ratings_filepath}")
        return True
    except Exception as e:
        print(f"Erreur lors de l'ajout des évaluations à {ratings_filepath}: {e}")
        return False

def delete_user_prenom_mapping(prenom_to_delete):
    """Supprime un utilisateur de la map Prénom <-> UserID."""
    id_map = _load_prenom_id_map()
    if prenom_to_delete in id_map:
        deleted_id = id_map.pop(prenom_to_delete)
        _save_prenom_id_map(id_map)
        print(f"Mappage pour {prenom_to_delete} (UserID {deleted_id}) supprimé.")
        st.info(f"Le mappage pour {prenom_to_delete} a été supprimé. "
                f"Note : Ses évaluations avec UserID {deleted_id} sont toujours dans ratings.csv. "
                "Un ré-entraînement des modèles est nécessaire pour que les changements soient pleinement effectifs "
                "et pour que cet UserID ne soit plus utilisable directement s'il n'a plus de mappage de nom.")
        return True
    return False