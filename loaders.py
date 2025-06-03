# third parties imports
import pandas as pd
from datetime import datetime
from surprise import Dataset, Reader
# local imports
from constants import Constant as C
import numpy as np
import json

def load_ratings(surprise_format=False):
    df_ratings = pd.read_csv(C.EVIDENCE_PATH / C.RATINGS_FILENAME)
    if surprise_format:
        reader = Reader(rating_scale=C.RATINGS_SCALE)
        data = Dataset.load_from_df(df_ratings[['userId', 'movieId', 'rating']], reader)
        return data
    else:
        return df_ratings

#print(load_ratings())

def load_items():
    df_items = pd.read_csv(C.CONTENT_PATH / C.ITEMS_FILENAME)
    df_items = df_items.set_index(C.ITEM_ID_COL)
    return df_items

def export_evaluation_report(df, model_name, accuracy=None, precision=None):
    """
    Export the evaluation report to the specified evaluation directory

    """

    today_str = datetime.now().strftime("%Y_%m_%d")
    base_filename = f"evaluation_report_{today_str}"
    i = 1
    while True:
        filename = f"{base_filename}_{i}.csv"
        path = C.EVALUATION_PATH / filename
        if not path.exists():
            break
        i += 1
    df = df.copy()
    # Insert model names as first column to keep track the model considered
    df.insert(0, "name", model_name)

    # Add accuracy and precision 
    if accuracy is not None:
        df["accuracy"] = accuracy
    if precision is not None:
        df["precision"] = precision
    df.to_csv(path, index=False)
    print(f"Evaluation report successfully exported to: {path}")



def convert_tags_json_to_csv():
    """
    Convertit un fichier tags.json où chaque ligne est un objet JSON {"tag": ..., "id": ...}
    en un fichier CSV avec les colonnes tagId et tag.
    """

    json_path = C.CONTENT_PATH / "tags.json"
    csv_path = C.CONTENT_PATH / "genome-tags-2.csv"
    tags = []

    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tags.append(json.loads(line))

    df = pd.DataFrame(tags)
    df = df.rename(columns={"id": "tagId"})
    df = df[["tagId", "tag"]]
    df.to_csv(csv_path, index=False)
    print(f"Tags CSV exported to: {csv_path}")

#convert_tags_json_to_csv()
def convert_glmer_to_genome_scores():
    """
    Convertit le fichier glmer.csv en genome-scores-2.csv avec les colonnes movieId, tagId, relevance.
    Associe les tags en lettres à leur tagId à partir de genome-tags-2.csv.
    Utilise la colonne 'score' de glmer.csv comme 'relevance'.
    """
    glmer_path = C.CONTENT_PATH / "glmer.csv"
    tags_path = C.CONTENT_PATH / "genome-tags-2.csv"
    output_path = C.CONTENT_PATH / "genome-scores-2.csv"

    # Charger les tags pour faire la correspondance tag -> tagId
    tags_df = pd.read_csv(tags_path)
    tag_to_id = dict(zip(tags_df["tag"], tags_df["tagId"]))

    # Charger le fichier glmer
    glmer_df = pd.read_csv(glmer_path)

    # Renommer item_id en movieId si nécessaire
    if "item_id" in glmer_df.columns:
        glmer_df = glmer_df.rename(columns={"item_id": "movieId"})

    # Conversion des tags en tagId
    if "tag" in glmer_df.columns:
        glmer_df["tagId"] = glmer_df["tag"].map(tag_to_id)
    else:
        raise ValueError("La colonne 'tag' est absente du fichier glmer.csv")

    # Vérifier la présence de la colonne 'score'
    if "score" not in glmer_df.columns:
        raise ValueError("La colonne 'score' est absente du fichier glmer.csv")

    # Renommer 'score' en 'relevance'
    glmer_df = glmer_df.rename(columns={"score": "relevance"})

    # Garder uniquement les colonnes nécessaires
    result_df = glmer_df[["movieId", "tagId", "relevance"]]

    # Exporter le résultat
    result_df.to_csv(output_path, index=False)
    print(f"Fichier genome-scores-2.csv exporté vers : {output_path}")

