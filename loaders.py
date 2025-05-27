# third parties imports
import pandas as pd
from datetime import datetime
from surprise import Dataset, Reader
# local imports
from constants import Constant as C
import numpy as np

def load_ratings(surprise_format=False):
    df_ratings = pd.read_csv(C.EVIDENCE_PATH / C.RATINGS_FILENAME)
    if surprise_format:
        reader = Reader(rating_scale=C.RATINGS_SCALE)
        data = Dataset.load_from_df(df_ratings[['userId', 'movieId', 'rating']], reader)
        return data
    else:
        return df_ratings

print(load_ratings())

def load_items():
    df_items = pd.read_csv(C.CONTENT_PATH / C.ITEMS_FILENAME)
    df_items = df_items.set_index(C.ITEM_ID_COL)
    return df_items

def export_evaluation_report(df, accuracy=None, precision=None):
    """
    Export the evaluation report to the specified evaluation directory in constants.
    Adds columns for accuracy and precision.
    Generates a new report file each time without overwriting previous ones.

    Args:
        df (DataFrame): The DataFrame containing the evaluation report.
        accuracy (float, optional): Accuracy score.
        precision (float, optional): Precision score.
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

    df.to_csv(path, index=False)
    print(f"Evaluation report successfully exported to: {path}")

