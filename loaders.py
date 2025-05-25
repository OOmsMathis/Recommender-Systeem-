# third parties imports
import pandas as pd
from datetime import datetime
from surprise import Dataset, Reader
# local imports
from constants import Constant as C

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

def export_evaluation_report(df, features_used, regression_method):
    """
    Export the evaluation report to the specified evaluation directory in constants.
    Adds columns for used features and regression method.
    Appends new evaluations to the existing file if it exists.

    Args:
        df (DataFrame): The DataFrame containing the evaluation report.
        features_used (list or str): Features used in the evaluation.
        regression_method (str): Regression method used.
    """
    today_str = datetime.now().strftime("%Y_%m_%d")
    filename = f"evaluation_report_{today_str}.csv"
    path = C.EVALUATION_PATH / filename

    # Add columns for features and regression method
    df = df.copy()
    df['features_used'] = ','.join(features_used) if isinstance(features_used, list) else str(features_used)
    df['regression_method'] = regression_method

    # If file exists, append without header; else, create new file with header
    if path.exists():
        df.to_csv(path, mode='a', header=False, index=False)
    else:
        df.to_csv(path, index=False)
    print(f"Evaluation report successfully exported to: {path}")

