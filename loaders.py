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

def export_evaluation_report(df):
    """
    Export the evaluation report to the specified evaluation directory in constants.
    The file name is based on the current date to avoid overwriting previous experiments.
    
    Args:
        df (DataFrame): The DataFrame containing the evaluation report.
    """
    # Format the current date to create a unique file name
    today_str = datetime.now().strftime("%Y_%m_%d")
    filename = f"evaluation_report_{today_str}.csv"
    
    # Construct the full path where the file will be saved
    path = C.EVALUATION_PATH / filename
    
    # Export the DataFrame to CSV
    df.to_csv(path, index=False)
    print(f"Evaluation report successfully exported to: {path}")

