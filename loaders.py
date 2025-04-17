# third parties imports
import pandas as pd

# local imports
from constants import Constant as C
from surprise import *

def load_ratings(surprise_format=False):
    df_ratings = pd.read_csv(C.EVIDENCE_PATH / C.RATINGS_FILENAME)
    if surprise_format:
        reader = Reader(rating_scale=C.RATINGS_SCALE)

        data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)
        return data 
    
    else:
        return df_ratings


def load_items():
    df_items = pd.read_csv(C.CONTENT_PATH / C.ITEMS_FILENAME)
    df_items = df_items.set_index(C.ITEM_ID_COL)
    return df_items


def export_evaluation_report(df):
    """ Export the report to the evaluation folder.

    The name of the report is versioned using today's date
    """
    pass

