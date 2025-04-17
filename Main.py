from loaders import *
from constants import Constant as C
import pandas as pd
from IPython.display import display


# Loading the data 
df_items = load_items()
df_raitings = load_ratings()
print(df_items.head())
print(df_raitings.head())

# Descriptive statistics
print("--------------------------------------")
print("Descriptive statistics")
n_movies = df_items.shape[0]
print(f"Number of movies: {n_movies}")

n_unique_user = df_raitings[C.USER_ID_COL].nunique()
print(f"Number of unique users: {n_unique_user}")

n_unique_movie = df_raitings[C.ITEM_ID_COL].nunique()
print(f"Number of unique movies: {n_unique_movie}")

