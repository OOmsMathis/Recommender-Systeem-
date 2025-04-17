import pandas as pd
import numpy as np
from loaders import load_ratings, load_items
from IPython.display import display
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO
import base64


#3 Loading data
df_ratings = load_ratings()
df_items = load_items()
print(df_items.head())


#4 Descriptive Statistics
n_ratings = df_ratings["rating"].count()
n_users = df_ratings["userId"].nunique()
n_films = df_items.index.nunique()
n_ratings_films_max = df_ratings["movieId"].value_counts().max() 
n_ratings_films_min = df_ratings["movieId"].value_counts().min() 
n_ratings_films_possible = sorted(df_ratings["rating"].unique())
n_films_not_rated = df_items.index.nunique() - df_ratings["movieId"].nunique()


print(f"(a) Total number of ratings : {n_ratings}")
print(f"(b) Total number of unique users : {n_users}")
print(f"(c)) Total number of unique movies : {n_films}")
print(f"(d) Number of ratings for the most rated movie : {n_ratings_films_max}")
print(f"(e) Number of ratings for the less rated movie : {n_ratings_films_min}")
print(f"(f) All possible rating values : {n_ratings_films_possible}")
print(f"(g) Number of movies that were not rated at all : {n_films_not_rated}")


#6. Long-tail property 
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
rating_counts = df_ratings["movieId"].value_counts()
rating_counts_sorted = rating_counts.sort_values(ascending=False)
plt.plot(range(1, len(rating_counts_sorted) + 1), rating_counts_sorted.values)
plt.xlabel('id of the movie')
plt.ylabel('Number of ratings')
plt.title('Distribution of ratings per movie (Long-tail property)')
plt.grid(True)
plt.show()



#7. Ratings matrix sparsity
sparsity = 1.0 - (n_ratings / (n_users * n_films))
# Source: https://www.jillcates.com/pydata-workshop/html/tutorial.html
def create_X(df):
    """
    Generates a sparse matrix from ratings dataframe.

    Args:
        df: pandas dataframe containing 3 columns (userId, movieId, rating)

    Returns:
        X: sparse matrix
        user_mapper: dict that maps user id's to user indices
        user_inv_mapper: dict that maps user indices to user id's
        movie_mapper: dict that maps movie id's to movie indices
        movie_inv_mapper: dict that maps movie indices to movie id's
    """
    M = df['userId'].nunique()
    N = df['movieId'].nunique()

    user_mapper = dict(zip(np.unique(df["userId"]), list(range(M))))
    movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(N))))

    user_inv_mapper = dict(zip(list(range(M)), np.unique(df["userId"])))
    movie_inv_mapper = dict(zip(list(range(N)), np.unique(df["movieId"])))

    user_index = [user_mapper[i] for i in df['userId']]
    item_index = [movie_mapper[i] for i in df['movieId']]

    X = csr_matrix((df["rating"], (user_index,item_index)), shape=(M,N))

    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_X(df_ratings)

plt.figure(figsize=(8, 8))
plt.spy(X[0:100, 0:100], markersize=1)
plt.title("Sparse Matrix (100 users x 100 movies)")
plt.xlabel("Movies")
plt.ylabel("Users")
plt.show()
print(f"Sparsity of the ratings matrix: {sparsity:.2%}")