# Coding 1 - The Analytics Module
For this first deliverable, we have implemented the main code `coding1.py`, which allows us to perform the various tasks requested.

## Path
Before using the code, we have defined our data path in the `constants.py` file, which contains a `Constant class` with the DATA_PATH. You should adapt this path based on where you store the necessary data files to run the code.

## Load data
To load the dataset, we used two functions defined in the `loaders.py`: file:
-`load_ratings`: loads user rating data from the corresponding CSV file.
-`load_items`: loads metadata about the items (e.g., movies) from a CSV file.


## Descriptive Statistics
User ratings (df_ratings) and item metadata (df_items) are loaded using the load_ratings() and load_items() functions as described in the previous paragraph
We apply counting methods such as .count() and .nunique() to extract key descriptive statistics about users, movies, and ratings.
These insights help us understand the dataset's structure and coverage before building recommendation models.

## Long-tail property

