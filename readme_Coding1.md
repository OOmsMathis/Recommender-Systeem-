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
In this section, we analyze the distribution of ratings per movie to explore the long-tail property. This property highlights the fact that a few movies receive a large number of ratings, while the majority of movies receive very few. This is typical in recommendation systems, where some items are highly popular, but most are less frequently rated.
The following code generates a plot to visualize this distribution:
X-axis: Movie IDs sorted by the number of ratings.
Y-axis: The number of ratings each movie has received.
For this part we used .value_counts()method of DataFrames.
The result is visualized using a plot generated with matplotlib.pyplot as plt

## Long-tail property


