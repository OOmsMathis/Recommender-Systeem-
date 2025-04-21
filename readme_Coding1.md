# Coding 1 - The Analytics Module
For this first deliverable, we have implemented the main code `coding1.py`, which allows us to perform the various tasks requested.

## Path
Before using the code, we have defined our data path in the `constants.py` file, which contains a `Constant class` with the DATA_PATH. You should adapt this path based on where you store the necessary data files to run the code.

## Load data
To load the dataset, we used two functions defined in the `loaders.py`: file:
-`load_ratings`: loads user rating data from the corresponding CSV file.
-`load_items`: loads metadata about the items (e.g., movies) from a CSV file.


## Descriptive Statistics
User ratings (df_ratings) and items data (df_items) are loaded using the load_ratings() and load_items() functions as described in the previous paragraph
We apply counting methods such as .count() and .nunique() to extract key descriptive statistics about users, movies, and ratings.
These insights help us understand the dataset's structure and coverage before building recommendation models.

## Long-tail property
In this section, we analyze the distribution of ratings per movie to explore the long-tail property. This property highlights the fact that a few movies receive a large number of ratings, while the majority of movies receive very few. This is typical in recommendation systems, where some items are highly popular, but most are less frequently rated.

The following code generates a plot to visualize this distribution:
X-axis: Movie IDs sorted by the number of ratings.
Y-axis: The number of ratings each movie has received.

For this part we used .value_counts()method of DataFrames.
The result is visualized using a plot generated with the module matplotlib.

## Ratings matrix sparsity
For this part we have used the following documentation: # Source: https://www.jillcates.com/pydata-workshop/html/tutorial.html

To analyze the sparsity of our ratings matrix, we first computed the sparsity value using the formula introduced in the lecture: sparsity = 1.0 - (n_ratings / (n_users * n_films))

We used the create_X() function to convert df_ratings into a sparse matrix. This function maps user and movie IDs to unique indices and stores the ratings in a memory-efficient compressed format using csr_matrix from scipy.

To visualize the structure of the sparse matrix, we plotted the non-zero values of a sparse using .spy() method of the matplotlib module. We displayed a 100x100 section (first 100 users vs. first 100 movies), which helps illustrate the sparsity visually.

### groupe 4
- Delhoute Charles
- Dubart Quentin
- Ducarme Maxime
- Ooms Mathis
















