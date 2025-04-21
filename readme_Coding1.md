# Coding 1 - The Analytics Module
pour ce premeier rendu, nous avons realiser dans notre code principale `coding1.py` differentes tâches demander.

## Path
avant d'utiliser le code nous avons mis notre chemin vers les donnés dans le fichier `constants.py` qui reprend une class contant avec le datapath a adapter selon l'emplacement ou vous stockez les donnés utiles pour faire tourner ce code.

## Load data
To load the dataset, we used two functions defined in the `loaders.py`: file:
-`load_ratings`: loads user rating data from the corresponding CSV file.
-`load_items`: loads metadata about the items (e.g., movies) from a CSV file.


## Descriptive Statistics
User ratings (df_ratings) and item metadata (df_items) are loaded using the load_ratings() and load_items() functions as described in the previous paragraph
We apply counting methods such as .count() and .nunique() to extract key descriptive statistics about users, movies, and ratings.
These insights help us understand the dataset's structure and coverage before building recommendation models.

## Long-tail property

