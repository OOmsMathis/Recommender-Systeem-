# Group 04 - Movie Recommender System

Welcome to this readme :)

This readme concerns the recommender system project developed by Group 4 as part of the Recommender System MLSMM2156 course. The purpose of this file is to provide all the necessary information to use the code in the most optimal way possible.

**Before starting to read, make sure you have installed all the libraries included in the 'tools_and_infos/Pipfile' file.**

## 1. Exploring the Structure
The following folders contain tools and information not essential to the main process:

- /assignements: contains scripts related to homework and instructions assigned at the project's foundation, with the exception of evaluators.ipynb. The files present allowed us to build our models step by step.
- /data: contains the data necessary for the proper functioning of the project, see point 1 for importing them.
- /hackaton: contains specific codes related to the hackathon of May 15, 2025.
- /tools_and_infos: contains specific tools for certain functionalities that are not necessary in the main process, as well as additional information about the project in general.
- The isolated files are of more interest to us.

## 2. General Process Execution
Please find below the detailed process required to access recommendations in Streamlit.
1. Import data
2. Ensure that constants.py is fully correct
3. Run *training.py*
4. Run *app.py*
5. Enjoy with Streamlit UI

### 2.1 Import data
Before starting the execution of .py files, you must import the following necessary data:
- the basic "small" folder: available on the drive related to the course.
- the "tmdb_full_features" .csv: simply run the "obtain_tmdb_data.py" file.
- the "genomes-scores" and "genomes-tags" .csv files: [available at this link](https://grouplens.org/datasets/movielens/25m/ "MovieLens 25M Dataset"), import them into 'data/small/content'.

### 2.2 Ensure that constants.py is fully correct
'constants.py' centralizes all calls to data. It is therefore crucial to verify that the paths included in this file are correctly assigned before proceeding to the next steps.

### 2.3 Run *training.py*
Use the command *python training.py* to execute this code.

This script trains the models contained in *models.py*. The trained models are then saved in 'data/small/recs' for later use. It is notably here that the parameters, features, and regression methods used for our recommendations are defined.

### 2.4 Run *app.py*
Use the command *streamlit run app.py* to execute this code.

This script configures Streamlit. It uses the models from the previous step and displays the best recommendations and their explanation from the *recommender.py* and *explanations.py* scripts. It uses data from *content.py* to display the right information at the right time.

### 2.5 Enjoy with Streamlit UI
Discover the different options of Streamlit at your leisure:
- explore the general menu, discover the top 20 movies, the top 20 documentaries, and the top 20 suggestions to discover.
- use *filter by gender* and *filter by year* to adapt suggestions to your specific desires.
- click on a movie title to open its TMDB page and discover its complete information.
- search for a specific movie using the central search bar, log in, and rate it to add it to your profile.
- view your personalized recommendations by clicking on *Log in* and selecting your user_id.
- rate the movies you have already seen and click on *save my news ratings* below the suggestions.
- create your own profile using the *Create new profile* option, fill out the form, and enjoy instant suggestions.

## 3. Recommendation Update Process
Please execute this process if you find yourself in one of the following situations:
- a new profile has been created since the last training and you want to update the recommendations.
- ratings have been added to some profiles since the last training and you want to update the recommendations.

1. Run *merge_new_ratings.py*
2. Run *training.py* 
3. Run *app.py*
4. Enjoy with Streamlit UI 

### 3.1 Run *merge_new_ratings.py*
Use the command *python run merge_new_ratings.py* to execute this code.

This script adds new ratings from a new user (or an existing one) to the general database.

### 3.2 Run *training.py* & Run *app.py* & Enjoy with Streamlit UI
See points [2.3](#2-3-run-training-py), [2.4](#2-4-run-app-py), and [2.5](#2-5-enjoy-with-streamlit-ui) respectively.

---
This project was carried out by Dubart Quentin, Delhoute Charles, Ooms Mathis, and Ducarme Maxime.

As part of the course: Recommender System, MSLMM2156.
