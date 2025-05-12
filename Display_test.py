from loaders import *
from constants import Constant as C
import pandas as pd
from IPython.display import display

import streamlit as st
import os

st.title("Recommender System platform")
st.write("Welcome to our Recommender System platform !")
st.write("This platform will offer you a personalized experience based on your films preferences.")
st.header("Explore the dataset")

data = load_items()
st.subheader("Dataset")

# Display the dataset with pagination and search functionality
search_title = st.text_input("Search by title:")
search_category = st.text_input("Search by category:")

# Filter the dataset based on search inputs
if search_title:
    data = data[data['title'].str.contains(search_title, case=False, na=False)]
if search_category:
    data = data[data['genres'].str.contains(search_category, case=False, na=False)]

# Display the first 50 rows with scrolling enabled
st.dataframe(data.head(50))
st.write(data.head())
st.subheader("Dataset description") 


data[['title', 'year']] = data['title'].str.extract(r'^(.*)\s\((\d{4})\)$')
data['year'] = pd.to_numeric(data['year'], errors='coerce')  # Convertir l'année en numérique

st.dataframe(data[['title', 'year', 'genres']].head(50))

# ...existing code...

if st.button("Show description"):
    st.write(data.describe())
    # Add a button to execute the Streamlit app


if st.button("Run Streamlit App"):
        os.system("streamlit run Main.py")
