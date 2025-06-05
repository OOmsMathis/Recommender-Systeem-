import streamlit as st
import pandas as pd
import os
st.set_page_config(page_title="Movie Recommendation", layout="wide")

import re
import random
import time
import math
import numpy as np # AJOUTÉ : Pour les opérations vectorielles (fold-in)
from surprise import Dataset, Reader, dump # AJOUTÉ : dump pour charger le modèle global
import requests
import zipfile
import constants as C_module
C = C_module.Constant()
import content

import recommender
import explanations
from loaders import load_items, load_ratings, load_posters_dict
# AJOUTÉ : ModelSVDpp depuis models.py, en supposant qu'il y est défini
from models import df_items_global as models_df_items_global
from pathlib import Path
import base64

# --- Constantes ---



N_RECOS_PERSONNALISEES_TOTAL_FETCH = 50
N_INSTANT_RECOS_NEW_USER = 10
CARDS_PER_ROW = 5
model_path = 'mlsmm2156/data/small/recs/svdpp_global_model.p' # AJOUTÉ



# ------------------------------------------------------- Configuration de la page Streamlit -------------------------------------------------------


# Ajoute un fond gris à la page principale
st.markdown(
    """
    <style>
    body, .stApp {
        background-color: #f5f5f5;
        background-image: url("https://www.transparenttextures.com/patterns/white-wall-3.png");
        background-attachment: fixed;
        background-size: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Conteneur pour le bouton de retour à l'accueil ---
# Utilisez des colonnes pour positionner le bouton à gauche
# Centrer le bouton avec l'image demandée
col_spacer_left, col_home_button, col_spacer_right = st.columns([0.01, 0.1, 0.01])
with col_spacer_left:
    if st.button(":material/house:", key="top_home_btn", use_container_width=True):
        st.session_state.active_page = "general"
        st.session_state.search_input_value = ""
        st.session_state.active_search_query = ""
        st.rerun()


st.markdown(
    f"""
    <style>
    .marquee-container {{
        overflow: hidden;
        position: relative;
        width: 90%;
        margin: 50px auto;
        background-color: transparent;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        height: 120px; /* Augmente la hauteur de l'encadré */
        min-height: 120px;
        display: flex;
        align-items: center;
    }}
    .marquee-container::before {{
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        z-index: 0;
        background-image: url("https://png.pngtree.com/background/20210710/original/pngtree-creative-black-gold-21st-shanghai-international-film-festival-banner-picture-image_1061201.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        border-radius: 15px;
    }}
    .marquee-text {{
        position: relative;
        z-index: 1;
        display: inline-block;
        white-space: nowrap;
        padding-left: 100%;
        animation: marquee 6s linear infinite;
        color: #FFFFFF;
        font-size: 3.5em;
        font-weight: 900;
        letter-spacing: 2.5px;
        font-family: 'Roboto', Arial, Helvetica, sans-serif;
        text-shadow: 1px 1px 2px #F5CB5C, 0 2px 8px #fffbe6;
        line-height: 120px;
    }}
    @keyframes marquee {{
        0%   {{ transform: translateX(0%); }}
        100% {{ transform: translateX(-100%); }}
    }}
    </style>
    <div class="marquee-container">
        <div class="marquee-text">Welcome to our application !</div>
    </div>
    """,
    unsafe_allow_html=True
)



# --- Chargement des données ---
# AJOUTÉ : Fonction pour charger le modèle SVD++ global pré-entraîné
@st.cache_resource 
def load_global_svdpp_model():
    model_path = 'mlsmm2156/data/small/recs/svdpp_global_model.p'
    if os.path.exists(model_path):
        print(f"Loading global SVD++ model from {model_path}...")
        try:
            _, model = dump.load(str(model_path))
            print("Global SVD++ model loaded successfully.")
            return model
        except Exception as e:
            st.error(f"Error loading the global SVD++ model: {e}")
            print(f"Critical error loading global SVD++ model: {e}")
            return None
    else:
        st.error(f"Global SVD++ model not found at: {model_path}. Please train it using training.py.")
        print(f"Critical Error: Global SVD++ model not found at {model_path}")
        return None

try:
    df_items_global_app = load_items()
    df_ratings_global_app = load_ratings() # Peut être vide initialement
    poster_map = load_posters_dict() # Chargement des posters
    if not poster_map:
        st.warning("No posters found in the specified directory. Movie posters will not be displayed.")
    
    if df_items_global_app.empty: # df_ratings_global_app peut être vide
        st.error("Critical error: Movie or rating data could not be loaded.")
        st.stop()

    # AJOUTÉ : Charger le modèle SVD++ global au démarrage
    global_svdpp_model = load_global_svdpp_model()
    if global_svdpp_model is None:
        # Gérer l'absence du modèle global, par exemple en affichant un avertissement
        # ou en désactivant les fonctionnalités qui en dépendent.
        print("WARNING: Global SVD++ model could not be loaded. New user recommendations via SVD++ will not work.")


except Exception as e_load:
    
    st.error(f"Fatal error during initial data loading: {e_load}")
    
    st.stop()



# --- Fonctions de récupération de données (repris de votre app(3).py) ---

@st.cache_data

def get_top_overall_movies_tmdb(n=N_RECOS_PERSONNALISEES_TOTAL_FETCH, genre_filter=None, year_min_filter=None, year_max_filter=None):
    if df_items_global_app.empty or not hasattr(C, 'VOTE_AVERAGE_COL') or C.VOTE_AVERAGE_COL not in df_items_global_app.columns:
        return pd.DataFrame()
    items_to_consider = df_items_global_app.copy()
    if genre_filter and genre_filter != "All Genres" and hasattr(C, 'GENRES_COL') and C.GENRES_COL in items_to_consider.columns:
        items_to_consider = items_to_consider[
            items_to_consider[C.GENRES_COL].astype(str).str.contains(re.escape(genre_filter), case=False, na=False, regex=True)
        ]
        if items_to_consider.empty: return pd.DataFrame()
    items_to_consider[C.VOTE_AVERAGE_COL] = pd.to_numeric(items_to_consider[C.VOTE_AVERAGE_COL], errors='coerce')
    vote_count_col_to_use = C.VOTE_COUNT_COL
    if not hasattr(C, 'VOTE_COUNT_COL') or C.VOTE_COUNT_COL not in items_to_consider.columns:
        items_to_consider['temp_vote_count'] = 100
        vote_count_col_to_use = 'temp_vote_count'
    else:
        items_to_consider[C.VOTE_COUNT_COL] = pd.to_numeric(items_to_consider[C.VOTE_COUNT_COL], errors='coerce').fillna(0)
    if year_min_filter is not None and year_max_filter is not None and hasattr(C, 'RELEASE_YEAR_COL') and C.RELEASE_YEAR_COL in items_to_consider.columns:
        items_to_consider[C.RELEASE_YEAR_COL] = pd.to_numeric(items_to_consider[C.RELEASE_YEAR_COL], errors='coerce').fillna(0)
        items_to_consider = items_to_consider[
            (items_to_consider[C.RELEASE_YEAR_COL] >= year_min_filter) &
            (items_to_consider[C.RELEASE_YEAR_COL] <= year_max_filter)
        ]
    if items_to_consider.empty: return pd.DataFrame()
    min_tmdb_votes_threshold = 100
    qualified_movies = items_to_consider[items_to_consider[vote_count_col_to_use] >= min_tmdb_votes_threshold]
    top_movies_source = qualified_movies if not qualified_movies.empty else items_to_consider
    top_movies_df = top_movies_source.sort_values(by=C.VOTE_AVERAGE_COL, ascending=False).head(n)
    cols_out = [C.ITEM_ID_COL, C.LABEL_COL, C.GENRES_COL, C.RELEASE_YEAR_COL, C.VOTE_AVERAGE_COL, C.VOTE_COUNT_COL, C.TMDB_ID_COL]
    return top_movies_df[[col for col in cols_out if col in top_movies_df.columns]]

@st.cache_data
def get_top_genre_movies_tmdb(genre, n=N_RECOS_PERSONNALISEES_TOTAL_FETCH, year_min_filter=None, year_max_filter=None):
    if df_items_global_app.empty or not hasattr(C, 'GENRES_COL') or C.GENRES_COL not in df_items_global_app.columns or \
       not hasattr(C, 'VOTE_AVERAGE_COL') or C.VOTE_AVERAGE_COL not in df_items_global_app.columns:
        return pd.DataFrame()
    genre_movies_df = df_items_global_app[
        df_items_global_app[C.GENRES_COL].astype(str).str.contains(re.escape(genre), case=False, na=False, regex=True)
    ]
    if genre_movies_df.empty: return pd.DataFrame()
    items_to_consider = genre_movies_df.copy()
    items_to_consider[C.VOTE_AVERAGE_COL] = pd.to_numeric(items_to_consider[C.VOTE_AVERAGE_COL], errors='coerce')
    vote_count_col_to_use = C.VOTE_COUNT_COL
    if not hasattr(C, 'VOTE_COUNT_COL') or C.VOTE_COUNT_COL not in items_to_consider.columns:
        items_to_consider['temp_vote_count'] = 50
        vote_count_col_to_use = 'temp_vote_count'
    else:
        items_to_consider[C.VOTE_COUNT_COL] = pd.to_numeric(items_to_consider[C.VOTE_COUNT_COL], errors='coerce').fillna(0)
    if year_min_filter is not None and year_max_filter is not None and hasattr(C, 'RELEASE_YEAR_COL') and C.RELEASE_YEAR_COL in items_to_consider.columns:
        items_to_consider[C.RELEASE_YEAR_COL] = pd.to_numeric(items_to_consider[C.RELEASE_YEAR_COL], errors='coerce').fillna(0)
        items_to_consider = items_to_consider[
            (items_to_consider[C.RELEASE_YEAR_COL] >= year_min_filter) &
            (items_to_consider[C.RELEASE_YEAR_COL] <= year_max_filter)
        ]
    if items_to_consider.empty: return pd.DataFrame()
    min_tmdb_votes_threshold = 50
    qualified_movies = items_to_consider[items_to_consider[vote_count_col_to_use] >= min_tmdb_votes_threshold]
    top_movies_source = qualified_movies if not qualified_movies.empty else items_to_consider
    top_genre_df_res = top_movies_source.sort_values(by=C.VOTE_AVERAGE_COL, ascending=False).head(n)
    cols_out = [C.ITEM_ID_COL, C.LABEL_COL, C.GENRES_COL, C.RELEASE_YEAR_COL, C.VOTE_AVERAGE_COL, C.VOTE_COUNT_COL, C.TMDB_ID_COL]
    return top_genre_df_res[[col for col in cols_out if col in top_genre_df_res.columns]]

@st.cache_data
def get_hidden_gems_movies(n=20, genre_filter=None, year_min_filter=None, year_max_filter=None,
                           min_vote_average_initial=6.0,
                           min_votes_initial=10,
                           num_novel_candidates=100,
                           excluded_genre="Documentary"):
    if df_items_global_app.empty or not hasattr(C, 'VOTE_COUNT_COL') or C.VOTE_COUNT_COL not in df_items_global_app.columns \
       or not hasattr(C, 'VOTE_AVERAGE_COL') or C.VOTE_AVERAGE_COL not in df_items_global_app.columns:
        return pd.DataFrame()
    items_to_consider = df_items_global_app.copy()
    if genre_filter and genre_filter != "All Genres" and hasattr(C, 'GENRES_COL') and C.GENRES_COL in items_to_consider.columns:
        items_to_consider = items_to_consider[
            items_to_consider[C.GENRES_COL].astype(str).str.contains(re.escape(genre_filter), case=False, na=False, regex=True)
        ]
        if items_to_consider.empty: return pd.DataFrame()
    if year_min_filter is not None and year_max_filter is not None and hasattr(C, 'RELEASE_YEAR_COL') and C.RELEASE_YEAR_COL in items_to_consider.columns:
        items_to_consider[C.RELEASE_YEAR_COL] = pd.to_numeric(items_to_consider[C.RELEASE_YEAR_COL], errors='coerce').fillna(0)
        items_to_consider = items_to_consider[
            (items_to_consider[C.RELEASE_YEAR_COL] >= year_min_filter) &
            (items_to_consider[C.RELEASE_YEAR_COL] <= year_max_filter)
        ]
        if items_to_consider.empty: return pd.DataFrame()
    if excluded_genre and hasattr(C, 'GENRES_COL') and C.GENRES_COL in items_to_consider.columns:
        items_to_consider = items_to_consider[
            ~items_to_consider[C.GENRES_COL].astype(str).str.contains(r'\b' + re.escape(excluded_genre) + r'\b', case=False, na=False, regex=True)
        ]
        if items_to_consider.empty: return pd.DataFrame()
    items_to_consider[C.VOTE_COUNT_COL] = pd.to_numeric(items_to_consider[C.VOTE_COUNT_COL], errors='coerce').fillna(0)
    items_to_consider[C.VOTE_AVERAGE_COL] = pd.to_numeric(items_to_consider[C.VOTE_AVERAGE_COL], errors='coerce').fillna(0)
    items_to_consider = items_to_consider[items_to_consider[C.VOTE_AVERAGE_COL] >= min_vote_average_initial]
    items_to_consider = items_to_consider[items_to_consider[C.VOTE_COUNT_COL] >= min_votes_initial]
    if items_to_consider.empty: return pd.DataFrame()
    novel_candidates_df = items_to_consider.sort_values(
        by=[C.VOTE_COUNT_COL, C.VOTE_AVERAGE_COL],
        ascending=[True, False]
    ).head(num_novel_candidates)
    if novel_candidates_df.empty: return pd.DataFrame()
    top_hidden_gems_df = novel_candidates_df.sort_values(
        by=C.VOTE_AVERAGE_COL,
        ascending=False
    ).head(n)
    cols_out = [C.ITEM_ID_COL, C.LABEL_COL, C.GENRES_COL, C.RELEASE_YEAR_COL, C.VOTE_AVERAGE_COL, C.VOTE_COUNT_COL, C.TMDB_ID_COL]
    return top_hidden_gems_df[[col for col in cols_out if col in top_hidden_gems_df.columns]]

# --- Fonctions d'affichage des bandeaux (repris de votre app(3).py) ---
def display_movie_carousel(carousel_id, carousel_title, movies_df,
                           enable_rating_for_user_id=None,
                           num_cards_to_show_at_once=CARDS_PER_ROW,
                           is_personalized=False):
    if movies_df.empty:
        st.info(f"No movies to display for carousel: {carousel_title}")
        return

    # Pagination logic
    if f'{carousel_id}_page' not in st.session_state:
        st.session_state[f'{carousel_id}_page'] = 0

    current_page = st.session_state[f'{carousel_id}_page']
    total_movies = len(movies_df)
    total_pages = math.ceil(total_movies / num_cards_to_show_at_once)

    start_index = current_page * num_cards_to_show_at_once
    end_index = start_index + num_cards_to_show_at_once
    movies_to_display_on_page = movies_df.iloc[start_index:end_index]

    title_col_ratio = 0.80
    button_col_ratio = (1.0 - title_col_ratio) / 2

    # Title only at the top
    st.markdown(f"<h3 style='color: #1E1E1E; margin-bottom: 10px;'>{carousel_title}</h3>", unsafe_allow_html=True)

    # --- Custom CSS for poster hover effect and card layout ---
    st.markdown("""
    <style>
    .movie-card-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
        background: #232323; /* Foncé */
        border-radius: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.13);
        padding: 0.5rem 0.5rem 0.7rem 0.5rem;
        margin-bottom: 0.5rem;
        min-height: 410px;
        max-width: 220px;
        min-width: 180px;
        height: 410px;
        position: relative;
        overflow: visible;
    }
    .movie-poster-hover {
        width: 150px;
        height: 225px;
        object-fit: cover;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.10);
        transition: transform 0.25s cubic-bezier(.4,2,.6,1), box-shadow 0.25s;
        margin-bottom: 0.5rem;
        margin-top: 0.5rem;
        background: #eaeaea;
        display: block;
    }
    .movie-poster-hover:hover {
        transform: scale(1.18);
        z-index: 10;
        box-shadow: 0 8px 24px rgba(0,0,0,0.18);
    }
    .movie-title-top {
        font-size: 1.15rem;
        font-weight: 700;
        color: #fff;
        text-align: center;
        margin-bottom: 0.2rem;
        margin-top: 0.2rem;
        min-height: 2.5em;
        line-height: 1.2em;
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
    }
    .movie-info-bottom {
        font-size: 0.98rem;
        color: #ccc;
        text-align: center;
        margin-top: 0.3rem;
        margin-bottom: 0.1rem;
        min-height: 2.2em;
    }
    .movie-score-badge {
        background: #ffe066;
        color: #222;
        border-radius: 8px;
        padding: 0.15em 0.6em;
        font-size: 1.05em;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.2em;
    }
    .movie-personal-score {
        background: #d0f5e8;
        color: #1a6c4e;
        border-radius: 8px;
        padding: 0.15em 0.6em;
        font-size: 1.05em;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.2em;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("""
    <style>
    .movie-card-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
        background: #f8f8f8;
        border-radius: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        padding: 0.5rem 0.5rem 0.7rem 0.5rem;
        margin-bottom: 0.5rem;
        min-height: 410px;
        max-width: 220px;
        min-width: 180px;
        height: 410px;
        position: relative;
        overflow: visible;
    }
    .movie-poster-hover {
        width: 150px;
        height: 225px;
        object-fit: cover;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.10);
        transition: transform 0.25s cubic-bezier(.4,2,.6,1), box-shadow 0.25s;
        margin-bottom: 0.5rem;
        margin-top: 0.5rem;
        background: #eaeaea;
        display: block;
    }
    .movie-poster-hover:hover {
        transform: scale(1.18);
        z-index: 10;
        box-shadow: 0 8px 24px rgba(0,0,0,0.18);
    }
    .movie-title-top {
        font-size: 1.15rem;
        font-weight: 700;
        color: #222;
        text-align: center;
        margin-bottom: 0.2rem;
        margin-top: 0.2rem;
        min-height: 2.5em;
        line-height: 1.2em;
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
    }
    .movie-info-bottom {
        font-size: 0.98rem;
        color: #444;
        text-align: center;
        margin-top: 0.3rem;
        margin-bottom: 0.1rem;
        min-height: 2.2em;
    }
    .movie-score-badge {
        background: #ffe066;
        color: #222;
        border-radius: 8px;
        padding: 0.15em 0.6em;
        font-size: 1.05em;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.2em;
    }
    .movie-personal-score {
        background: #d0f5e8;
        color: #1a6c4e;
        border-radius: 8px;
        padding: 0.15em 0.6em;
        font-size: 1.05em;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.2em;
    }
    </style>
    """, unsafe_allow_html=True)

    if not movies_to_display_on_page.empty:
        cols_cards = st.columns(num_cards_to_show_at_once)

        for idx, (_, movie_data) in enumerate(movies_to_display_on_page.iterrows()):
            with cols_cards[idx]:
                genres_display_text = str(movie_data.get(C.GENRES_COL, "N/A")).replace('|', ', ')
                title_text_plain = str(movie_data.get(C.LABEL_COL, "Unknown Title"))
                tmdb_id_val = movie_data.get(C.TMDB_ID_COL)
                movie_id_current = movie_data.get(C.ITEM_ID_COL)
                year_val = movie_data.get(C.RELEASE_YEAR_COL)
                year_display = int(year_val) if pd.notna(year_val) and year_val != 0 else "N/A"

                # --- Card HTML ---
                card_html = f"""<div class="movie-card-container">"""

                # Title at the top, large and bold, with link if possible
                if pd.notna(tmdb_id_val):
                    try:
                        title_link_url = f"https://www.themoviedb.org/movie/{int(tmdb_id_val)}"
                        card_html += f"""<div class="movie-title-top"><a href="{title_link_url}" target="_blank" style="color:inherit;text-decoration:none;">{title_text_plain}</a></div>"""
                    except ValueError:
                        card_html += f"""<div class="movie-title-top">{title_text_plain}</div>"""
                else:
                    card_html += f"""<div class="movie-title-top">{title_text_plain}</div>"""

                # Poster image (with hover effect)
                poster_path_for_st_image = poster_map.get(movie_id_current)
                if poster_path_for_st_image and os.path.exists(poster_path_for_st_image):
                    # Use base64 to embed image in HTML for hover effect
                    with open(poster_path_for_st_image, "rb") as img_file:
                        img_bytes = img_file.read()
                        img_b64 = base64.b64encode(img_bytes).decode()
                    card_html += f"""<img src="data:image/jpeg;base64,{img_b64}" class="movie-poster-hover" alt="Poster" />"""
                else:
                    # Fallback: empty rectangle with warning
                    card_html += f"""<div class="movie-poster-hover" style="display:flex;align-items:center;justify-content:center;background:#ddd;color:#888;font-size:0.95em;">No poster</div>"""

                # Info at the bottom
                card_html += f"""<div class="movie-info-bottom">{genres_display_text} | {year_display}</div>"""

                # Score
                if is_personalized and 'estimated_score' in movie_data and pd.notna(movie_data['estimated_score']):
                    score_display = f"{movie_data['estimated_score']:.1f}/5.0"
                    card_html += f"""<div class="movie-personal-score">For you: {score_display}</div>"""
                elif C.VOTE_AVERAGE_COL in movie_data and pd.notna(movie_data[C.VOTE_AVERAGE_COL]):
                    vote_avg_display = f"{movie_data[C.VOTE_AVERAGE_COL]:.1f}/10"
                    card_html += f"""<div class="movie-score-badge">TMDB: {vote_avg_display}</div>"""
                else:
                    card_html += f"""<div class="movie-score-badge">TMDB: N/A</div>"""

                card_html += "</div>"

                st.markdown(card_html, unsafe_allow_html=True)

                # --- Rating selectbox (below the card, not inside) ---
                if enable_rating_for_user_id:
                    rating_opts = [None, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
                    fmt_fn = lambda x: "Rate" if x is None else f"{x} ★"
                    clean_carousel_id_for_key = re.sub(r'\W+', '', carousel_id.lower())[:15]
                    rating_key = f"rating_{clean_carousel_id_for_key}_{str(movie_id_current)}_{current_page}_{str(enable_rating_for_user_id)}"

                    current_buffered_rating = st.session_state.logged_in_user_ratings_buffer.get(movie_id_current)
                    idx_rating = rating_opts.index(current_buffered_rating) if current_buffered_rating in rating_opts else 0

                    previous_rating_in_buffer = st.session_state.logged_in_user_ratings_buffer.get(movie_id_current)

                    user_rating_input = st.selectbox(
                        label="Your rating:",
                        options=rating_opts,
                        index=idx_rating,
                        format_func=fmt_fn,
                        key=rating_key
                    )

                    if user_rating_input != previous_rating_in_buffer:
                        if user_rating_input is not None:
                            st.session_state.logged_in_user_ratings_buffer[movie_id_current] = user_rating_input
                        elif movie_id_current in st.session_state.logged_in_user_ratings_buffer and user_rating_input is None:
                            del st.session_state.logged_in_user_ratings_buffer[movie_id_current]

    # Pagination buttons at the bottom
    if total_pages > 1:
        col_title, col_prev, col_next = st.columns([title_col_ratio, button_col_ratio, button_col_ratio])
        with col_prev:
            if st.button(":material/arrow_back_ios:", key=f"prev_{carousel_id}", use_container_width=True, disabled=(current_page == 0)):
                st.session_state[f'{carousel_id}_page'] -= 1
                st.rerun()
        with col_next:
            if st.button(":material/arrow_forward_ios:", key=f"next_{carousel_id}", use_container_width=True, disabled=(current_page >= total_pages - 1)):
                st.session_state[f'{carousel_id}_page'] += 1
                st.rerun()
    else:
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<hr style='margin-top: 0.5rem; margin-bottom: 1rem;'>", unsafe_allow_html=True)
    
        
# --- Session State (MODIFIÉ pour SVD++ fold-in) ---
if 'last_processed_radio_selection' not in st.session_state:st.session_state.last_processed_radio_selection = None
if 'active_page' not in st.session_state: st.session_state.active_page = "general"
if 'current_user_id' not in st.session_state: st.session_state.current_user_id = None
if 'new_user_ratings' not in st.session_state: st.session_state.new_user_ratings = {}
if 'new_user_name_input' not in st.session_state: st.session_state.new_user_name_input = ''
if 'last_selected_user_id' not in st.session_state: st.session_state.last_selected_user_id = None
# REMPLACÉ: 'instant_reco_model_new_user' par les facteurs inférés
if 'instant_reco_model_new_user' in st.session_state: # Nettoyage de l'ancienne clé si elle existe
    del st.session_state['instant_reco_model_new_user']
if 'new_user_inferred_factors' not in st.session_state: st.session_state.new_user_inferred_factors = None
if 'new_user_inferred_bias' not in st.session_state: st.session_state.new_user_inferred_bias = None
if 'new_user_id_generated' not in st.session_state: st.session_state.new_user_id_generated = None
if 'logged_in_user_ratings_buffer' not in st.session_state:st.session_state.logged_in_user_ratings_buffer = {}
if 'search_input_value' not in st.session_state: st.session_state.search_input_value = ""
if 'active_search_query' not in st.session_state: st.session_state.active_search_query = ""

# --- Sidebar setup (repris de votre app(3).py) ---
# Centrer et agrandir le logo dans la sidebar, positionné plus haut
# --- Sidebar toggle button ---

# Encadre la sidebar d'un encadré de couleur #F5CB5C
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] > div:first-child {
        background-color: #F5CB5C !important;
        border-radius: 18px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        padding: 16px 8px 16px 8px;
        margin: 8px 0 8px 0;
        border: 4px solid #242423 !important; /* Ajoute un contour épais noir */
    }
    </style>
    """,
    unsafe_allow_html=True
)
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-logo-container" style="display: flex; flex-direction: column; align-items: center; justify-content: flex-start; height: 140px; margin-bottom: 0px; margin-top: -60px;">
            <img src="https://fbi.cults3d.com/uploaders/20952150/illustration-file/421c5c91-423c-49af-bbd6-7f3839622ab0/pngwing.com-2022-02-20T081900.534.png" width="140">
        </div>
        """,
        unsafe_allow_html=True
    )
all_genres_list_sidebar = ["All Genres"]
if not df_items_global_app.empty and hasattr(C, 'GENRES_COL') and C.GENRES_COL in df_items_global_app.columns:
    try:
        genres_series = df_items_global_app[C.GENRES_COL].fillna('').astype(str); s_genres = genres_series.str.split('|').explode()
        unique_sidebar_genres = sorted([ g.strip() for g in s_genres.unique() if g.strip() and g.strip().lower() != '(no genres listed)' ])
        if unique_sidebar_genres: all_genres_list_sidebar.extend(unique_sidebar_genres)
    except Exception as e_g_sb: print(f"Sidebar error (genre list): {e_g_sb}"); st.sidebar.error("Error loading genres.")

# Custom style for the genre filter label

st.sidebar.markdown("""
<div style='display: flex; align-items: center; gap: 8px; margin-bottom: -25px; margin-top: -8px;'>
    <span style='font-weight:bold; color:#242423;'>Filter by genre:</span>
</div>
""", unsafe_allow_html=True)

selected_genre_sidebar = st.sidebar.selectbox(
    "", all_genres_list_sidebar, key="genre_filter_sb", format_func=lambda x: x
)

slider_min, slider_max, current_slider_val = 1900, pd.Timestamp.now().year, (1900, pd.Timestamp.now().year)
if not df_items_global_app.empty and hasattr(C, 'RELEASE_YEAR_COL') and C.RELEASE_YEAR_COL in df_items_global_app.columns:
    valid_years = pd.to_numeric(df_items_global_app[C.RELEASE_YEAR_COL], errors='coerce').dropna()
    if not valid_years.empty:
        calc_min, calc_max = int(valid_years.min()), int(valid_years.max())
        if calc_min <= calc_max and calc_min > 1800: slider_min = calc_min
        if calc_max <= pd.Timestamp.now().year + 5: slider_max = calc_max
        current_slider_val = (slider_min, slider_max)
if slider_max < slider_min: slider_max = slider_min

# Style rapproché et couleurs personnalisées pour la barre et les labels
st.sidebar.markdown("""
<div style='display: flex; align-items: center; gap: 8px; margin-bottom: -10px;'>
    <span style='font-weight:bold; color:#242423;'>Filter by year:</span>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown(
    """
    <style>
    /* Slider bar and handle color */
    div[data-baseweb="slider"] .rc-slider-track, 
    div[data-baseweb="slider"] .rc-slider-handle, 
    div[data-baseweb="slider"] .rc-slider-dot-active {
        background: #242423 !important;
        border-color: #242423 !important;
    }
    div[data-baseweb="slider"] .rc-slider-handle {
        box-shadow: 0 0 0 2px #24242333 !important;
    }
    /* Selected value color above the slider */
    div[data-baseweb="slider"] .rc-slider-tooltip-inner {
        background: #242423 !important;
        color: #fff !important;
        font-weight: bold;
        border-radius: 6px;
        border: none;
        box-shadow: none;
    }
    /* Remove extra margin above slider */
    div[data-baseweb="slider"] {
        margin-top: -25px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
selected_year_range_sidebar = st.sidebar.slider(
    "", min_value=slider_min, max_value=slider_max, value=current_slider_val, key="year_filter_sb"
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <style>
    @keyframes zoomInOut {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.2); }
    }
    @keyframes borderPulse {
        0%, 100% { box-shadow: 0 0 0 0 #F5CB5C; border-color: #F5CB5C; }
        50% { box-shadow: 0 0 12px 4px #F5CB5C99; border-color: #FFD700; }
    }
    .animated-header {
        animation: zoomInOut 2s ease-in-out infinite;
        font-weight: bold;
        font-size: 1.45rem;
        margin-bottom: 0.5rem;
        color: #333533;
        text-align: center;
    }
    .user-space-box {
        display: flex;
        justify-content: center;
        align-items: center;
        border: 2.5px solid #242423; /* Noir */
        border-radius: 10px;
        padding: 7px 0;
        margin-bottom: 10px;
        background: #fffbe6;
        animation: borderPulse 2.5s infinite;
        box-shadow: 0 2px 6px rgba(245,203,92,0.10);
        max-width: 92%;
        margin-left: auto;
        margin-right: auto;
    }
    .radio-container {
        border: 2px solid black;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 10px;
        background-color: #333533;
    }
    </style>
    """, unsafe_allow_html=True
)

# Affiche titre animé centré dans un encadré animé
st.sidebar.markdown(
    '<div class="user-space-box"><div class="animated-header">User Space</div></div>',
    unsafe_allow_html=True
)


# Mettre à jour user_opts en fonction des nouvelles exigences
user_opts = ["Explore Movies"]
can_sel_existing = df_ratings_global_app is not None and not df_ratings_global_app.empty and \
                    hasattr(C, 'USER_ID_COL') and C.USER_ID_COL in df_ratings_global_app.columns and \
                    not df_ratings_global_app[C.USER_ID_COL].empty
# MODIFICATION ICI: Changer le nom de l'option
if can_sel_existing: user_opts.append("Log In") # ANCIEN: "Log In (Existing Profile)"
user_opts.append("Create New Profile")

# Déterminer l'index par défaut pour le selectbox
default_index = 0
if st.session_state.active_page == "search_results":
    if st.session_state.get('last_processed_selectbox_selection', "Explore Movies") == "Explore Movies":
        default_index = user_opts.index("Explore Movies") if "Explore Movies" in user_opts else 0
    elif st.session_state.get('last_processed_selectbox_selection', "Explore Movies") == "Create New Profile":
        default_index = user_opts.index("Create New Profile") if "Create New Profile" in user_opts else 0
    # MODIFICATION ICI: Référence au nouveau nom
    elif st.session_state.get('last_processed_selectbox_selection', "Explore Movies") == "Log In": # ANCIEN: "Log In (Existing Profile)"
        default_index = user_opts.index("Log In") if "Log In" in user_opts else 0 # ANCIEN: "Log In (Existing Profile)"
    else:
        default_index = user_opts.index("Explore Movies") if "Explore Movies" in user_opts else 0
elif st.session_state.active_page == "general":
    default_index = user_opts.index("Explore Movies") if "Explore Movies" in user_opts else 0
elif st.session_state.active_page in ["new_user_profiling", "new_user_instant_recs"]:
    default_index = user_opts.index("Create New Profile") if "Create New Profile" in user_opts else 0
# MODIFICATION ICI: Référence au nouveau nom
elif st.session_state.active_page == "user_specific" and "Log In" in user_opts: # ANCIEN: "Log In (Existing Profile)"
    default_index = user_opts.index("Log In") # ANCIEN: "Log In (Existing Profile)"


# Création du menu déroulant (selectbox) avec moins d'écart
st.sidebar.markdown(
    """
    <div style='font-weight:bold; font-size:1.15em; margin-bottom: -20px; margin-top: 15px;'>Choose an option :</div>
    """,
    unsafe_allow_html=True
)
user_sel_opt = st.sidebar.selectbox(
    "",
    user_opts,
    key="user_sel_main_selectbox",
    index=default_index
)

# Détecter si l'utilisateur a fait un nouveau choix dans le menu déroulant
user_has_made_new_choice = False
if st.session_state.get('last_processed_selectbox_selection') != user_sel_opt:
    user_has_made_new_choice = True
    st.session_state.last_processed_selectbox_selection = user_sel_opt


if user_has_made_new_choice:
    intended_page_from_choice = st.session_state.active_page
    intended_uid_from_choice = st.session_state.current_user_id
    if user_sel_opt == "Explore Movies":
        intended_page_from_choice, intended_uid_from_choice = "general", None
    elif user_sel_opt == "Create New Profile":
        intended_page_from_choice = "new_user_profiling"
        intended_uid_from_choice = "new_user_temp"
        if st.session_state.active_page not in ["new_user_profiling", "new_user_instant_recs"]:
            st.session_state.new_user_ratings, st.session_state.new_user_name_input = {}, ''
            st.session_state.new_user_id_generated = None
            st.session_state.new_user_inferred_factors = None
            st.session_state.new_user_inferred_bias = None
    # MODIFICATION ICI: Référence au nouveau nom
    elif user_sel_opt == "Log In" and can_sel_existing: # ANCIEN: "Log In (Existing Profile)"
        intended_page_from_choice = "user_specific"
        if st.session_state.current_user_id is None or st.session_state.current_user_id == "new_user_temp":
            uids_list_for_default = sorted(df_ratings_global_app[C.USER_ID_COL].unique()) if df_ratings_global_app is not None and not df_ratings_global_app.empty else []
            last_id_sel = st.session_state.last_selected_user_id
            intended_uid_from_choice = last_id_sel if last_id_sel in uids_list_for_default else (uids_list_for_default[0] if uids_list_for_default else None)

    if st.session_state.active_page != intended_page_from_choice or st.session_state.current_user_id != intended_uid_from_choice:
        st.session_state.active_page = intended_page_from_choice
        st.session_state.current_user_id = intended_uid_from_choice
        if intended_page_from_choice != "search_results": st.session_state.search_input_value = ""; st.session_state.active_search_query = ""
        st.rerun()
        
# --- MODIFIED: Search bar in the main page ---
search_placeholder = st.empty() # Placeholder for the search bar container

with search_placeholder.container():
    # Only show search bar if not in new user profiling or instant recs page
    if st.session_state.active_page not in ["new_user_profiling", "new_user_instant_recs"]:
        # Custom CSS for search box border highlight
        st.markdown("""
        <style>
        .search-box-highlight input {
            border: 10px solid #F5CB5C !important;
            border-radius: 10px !important;
            box-shadow: 0 0 0 2px #24242333;
            background: #fffbe6 !important;
            font-size: 1.1em;
        }
        </style>
        """, unsafe_allow_html=True)
        search_col_1, search_col_2, search_col_3 = st.columns([0.5, 0.6, 0.5])
        with search_col_2:
            # Add a div with the custom class for the search box
            st.markdown('<div class="search-box-highlight">', unsafe_allow_html=True)
            search_text_input_main = st.text_input(
                "",
                value=st.session_state.search_input_value,
                key="movie_search_main_key",
                help="Enter part of the title and press Enter.",
                placeholder="Search for a movie"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        if search_text_input_main != st.session_state.search_input_value:
            st.session_state.search_input_value = search_text_input_main

            if st.session_state.search_input_value:
                st.session_state.active_search_query = st.session_state.search_input_value
                if st.session_state.active_page != "search_results":
                    st.session_state.active_page = "search_results"
                st.rerun()
            elif not st.session_state.search_input_value and st.session_state.active_page == "search_results":
                st.session_state.active_search_query = ""
                current_radio_selection = st.session_state.get('last_processed_radio_selection', "Explore Movies")
                if current_radio_selection == "Explore Movies":
                    st.session_state.active_page = "general"
                elif current_radio_selection == "Log In (Existing Profile)":
                    st.session_state.active_page = "user_specific"
                elif current_radio_selection == "Create New Profile":
                    st.session_state.active_page = "new_user_profiling"
                else:
                    st.session_state.active_page = "general"
                st.rerun()
        st.markdown(
            "<hr style='border-top: 3px solid #242423; margin-top: 1.5rem; margin-bottom: 1.5rem;'>",
            unsafe_allow_html=True
        )

# --- MODIFIED SECTION (Gestion de la sélection d'ID de profil) ---
uid_for_reco = None
user_profiles_map = {}
if st.session_state.active_page == "user_specific":
    # MODIFICATION ICI: Référence au nouveau nom
    if user_sel_opt == "Log In" and can_sel_existing: # ANCIEN: "Log In (Existing Profile)"
        user_profiles_path = C.DATA_PATH / getattr(C, 'USER_PROFILES_FILENAME', 'user_profiles.csv')
        if os.path.exists(user_profiles_path):
            try:
                df_profiles = pd.read_csv(user_profiles_path)
                if 'userId' in df_profiles.columns and 'userName' in df_profiles.columns:
                    user_profiles_map = pd.Series(df_profiles.userName.values, index=df_profiles.userId).to_dict()
            except Exception as e_pf: print(f"Error loading user_profiles.csv: {e_pf}")

        def disp_opts_func(uid_val):
            user_name = user_profiles_map.get(uid_val)
            if user_name:
                return f"ID {uid_val} - {user_name}"
            else:
                return f"ID {uid_val}"

        uids_from_ratings = df_ratings_global_app[C.USER_ID_COL].unique()
        user_sort_list = []
        for uid_val_loop in uids_from_ratings:
            actual_name = user_profiles_map.get(uid_val_loop)
            has_profile_name_sort_key = 0 if actual_name else 1
            # display_name_for_sort = f"{actual_name if actual_name else 'User'} (ID: {uid_val_loop})" # Plus nécessaire pour l'affichage, mais peut rester pour le tri
            user_sort_list.append({'uid': uid_val_loop, 'sort_key': has_profile_name_sort_key, 'display_text': str(uid_val_loop)}) # Utilise l'ID pour le tri si pas de nom

        # Tri par clé de tri (profil nommé ou non) puis par texte affiché (l'ID dans ce cas)
        user_sort_list.sort(key=lambda x: (x['sort_key'], x['display_text']))
        uids_avail = [user['uid'] for user in user_sort_list]

        if uids_avail:
            current_selection_uid = st.session_state.current_user_id
            if current_selection_uid not in uids_avail:
                current_selection_uid = uids_avail[0]
                st.session_state.current_user_id = current_selection_uid
            idx_sel_box = uids_avail.index(current_selection_uid)

            # --- DÉBUT DES MODIFICATIONS POUR LA BARRE DE RECHERCHE D'ID ---
            st.sidebar.markdown("---") # Séparateur pour la clarté

            # Input de recherche pour l'ID
            search_uid_query = st.sidebar.text_input(
                "Search User ID or Name:", # Nouveau label
                value=st.session_state.get('search_uid_input', ''),
                key="search_uid_input_key"
            )

            # Filtrer les UIDs disponibles en fonction de la recherche (ID ou nom)
            filtered_uids_avail = []
            if search_uid_query:
                query_lower = search_uid_query.lower()
                for uid in uids_avail:
                    # Recherche par ID (partielle)
                    if query_lower in str(uid).lower():
                        filtered_uids_avail.append(uid)
                    else:
                        # Recherche par nom (partielle et insensible à la casse)
                        user_name = user_profiles_map.get(uid)
                        if user_name and query_lower in user_name.lower():
                            filtered_uids_avail.append(uid)
            else:
                filtered_uids_avail = uids_avail  # Si la recherche est vide, affiche tous les UIDs

            # Assurez-vous que la sélection actuelle est toujours dans les options filtrées, sinon réinitialisez
            if current_selection_uid not in filtered_uids_avail and filtered_uids_avail:
                current_selection_uid = filtered_uids_avail[0]  # Sélectionne le premier ID filtré
                st.session_state.current_user_id = current_selection_uid
                st.session_state.last_selected_user_id = current_selection_uid  # Mise à jour
                st.rerun()  # Pour refléter le changement de sélection

            # Déterminer l'index pour le selectbox filtré
            idx_sel_box_filtered = filtered_uids_avail.index(current_selection_uid) if current_selection_uid in filtered_uids_avail else 0
            if not filtered_uids_avail:  # Si aucun ID ne correspond au filtre
                st.sidebar.warning("No matching User ID or Name found.")
                uid_sel_box_val = None
            else:
                uid_sel_box_val = st.sidebar.selectbox(
                    f"Select ID:",
                    options=filtered_uids_avail,  # Utilisez la liste filtrée
                    format_func=disp_opts_func,  # C'est ici que nous utilisons la fonction modifiée
                    index=idx_sel_box_filtered,  # Index pour la liste filtrée
                    key="uid_sel_box"
                )
            # --- FIN DES MODIFICATIONS POUR LA BARRE DE RECHERCHE D'ID ---

            if st.session_state.current_user_id != uid_sel_box_val and uid_sel_box_val is not None:
                st.session_state.current_user_id = uid_sel_box_val
                st.session_state.last_selected_user_id = uid_sel_box_val
                st.rerun()
            uid_for_reco = st.session_state.current_user_id
        else:
            st.sidebar.warning("No user ratings available to select a profile.")
            uid_for_reco = None
    elif st.session_state.current_user_id not in [None, "new_user_temp"]:
            uid_for_reco = st.session_state.current_user_id
                    
# --- User profile inference function (Fold-in) ---
def infer_new_user_profile(new_ratings_dict, model, n_epochs=10, lr=0.005, reg=0.02): #
    """
    Inference of latent factors (pu) and bias (bu) for a new user.
    new_ratings_dict: Dictionary {item_id: rating}
    model: Pre-trained global SVD++ model from Surprise
    """ # changed docstring
    if not new_ratings_dict: #
        return None, None #

    # Initialization
    n_factors = model.n_factors #
    pu_new = np.random.normal(0, .1, n_factors) # User factors #
    bu_new = 0.0  # User bias #
    global_mean = model.trainset.global_mean #

    # SGD iterations to optimize pu_new and bu_new
    for _ in range(n_epochs): #
        for item_id_str, rating in new_ratings_dict.items(): #
            try:
                # Convert item_id to inner_id of the global model's trainset
                item_inner_id = model.trainset.to_inner_iid(item_id_str) #
                
                # Get item factors and bias from the global model
                qi = model.qi[item_inner_id] #
                bi = model.bi[item_inner_id] #
                
                # Calculate error
                pred = global_mean + bu_new + bi + np.dot(pu_new, qi) #
                # For SVD++, the implicit term y_j is also needed.
                # To simplify, we can omit it for fold-in or assume it's absorbed.
                # A more complete version would consider items rated by the user
                # to calculate sum(y_j / sqrt(N(u)))
                # For now, we simplify without the y_j term for fold-in.

                err = rating - pred #
                
                # Update bu_new and pu_new
                bu_new += lr * (err - reg * bu_new) #
                pu_new += lr * (err * qi - reg * pu_new) #
            except ValueError: # Item is not in the global model's trainset (rare if df_items_global_app is the source)
                print(f"Warning: Item {item_id_str} rated by the new user is not in the global model's trainset.") # changed "Attention : L'item ... n'est pas dans le trainset du modèle global."
                continue #
    return pu_new, bu_new #

# --- Main Display Logic ---
if st.session_state.active_page == "general": #
    yr_min, yr_max = selected_year_range_sidebar[0], selected_year_range_sidebar[1] #
    genre_f_general = selected_genre_sidebar if selected_genre_sidebar != "All Genres" else None # changed "Tous les genres"
    genre_suffix = f" : {genre_f_general}" if genre_f_general else "" #
    top_tmdb_movies = get_top_overall_movies_tmdb(genre_filter=genre_f_general, year_min_filter=yr_min, year_max_filter=yr_max) #
    st.markdown(
        """
        <div style='text-align: center; font-size: 3.5em; font-weight: 900; letter-spacing: 1.5px; color: #222; margin-bottom: 18px; margin-top: -20px; text-shadow: 1px 1px 2px #F5CB5C, 0 2px 8px #fffbe6;'>
            Our recommendations and discoveries
        </div>
        """,
        unsafe_allow_html=True
    )
    display_movie_carousel("top_tmdb", f"Top Rated Movies{genre_suffix}", top_tmdb_movies) # changed "Films les Mieux Notés"
    top_documentaries = get_top_genre_movies_tmdb(genre="Documentary", year_min_filter=yr_min, year_max_filter=yr_max) #



    display_movie_carousel("top_documentaries", "Must-Watch Documentaries", top_documentaries) # changed "Documentaires Incontournables"
    hidden_gems = get_hidden_gems_movies(genre_filter=genre_f_general, year_min_filter=yr_min, year_max_filter=yr_max) #



    display_movie_carousel("Quiet Masterpieces", f"Quiet Masterpieces{genre_suffix}", hidden_gems) # changed "Pépites Cachées"

elif st.session_state.active_page == "user_specific" and uid_for_reco is not None: #
    user_display_name_map_val = user_profiles_map.get(uid_for_reco, f"User {uid_for_reco}") # changed "Utilisateur"
    st.header(f"Recommendations For You : {user_display_name_map_val}") # changed "Recommandations Pour Vous,"
    yr_min_p, yr_max_p = selected_year_range_sidebar[0], selected_year_range_sidebar[1] #
    genre_f_perso = selected_genre_sidebar if selected_genre_sidebar != "All Genres" else None # changed "Tous les genres"
    models_p_dir = str(C.DATA_PATH / 'recs') # Path to pre-calculated models for existing users #
    avail_model_files = [f for f in os.listdir(models_p_dir) if f.endswith('.p') and not 'personalized' in f.lower() and not model_path in f] if os.path.exists(models_p_dir) and os.path.isdir(models_p_dir) else [] #
    if not avail_model_files: st.error(f"No general pre-trained models found in {models_p_dir} (excluding global SVD++).") # changed "Aucun modèle général pré-entraîné trouvé dans ... (hors SVD++ global)."
    else:
        user_profile_for_titles = explanations.get_user_profile_for_explanation(uid_for_reco, top_n_movies=2, min_rating=3.5) #
        model_types_config = [ #
            ("content_based", "content_based", "Content-Based Suggestions"), # changed "Suggestions Basées sur le Contenu"
            ("user_based", "user_based", "Liked by Similar Profiles"), # changed "Aimé par des Profils Similaires"
            ("svd", "svd", "Algorithmic Discoveries For You") # changed "Découvertes Algorithmiques Pour Vous"
        ] #
        for model_key, file_keyword, fallback_carousel_title in model_types_config: #
            m_file = next((mfile for mfile in avail_model_files if file_keyword in mfile.lower() and 'final' in mfile.lower()), None) #
            if not m_file: m_file = next((mfile for mfile in avail_model_files if file_keyword in mfile.lower()), None) #
            if m_file: #
                recs_data = recommender.get_top_n_recommendations(
                    uid_for_reco, m_file, n=N_RECOS_PERSONNALISEES_TOTAL_FETCH,
                    filter_genre=genre_f_perso, filter_year_range=(yr_min_p, yr_max_p)
                ) #
                carousel_title_final = fallback_carousel_title # Title logic unchanged #
                if model_key == "content_based": #
                    if user_profile_for_titles: #
                        anchor_movie = user_profile_for_titles[0] #
                        carousel_title_final = f"""<span style="font-family: 'Poppins', cursive; font-size:0.8em;">Because you liked {anchor_movie['title']} ({anchor_movie['rating']:.1f}/5):</span>""" # Poppins font, reduced size

                        # Inject Google Fonts Poppins if not already present
                        st.markdown("""
                        <link href="https://fonts.googleapis.com/css?family=Poppins&display=swap" rel="stylesheet">
                        <style>
                        .poppins-font {
                            font-family: 'Poppins', cursive !important;
                            font-size: 0.6em !important;
                        }
                        </style>
                        """, unsafe_allow_html=True)
                elif model_key == "user_based": #
                    if len(user_profile_for_titles) >= 2: #
                        movie1, movie2 = user_profile_for_titles[0], user_profile_for_titles[1] #
                        carousel_title_final = f"<span style='font-family: Poppins, cursive; font-size:0.8em;'>Fans of {movie1['title']} and {movie2['title']} also enjoy:</span>" #
                    elif len(user_profile_for_titles) == 1: #
                        movie1 = user_profile_for_titles[0] #
                        carousel_title_final = f"<span style='font-family: Poppins, cursive; font-size:0.8em;'>Fans of {movie1['title']} ({movie1['rating']:.1f}/5) also enjoy:</span>" #
                elif model_key == "svd": #
                    carousel_title_final = f"""<span style="font-family: 'Poppins', cursive; font-size:0.8em;">Based on your overall behavior...</span>"""  # Poppins font, reduced size

                    # Inject Google Fonts Poppins if not already present
                    st.markdown("""
                    <link href="https://fonts.googleapis.com/css?family=Poppins&display=swap" rel="stylesheet">
                    <style>
                    .poppins-font {
                        font-family: 'Poppins', cursive !important;
                        font-size: 0.6em !important;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                carousel_id_perso = f"{model_key}_{str(uid_for_reco).replace('.', '_')}" #
                display_movie_carousel(
                    carousel_id_perso, carousel_title_final, recs_data,
                    enable_rating_for_user_id=uid_for_reco, is_personalized=True
                ) #
            else: st.warning(f"No model of type '{file_keyword}' found.") # changed "Aucun modèle de type ... trouvé."
        if st.session_state.logged_in_user_ratings_buffer: # Rating saving logic unchanged #
            st.markdown("---") #
            num_buffered_ratings = len(st.session_state.logged_in_user_ratings_buffer) #
            cols_save_button = st.columns([0.3, 0.4, 0.3]) #
            with cols_save_button[1]: #
                 if st.button(f"✔️ Save my {num_buffered_ratings} new rating(s)", key="save_logged_in_ratings_final", use_container_width=True): # changed "Enregistrer mes ... nouvelle(s) note(s)"
                    # ... (saving code unchanged) ...
                    ratings_to_save_list = [] #
                    current_ts = int(time.time()) #
                    user_id_to_save = uid_for_reco #
                    for movie_id_key, rating_val_key in st.session_state.logged_in_user_ratings_buffer.items(): #
                        ratings_to_save_list.append({ #
                            C.USER_ID_COL: user_id_to_save, C.ITEM_ID_COL: movie_id_key, #
                            C.RATING_COL: rating_val_key, C.TIMESTAMP_COL: current_ts #
                        }) #
                    if ratings_to_save_list: #
                        df_new_ratings_to_save_out = pd.DataFrame(ratings_to_save_list) #
                        pending_ratings_filepath = C.EVIDENCE_PATH / getattr(C, 'NEW_RATINGS_PENDING_FILENAME', 'new_ratings_pending.csv') #
                        file_exists_pending = os.path.exists(pending_ratings_filepath) #
                        try:
                            df_new_ratings_to_save_out.to_csv(pending_ratings_filepath, mode='a', header=not file_exists_pending, index=False) #
                            st.success(f"{len(ratings_to_save_list)} rating(s) saved!") # changed "note(s) enregistrée(s) !"
                            st.session_state.logged_in_user_ratings_buffer = {} #
                            st.rerun() #
                        except Exception as e_save_rating: #
                            st.error(f"Error saving ratings: {e_save_rating}") # changed "Erreur lors de l'enregistrement des notes :"

elif st.session_state.active_page == "new_user_profiling": #
    search_placeholder.empty() #
    st.markdown("""
        <style>
        @keyframes zoomInOut {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.12); }
        }
        .animated-create-profile-header {
            animation: zoomInOut 2.2s ease-in-out infinite;
            font-family: 'Poppins', sans-serif !important;
            font-size: 2.5em !important;
            font-weight: 900;
            color: #222;
            text-align: center;
            margin-bottom: 1.0em;
            margin-top: 0.2em;
            letter-spacing: 1.5px;
        }
        </style>
        <link href="https://fonts.googleapis.com/css?family=Poppins:700&display=swap" rel="stylesheet">
        <div class="animated-create-profile-header">Create Your Profile Here !</div>
    """, unsafe_allow_html=True)
    st.write("To help us understand your preferences, please enter your name and rate some movies.") # changed "Pour nous aider à comprendre vos préférences, veuillez entrer votre nom et noter quelques films."
    # Réduit la taille de la boîte de saisie du nom avec du CSS personnalisé
    st.markdown("""
        <style>
        .small-name-input input {
            width: 220px !important;
            font-size: 1em !important;
            padding: 6px 8px !important;
            border-radius: 7px !important;
        }
        </style>
        <div class="small-name-input">
    """, unsafe_allow_html=True)
    new_user_name = st.text_input("What is your name ?", st.session_state.get('new_user_name_input', ''), key="new_user_name_input_box")
    st.markdown("</div>", unsafe_allow_html=True)
    st.session_state.new_user_name_input = new_user_name

    # Movie selection logic for profiling (unchanged)
    movies_for_profiling_pool_initial = df_items_global_app.copy() if not df_items_global_app.empty else pd.DataFrame() #
    sample_size = 20 ; min_prefs_needed = 5 ; movies_to_display_df = pd.DataFrame() #
    if not movies_for_profiling_pool_initial.empty: #
        # ... (movie selection code unchanged) ...
        num_candidate_popular_movies = 500; movies_per_genre_target = 1; candidate_pool_df = pd.DataFrame() #
        if C.VOTE_COUNT_COL in movies_for_profiling_pool_initial.columns: #
            movies_for_profiling_pool_initial[C.VOTE_COUNT_COL] = pd.to_numeric(movies_for_profiling_pool_initial[C.VOTE_COUNT_COL], errors='coerce').fillna(0) #
            candidate_pool_df = movies_for_profiling_pool_initial.sort_values(by=C.VOTE_COUNT_COL, ascending=False).head(num_candidate_popular_movies) #
        if candidate_pool_df.empty: #
            if C.POPULARITY_COL in movies_for_profiling_pool_initial.columns and not movies_for_profiling_pool_initial[C.POPULARITY_COL].isnull().all(): #
                movies_for_profiling_pool_initial[C.POPULARITY_COL] = pd.to_numeric(movies_for_profiling_pool_initial[C.POPULARITY_COL], errors='coerce').fillna(0) #
                candidate_pool_df = movies_for_profiling_pool_initial.sort_values(by=C.POPULARITY_COL, ascending=False).head(num_candidate_popular_movies) #
            elif len(movies_for_profiling_pool_initial) > 0: #
                candidate_pool_df = movies_for_profiling_pool_initial.sample(n=min(num_candidate_popular_movies, len(movies_for_profiling_pool_initial)), random_state=42) #
        selected_movies_for_profiling_list = [] ; selected_movie_ids = set() #
        if not candidate_pool_df.empty and C.GENRES_COL in candidate_pool_df.columns: #
            genres_df_for_selection = candidate_pool_df.copy() #
            genres_df_for_selection['genre_list'] = genres_df_for_selection[C.GENRES_COL].astype(str).str.split('|') #
            genres_exploded_df = genres_df_for_selection.explode('genre_list') #
            genres_exploded_df['genre_list'] = genres_exploded_df['genre_list'].str.strip() #
            genres_exploded_df = genres_exploded_df[genres_exploded_df['genre_list'].notna() & (genres_exploded_df['genre_list'] != '') & (genres_exploded_df['genre_list'].str.lower() != '(no genres listed)')] #
            unique_genres_in_pool = genres_exploded_df['genre_list'].unique() ; random.shuffle(unique_genres_in_pool) #
            for genre in unique_genres_in_pool: #
                if len(selected_movies_for_profiling_list) >= sample_size: break #
                movies_in_genre_from_exploded = genres_exploded_df[genres_exploded_df['genre_list'] == genre] #
                sort_col_for_genre_picking = C.VOTE_COUNT_COL if C.VOTE_COUNT_COL in movies_in_genre_from_exploded else C.POPULARITY_COL #
                if sort_col_for_genre_picking in movies_in_genre_from_exploded: movies_in_genre_sorted = movies_in_genre_from_exploded.sort_values(by=sort_col_for_genre_picking, ascending=False) #
                else: movies_in_genre_sorted = movies_in_genre_from_exploded #
                added_for_this_genre = 0 #
                for _, movie_data_from_exploded in movies_in_genre_sorted.iterrows(): #
                    movie_id = movie_data_from_exploded[C.ITEM_ID_COL] #
                    if movie_id not in selected_movie_ids: #
                        full_movie_row_df = candidate_pool_df[candidate_pool_df[C.ITEM_ID_COL] == movie_id] #
                        if not full_movie_row_df.empty: #
                            selected_movies_for_profiling_list.append(full_movie_row_df.iloc[0].to_dict()) ; selected_movie_ids.add(movie_id) ; added_for_this_genre += 1 #
                            if added_for_this_genre >= movies_per_genre_target or len(selected_movies_for_profiling_list) >= sample_size: break #
        if len(selected_movies_for_profiling_list) < sample_size and not candidate_pool_df.empty: #
            remaining_candidates = candidate_pool_df[~candidate_pool_df[C.ITEM_ID_COL].isin(selected_movie_ids)] ; needed = sample_size - len(selected_movies_for_profiling_list) #
            for _, movie_row_to_add in remaining_candidates.head(needed).iterrows(): #
                if movie_row_to_add[C.ITEM_ID_COL] not in selected_movie_ids: #
                    selected_movies_for_profiling_list.append(movie_row_to_add.to_dict()) ; selected_movie_ids.add(movie_row_to_add[C.ITEM_ID_COL]) #
        if selected_movies_for_profiling_list: #
            movies_to_display_df = pd.DataFrame(selected_movies_for_profiling_list) #
            if len(movies_to_display_df) > sample_size: movies_to_display_df = movies_to_display_df.sample(n=sample_size, random_state=43) #
        else:
            st.warning("Diversified movie selection could not be performed. Using global popularity selection mode.") # changed "La sélection diversifiée de films n'a pas pu être effectuée. Utilisation du mode de sélection par popularité globale."
            # ... (fallback unchanged) ...
            if hasattr(C, 'POPULARITY_COL') and C.POPULARITY_COL in movies_for_profiling_pool_initial.columns and not movies_for_profiling_pool_initial[C.POPULARITY_COL].isnull().all(): #
                movies_for_profiling_pool_initial[C.POPULARITY_COL] = pd.to_numeric(movies_for_profiling_pool_initial[C.POPULARITY_COL], errors='coerce').fillna(0) #
                movies_to_display_df_temp = movies_for_profiling_pool_initial.sort_values(by=C.POPULARITY_COL, ascending=False).head(150) #
                if len(movies_to_display_df_temp) >= sample_size: movies_to_display_df = movies_to_display_df_temp.sample(n=sample_size, random_state=42) #
            if movies_to_display_df.empty: #
                if len(movies_for_profiling_pool_initial) >= sample_size: movies_to_display_df = movies_for_profiling_pool_initial.sample(n=sample_size, random_state=42) #
                elif not movies_for_profiling_pool_initial.empty : movies_to_display_df = movies_for_profiling_pool_initial.copy() #

    if movies_to_display_df.empty: st.error("Could not load movies for profiling.") # changed "Impossible de charger les films pour le profilage."
    else:
        movies_to_display_final_unique = movies_to_display_df.drop_duplicates(subset=[C.ITEM_ID_COL]) #
        if len(movies_to_display_final_unique) > sample_size: #
             movies_to_display_final_unique = movies_to_display_final_unique.sample(n=sample_size, random_state=44) #

        with st.form(key="new_user_profiling_form"): #
            st.subheader("Rate the following movies") # changed "Notez les films suivants"
            st.caption("Leave as 'No rating' if you don't know the movie or have no opinion.") # changed "Laissez 'Pas de note' si vous ne connaissez pas le film ou n'avez pas d'opinion."
            for _, row in movies_to_display_final_unique.iterrows(): #
                movie_id, title = row[C.ITEM_ID_COL], row[C.LABEL_COL] #
                rating_key = f"rating_select_{movie_id}" #
                current_rating = st.session_state.new_user_ratings.get(movie_id) #
                col1, col2 = st.columns([3,2]) #
                with col1: #
                    st.markdown(f"**{title}**") #
                    if hasattr(C, 'GENRES_COL') and C.GENRES_COL in row and pd.notna(row[C.GENRES_COL]): #
                        genres_display_text_profiling = str(row[C.GENRES_COL]).replace('|', ', ') #
                        st.caption(f"Genres: {genres_display_text_profiling}") #
                with col2: #
                    opts = [None, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0] #
                    fmt_fn = lambda x: "No rating" if x is None else f"{x} ★" # changed "Pas de note"
                    idx = opts.index(current_rating) if current_rating in opts else 0 #
                    new_r = st.selectbox(f"Rating for {title}:", opts, index=idx, format_func=fmt_fn, key=rating_key, label_visibility="collapsed") # changed "Note pour {title}:"
                    if new_r is not None: st.session_state.new_user_ratings[movie_id] = new_r #
                    elif movie_id in st.session_state.new_user_ratings: del st.session_state.new_user_ratings[movie_id] #
            form_submitted = st.form_submit_button("✔️ Save and See Initial Suggestions") # changed "Enregistrer et Voir les Suggestions Initiales"

        final_prefs = st.session_state.new_user_ratings.copy() #
        num_total_prefs = len(final_prefs) #
        st.info(f"You have provided {num_total_prefs} rating(s). At least {min_prefs_needed} are required.") # changed "Vous avez fourni ... note(s). Au moins ... sont requises."
        if form_submitted: #
            if not new_user_name.strip(): st.warning("Please enter your name.") # changed "Veuillez entrer votre nom."
            elif num_total_prefs < min_prefs_needed: st.warning(f"Please rate at least {min_prefs_needed} movies.") # changed "Veuillez noter au moins ... films."
            elif global_svdpp_model is None: # Check if global model is loaded #
                st.error("The global recommendation model is not available. Cannot generate suggestions.") # changed "Le modèle de recommandation global n'est pas disponible. Impossible de générer des suggestions."
            else:
                try:
                    # Generate a new user ID
                    current_max_user_id = 0 #
                    if not df_ratings_global_app.empty and C.USER_ID_COL in df_ratings_global_app.columns: #
                        current_max_user_id = df_ratings_global_app[C.USER_ID_COL].max() #
                    
                    # Also check IDs in user_profiles.csv to avoid conflicts
                    user_profiles_filepath = C.DATA_PATH / getattr(C, 'USER_PROFILES_FILENAME', 'user_profiles.csv') #
                    if os.path.exists(user_profiles_filepath): #
                        try:
                            df_profiles_temp = pd.read_csv(user_profiles_filepath) #
                            if 'userId' in df_profiles_temp.columns and not df_profiles_temp.empty: #
                                current_max_user_id = max(current_max_user_id, df_profiles_temp['userId'].max()) #
                        except pd.errors.EmptyDataError: #
                            pass # File is empty, no IDs to check #
                        except Exception as e_pf_read: #
                            print(f"Warning: Could not read {user_profiles_filepath} to check IDs: {e_pf_read}") # changed "Avertissement: Impossible de lire ... pour vérifier les IDs:"


                    new_user_id_val = current_max_user_id + 1 #
                    st.session_state.new_user_id_generated = new_user_id_val #

                    # Save ratings and profile (unchanged)
                    ratings_to_save_list = [] ; current_ts = int(time.time()) #
                    for movie_id_key, rating_val_key in final_prefs.items(): #
                        ratings_to_save_list.append({C.USER_ID_COL: new_user_id_val, C.ITEM_ID_COL: movie_id_key, C.RATING_COL: rating_val_key, C.TIMESTAMP_COL: current_ts}) #
                    if ratings_to_save_list: #
                        df_new_ratings_to_save_out = pd.DataFrame(ratings_to_save_list) #
                        pending_ratings_filepath = C.EVIDENCE_PATH / getattr(C, 'NEW_RATINGS_PENDING_FILENAME', 'new_ratings_pending.csv') #
                        file_exists_pending = os.path.exists(pending_ratings_filepath) #
                        df_new_ratings_to_save_out.to_csv(pending_ratings_filepath, mode='a', header=not file_exists_pending, index=False) #
                        
                        user_profile_data_out = pd.DataFrame([{'userId': new_user_id_val, 'userName': new_user_name.strip()}]) #
                        file_exists_profiles = os.path.exists(user_profiles_filepath) #
                        user_profile_data_out.to_csv(user_profiles_filepath, mode='a', header=not file_exists_profiles, index=False) #
                        st.success(f"Profile for {new_user_name.strip()} (ID: {new_user_id_val}) saved with {len(ratings_to_save_list)} ratings.") # changed "Profil pour ... enregistré avec ... notes."

                    # --- MODIFIED: User profile inference (Fold-in) ---
                    st.info("Calculating your first personalized suggestions...") # changed "Calcul de vos premières suggestions personnalisées..."
                    
                    # New user's ratings are in final_prefs {item_id: rating}
                    # item_id must be the raw_id (from your CSV files)
                    pu_new, bu_new = infer_new_user_profile(final_prefs, global_svdpp_model) #
                    
                    if pu_new is not None and bu_new is not None: #
                        st.session_state.new_user_inferred_factors = pu_new #
                        st.session_state.new_user_inferred_bias = bu_new #
                        st.session_state.active_page = "new_user_instant_recs" #
                        st.rerun() #
                    else:
                        # This specific error message was already in English in the source file, but I'm including it for completeness if it needed translation.
                        # The original message in the *uploaded app.py* (line 716) seems to be what the user cited as problematic.
                        # If the user's actual current file has "Impossible d'inférer le profil utilisateur. Veuillez réessayer.", then this would be the translation.
                        # Assuming the provided snippet's "Impossible d'inférer le profil utilisateur. Veuillez réessayer." is the one to translate.
                        st.error("Could not infer user profile. Please try again.") # changed "Impossible d'inférer le profil utilisateur. Veuillez réessayer."

                except Exception as e_profile_processing: #
                    st.error(f"Error creating your profile: {e_profile_processing}") # changed "Erreur lors de la création de votre profil :"
                    st.exception(e_profile_processing) #

elif st.session_state.active_page == "new_user_instant_recs": #
    search_placeholder.empty() #
    st.markdown("""
        <style>
        @keyframes zoomInOut {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.18); }
        }
        .animated-instant-recs-header {
            animation: zoomInOut 2.2s ease-in-out infinite;
            font-family: 'Poppins', sans-serif !important;
            font-size: 3.2em !important;
            font-weight: 900;
            color: #222;
            text-align: center;
            margin-bottom: 1.0em;
            margin-top: 0.2em;
            letter-spacing: 1.5px;
        }
        </style>
        <link href="https://fonts.googleapis.com/css?family=Poppins:700&display=swap" rel="stylesheet">
        <div class="animated-instant-recs-header">Your First Movie Suggestions Just Here !</div>
    """, unsafe_allow_html=True)
    st.markdown(
        "<span style='font-family: Poppins, sans-serif; font-weight: bold; font-size: 1.15em;'>Based on the ratings you just provided.</span>",
        unsafe_allow_html=True
    )

    pu_new = st.session_state.get('new_user_inferred_factors') #
    bu_new = st.session_state.get('new_user_inferred_bias') #
    new_user_ratings_keys = st.session_state.get('new_user_ratings', {}).keys() # Already rated items #

    if global_svdpp_model is None: #
        st.error("The global recommendation model is not available.") # changed "Le modèle de recommandation global n'est pas disponible."
    elif pu_new is None or bu_new is None: #
        st.error("Inferred user profile not available. Please return to profile creation.") # changed "Le profil utilisateur inféré n'est pas disponible. Veuillez retourner à la création de profil."
    elif models_df_items_global is None or models_df_items_global.empty: #
         st.warning("Item data is not available to generate suggestions.") # changed "Les données des items ne sont pas disponibles pour générer des suggestions."
    else:
        all_movie_ids_global_raw = df_items_global_app[C.ITEM_ID_COL].unique() #
        # Exclude movies already rated by the new user
        movies_to_predict_raw_ids = [mid for mid in all_movie_ids_global_raw if mid not in new_user_ratings_keys] #

        if not movies_to_predict_raw_ids: #
            st.info("No other movies to suggest at the moment (all known movies have been rated or no movies available).") # changed "Aucun autre film à suggérer pour le moment (tous les films connus ont été notés ou aucun film disponible)."
        else:
            preds_list_instant = [] #
            global_mean = global_svdpp_model.trainset.global_mean #
            
            # For SVD++, there is also the implicit term sum(y_j / sqrt(N(u)))
            # To calculate it, we need the items rated by the user (new_user_ratings_keys)
            # and the y_j factors from the global model.
            implicit_feedback_term = np.zeros(global_svdpp_model.n_factors) #
            num_rated_items = 0 #
            if hasattr(global_svdpp_model, 'yj'): # Check if the model has yj (SVD++) #
                for item_id_str in new_user_ratings_keys: #
                    try:
                        item_inner_id = global_svdpp_model.trainset.to_inner_iid(item_id_str) #
                        implicit_feedback_term += global_svdpp_model.yj[item_inner_id] #
                        num_rated_items +=1 #
                    except ValueError: # Item is not in the global trainset #
                        pass #
                if num_rated_items > 0: #
                    implicit_feedback_term /= np.sqrt(num_rated_items) #


            for item_raw_id in random.sample(movies_to_predict_raw_ids, min(len(movies_to_predict_raw_ids), 500)): # Predict on a sample #
                try:
                    item_inner_id = global_svdpp_model.trainset.to_inner_iid(item_raw_id) #
                    bi = global_svdpp_model.bi[item_inner_id] #
                    qi = global_svdpp_model.qi[item_inner_id] #
                    
                    # SVD++ Prediction: mu + bu + bi + q_i^T * (p_u + |N(u)|^(-1/2) * sum_{j in N(u)} y_j)
                    pred_score = global_mean + bu_new + bi + np.dot(qi, pu_new + implicit_feedback_term) #
                    
                    preds_list_instant.append({C.ITEM_ID_COL: item_raw_id, 'estimated_score': pred_score}) #
                except ValueError: # Item is not in the global model's trainset #
                    # print(f"Item {item_raw_id} not found in global trainset for instant prediction.")
                    continue #
                except Exception as e_pred_inst: #
                    print(f"Error during instant prediction for item {item_raw_id}: {e_pred_inst}") # changed "Erreur de prédiction instantanée pour item ..."


            if preds_list_instant: #
                recs_instant_df_raw = pd.DataFrame(preds_list_instant).sort_values(by='estimated_score', ascending=False).head(N_INSTANT_RECOS_NEW_USER * 2) # Take a bit more for merging #
                if not recs_instant_df_raw.empty and not df_items_global_app.empty: #
                    final_recs_instant_df = pd.merge(
                        recs_instant_df_raw,
                        df_items_global_app[[C.ITEM_ID_COL, C.LABEL_COL, C.GENRES_COL, C.RELEASE_YEAR_COL, C.VOTE_AVERAGE_COL, C.TMDB_ID_COL]],
                        on=C.ITEM_ID_COL, how='left'
                    ).head(N_INSTANT_RECOS_NEW_USER) #
                    display_movie_carousel("instant_recs_new_user", "Quick suggestions for you:", final_recs_instant_df, is_personalized=True) # changed "Suggestions rapides pour vous :"
                elif not recs_instant_df_raw.empty: #
                    st.write(recs_instant_df_raw.head(N_INSTANT_RECOS_NEW_USER)) # Fallback if merge fails #
                else:
                    st.info("Could not generate instant suggestions (after filtering).") # changed "Impossible de générer des suggestions instantanées (après filtrage)."
            else:
                st.info("No instant suggestions could be generated.") # changed "Aucune suggestion instantanée n'a pu être générée."
    
    if st.button("Explore other movies"): # changed "Explorer d'autres films"
        st.session_state.active_page = "general" #
        st.session_state.new_user_ratings, st.session_state.new_user_name_input = {}, '' #
        st.session_state.new_user_id_generated = None #
        st.session_state.new_user_inferred_factors = None #
        st.session_state.new_user_inferred_bias = None #
        st.session_state.search_input_value = "" #
        st.session_state.active_search_query = "" #
        st.rerun() #

elif st.session_state.active_page == "search_results": # Search section (unchanged) #
    query = st.session_state.active_search_query #
    st.header(f"Search Results for : \"{query}\"") # changed "Résultats de Recherche pour :"
    if not query: #
        st.warning("Please enter a term in the search bar above.") # changed "Veuillez entrer un terme dans la barre de recherche ci-dessus."
        if st.button("Back to Home"): # changed "Retour à l'Accueil"
            st.session_state.active_page = "general"; st.session_state.search_input_value = ""; st.session_state.active_search_query = ""; st.rerun() #
    else:
        results_df = df_items_global_app[df_items_global_app[C.LABEL_COL].str.contains(re.escape(query), case=False, na=False)] #
        if results_df.empty: st.info(f"No movies found containing \"{query}\".") # changed "Aucun film trouvé contenant ..."
        else:
            st.markdown(f"**{len(results_df)} movie(s) found:**") # changed "film(s) trouvé(s) :"
            for _, movie_data in results_df.iterrows():
                movie_id = movie_data[C.ITEM_ID_COL]; title = movie_data[C.LABEL_COL]
                genres_text_search = str(movie_data.get(C.GENRES_COL, "N/A")).replace('|', ', ')
                year_val = movie_data.get(C.RELEASE_YEAR_COL)
                year_display = int(year_val) if pd.notna(year_val) and year_val != 0 else "N/A"

                with st.container(border=True):
                    # Créez de nouvelles colonnes pour le poster et le reste des informations
                    # Ratio: une petite colonne pour le poster, une plus grande pour le texte et la notation
                    col_poster, col_info_and_rating = st.columns([0.15, 0.85])

                    with col_poster:
                        # --- Affichage du Poster ici ---
                        poster_path_for_st_image = poster_map.get(movie_id)
                        if poster_path_for_st_image and os.path.exists(poster_path_for_st_image):
                            st.markdown(
                                f"""
                                <div style="width:100%;display:flex;justify-content:center;align-items:center;">
                                    <img src="data:image/jpeg;base64,{base64.b64encode(open(poster_path_for_st_image, 'rb').read()).decode()}" style="width:100%;height:auto;object-fit:cover;border-radius:10px;"/>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                """
                                <div style="width:100%;display:flex;justify-content:center;align-items:center;">
                                    <img src="https://via.placeholder.com/200x300?text=No+Poster" style="width:100%;height:auto;object-fit:cover;border-radius:10px;"/>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                    with col_info_and_rating:
                        # Déplacez les colonnes d'information et de notation à l'intérieur de cette nouvelle colonne
                        cols_info, cols_rating_form = st.columns([0.6, 0.4]) # Ces colonnes restent comme avant

                        with cols_info:
                            st.subheader(f"{title}")
                            st.caption(f"Genres: {genres_text_search} | Year: {year_display}")
                            tmdb_id_val = movie_data.get(C.TMDB_ID_COL)
                            # Place the "More details" link at the bottom of the card/info section
                            more_details_html = ""
                            if pd.notna(tmdb_id_val):
                                try:
                                    title_link = f"https://www.themoviedb.org/movie/{int(tmdb_id_val)}"
                                    more_details_html = f"""
                                        <div style='width: 100%; text-align: left; margin-top: 18px;'>
                                            <a href='{title_link}' target='_blank' style='font-size:0.95em; color: #1a73e8; text-decoration: underline;'>More informations about the movie</a>
                                        </div>
                                    """
                                except ValueError:
                                    pass
                                    
                            st.markdown(more_details_html, unsafe_allow_html=True)
                        with cols_rating_form: # Rating logic in search (unchanged)
                            st.markdown("##### Rate this movie:")
                            rating_key_sr = f"search_rating_{movie_id}_{re.sub(r'[^a-zA-Z0-9]', '', query)}"
                            userid_key_sr = f"search_userid_{movie_id}_{re.sub(r'[^a-zA-Z0-9]', '', query)}"
                            button_key_sr = f"search_save_{movie_id}_{re.sub(r'[^a-zA-Z0-9]', '', query)}"
                            rating_opts_sr = [None, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
                            fmt_fn_sr = lambda x: "Rate" if x is None else f"{x} ★"
                            selected_rating_sr = st.selectbox("Your rating:", options=rating_opts_sr, index=0, format_func=fmt_fn_sr, key=rating_key_sr, label_visibility="collapsed")
                            user_id_for_rating_sr = None
                            current_session_user_id_sr = st.session_state.get('current_user_id')
                            is_user_identified = current_session_user_id_sr and current_session_user_id_sr != "new_user_temp" and \
                                                 (st.session_state.active_page == "user_specific" or st.session_state.get('last_processed_radio_selection') == "Log In (Existing Profile)" or \
                                                  (st.session_state.get('new_user_id_generated') is not None and st.session_state.new_user_id_generated == current_session_user_id_sr))
                            if is_user_identified:
                                st.caption(f"As User ID: {current_session_user_id_sr}")
                                user_id_for_rating_sr = current_session_user_id_sr
                            else:
                                st.caption("Log in or create a profile to save ratings easily, or enter your User ID manually.")
                                user_id_for_rating_input_val_sr = st.text_input("Your User ID (if known):", key=userid_key_sr, help="Enter your User ID (numeric) to save the rating.")
                                if user_id_for_rating_input_val_sr: user_id_for_rating_sr = user_id_for_rating_input_val_sr.strip()
                            if st.button("Save", key=button_key_sr, use_container_width=True):
                                if selected_rating_sr is None: st.warning("Please select a rating.", icon="⚠️")
                                elif not user_id_for_rating_sr: st.warning("Please provide a User ID or log in.", icon="⚠️")
                                else:
                                    try: # ... (saving code unchanged) ...
                                        user_id_to_save_sr = int(user_id_for_rating_sr)
                                        rating_to_save_single_sr = { C.USER_ID_COL: user_id_to_save_sr, C.ITEM_ID_COL: movie_id, C.RATING_COL: selected_rating_sr, C.TIMESTAMP_COL: int(time.time()) }
                                        df_single_rating_sr = pd.DataFrame([rating_to_save_single_sr])
                                        pending_ratings_filepath_sr = C.EVIDENCE_PATH / getattr(C, 'NEW_RATINGS_PENDING_FILENAME', 'new_ratings_pending.csv')
                                        file_exists_sr = os.path.exists(pending_ratings_filepath_sr)
                                        df_single_rating_sr.to_csv(pending_ratings_filepath_sr, mode='a', header=not file_exists_sr, index=False)
                                        st.success(f"Rating ({selected_rating_sr}★) for '{title}' (User ID: {user_id_to_save_sr}) saved!")
                                    except ValueError: st.error("User ID must be an integer.", icon="❌")
                                    except Exception as e_save_sr: st.error(f"Saving error: {e_save_sr}", icon="❌")
                st.markdown("---")
else:
    search_placeholder.empty()
    valid_pages = ["general", "user_specific", "new_user_profiling", "new_user_instant_recs", "search_results"]
    if st.session_state.active_page not in valid_pages:
        st.session_state.active_page = "general"
        st.session_state.search_input_value = ""
        st.session_state.active_search_query = ""
        st.rerun()
# Add project footer both in the sidebar and at the bottom of the main page

# Sidebar footer
with st.sidebar:
    st.markdown(
        """
        <div style='margin-top: 40px; text-align: center; font-size: 0.85em; color: #888;'>
            Recommendation Systems Project MLSMM2156
        </div>
        """,
        unsafe_allow_html=True
    )
