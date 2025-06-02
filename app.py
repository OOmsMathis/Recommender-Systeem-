import streamlit as st
import pandas as pd
import os
import re
import random
import time
import math

import constants as C_module
C = C_module.Constant()
import content
import recommender
import explanations
from loaders import load_items, load_ratings
from models import ContentBased, df_items_global as models_df_items_global
from surprise import Dataset, Reader

# --- Constantes ---
N_RECOS_PERSONNALISEES_TOTAL_FETCH = 50
N_INSTANT_RECOS_NEW_USER = 10
CARDS_PER_ROW = 5

# --- Chargement des données ---
try:
    df_items_global_app = load_items()
    df_ratings_global_app = load_ratings()
    if df_items_global_app.empty or df_ratings_global_app.empty:
        st.error("Critical error: Movie or rating data could not be loaded.")
        st.stop()
except Exception as e_load:
    st.error(f"Fatal error during initial data loading: {e_load}")
    st.stop()

st.set_page_config(page_title="Movie Recommendation", layout="wide")


# MODIFIED: Custom styled main title (unchanged from previous version as it suits light/dark themes)
st.markdown(
    f"""
    <div style="background-color: rgb(247, 75, 75); padding: 15px 10px 12px 10px; border-radius: 5px; margin-bottom: 30px; margin-top: -50px;">
        <h1 style="color: white; text-align: center; margin: 0; font-size: 2.5em; font-family: 'Roboto', sans-serif; font-weight: 700;">🎬 Movie Recommendation System</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Fonctions de récupération de données --- (Identique à la version précédente)
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

# --- Fonctions d'affichage des bandeaux ---
def display_movie_carousel(carousel_id, carousel_title, movies_df,
                           enable_rating_for_user_id=None,
                           num_cards_to_show_at_once=CARDS_PER_ROW,
                           is_personalized=False):
    if movies_df.empty:
        return

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

    if total_pages > 1:
        col_title, col_prev, col_next = st.columns([title_col_ratio, button_col_ratio, button_col_ratio])
    else:
        col_title = st
        col_prev, col_next = None, None

    with col_title:
        st.markdown(f"<h3 style='color: #1E1E1E; margin-bottom: 10px;'>{carousel_title}</h3>", unsafe_allow_html=True)


    if total_pages > 1 and col_prev is not None and col_next is not None:
        with col_prev:
            if st.button("⬅️", key=f"prev_{carousel_id}", use_container_width=True, disabled=(current_page == 0)):
                st.session_state[f'{carousel_id}_page'] -= 1
                st.rerun()
        with col_next:
            if st.button("➡️", key=f"next_{carousel_id}", use_container_width=True, disabled=(current_page >= total_pages - 1)):
                st.session_state[f'{carousel_id}_page'] += 1
                st.rerun()
    elif col_prev is not None and col_next is not None:
         with col_prev: st.empty()
         with col_next: st.empty()

    if not movies_to_display_on_page.empty:
        num_actual_cards_on_page = len(movies_to_display_on_page)
        cols_cards = st.columns(num_actual_cards_on_page)

        for idx, (_, movie_data) in enumerate(movies_to_display_on_page.iterrows()):
            with cols_cards[idx]:
                genres_display_text = str(movie_data.get(C.GENRES_COL, "N/A")).replace('|', ', ')

                title_text_plain = str(movie_data.get(C.LABEL_COL, "Unknown Title"))
                tmdb_id_val = movie_data.get(C.TMDB_ID_COL)
                core_title_html = f"<h6 style='font-weight: bold; color: white; margin: 0; padding: 0; line-height: 1.3; word-wrap: break-word; font-family: \"Roboto\", sans-serif;'>{title_text_plain}</h6>"
                if pd.notna(tmdb_id_val):
                    try:
                        title_link_url = f"https://www.themoviedb.org/movie/{int(tmdb_id_val)}"
                        core_title_html = f"<h6 style='font-weight: bold; margin: 0; padding: 0; line-height: 1.3; word-wrap: break-word; font-family: \"Roboto\", sans-serif;'><a href='{title_link_url}' target='_blank' style='color: white; text-decoration: none;'>{title_text_plain}</a></h6>"
                    except ValueError:
                        pass

                title_slot_html = (
                    f"<div style='height: 48px; overflow: hidden; display: flex; align-items: flex-start; margin-bottom: 3px;'>"
                    f"{core_title_html}"
                    f"</div>"
                )

                year_val = movie_data.get(C.RELEASE_YEAR_COL)
                year_display = int(year_val) if pd.notna(year_val) and year_val != 0 else "N/A"
                genre_year_html = f"<p style='font-size: 0.8em; color: #f0f0f0; margin: 0 0 5px 0; line-height: 1.3;'>{genres_display_text} | {year_display}</p>"

                top_content_html = title_slot_html + genre_year_html

                bottom_html_parts = []
                if is_personalized and 'estimated_score' in movie_data and pd.notna(movie_data['estimated_score']):
                    display_score = movie_data['estimated_score']
                    if hasattr(C, 'RATINGS_SCALE') and (C.RATINGS_SCALE == (1,5) or C.RATINGS_SCALE == (0.5, 5.0)) :
                        display_score *=2
                    bottom_html_parts.append(f"<small style='color: white; line-height: 1.3;'>For you: <strong>{display_score:.1f}/10</strong></small>")

                tmdb_avg_val = movie_data.get('tmdb_vote_average')
                if pd.isna(tmdb_avg_val) and hasattr(C, 'VOTE_AVERAGE_COL'):
                    tmdb_avg_val = movie_data.get(C.VOTE_AVERAGE_COL)

                if pd.notna(tmdb_avg_val):
                    try:
                        bottom_html_parts.append(f"<small style='color: #f0f0f0; line-height: 1.3;'>TMDB Rating: {pd.to_numeric(tmdb_avg_val, errors='coerce'):.1f}/10</small>")
                    except: pass

                bottom_content_html = "<br>".join(bottom_html_parts)
                if not bottom_content_html:
                    bottom_content_html = "<small style='line-height: 1.3;'>&nbsp;</small>"
                
                st.markdown(
                    f"<div style='background-color: #CD5C5C; color: white; border-radius: 8px; "
                    f"padding: 15px; height: 300px; box-shadow: 3px 3px 8px rgba(0,0,0,0.2); " # Shadow lightened for light bg
                    f"margin-bottom: 8px; display: flex; flex-direction: column;'>"
                    f"<div>{top_content_html}</div>"
                    f"<div style='margin-top: auto;'>{bottom_content_html}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

                movie_id_current = movie_data.get(C.ITEM_ID_COL)
                if enable_rating_for_user_id is not None and movie_id_current is not None:
                    rating_opts = [None, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
                    fmt_fn = lambda x: "Rate" if x is None else f"{x} ★"
                    clean_carousel_id_for_key = re.sub(r'\W+', '', carousel_id.lower())[:15]
                    rating_key = f"rating_{clean_carousel_id_for_key}_{str(movie_id_current)}_{current_page}_{str(enable_rating_for_user_id)}"

                    current_buffered_rating = st.session_state.logged_in_user_ratings_buffer.get(movie_id_current)
                    idx_rating = rating_opts.index(current_buffered_rating) if current_buffered_rating in rating_opts else 0

                    previous_rating_in_buffer = st.session_state.logged_in_user_ratings_buffer.get(movie_id_current)

                    user_rating_input = st.selectbox(
                        label="Your rating:", options=rating_opts, index=idx_rating,
                        format_func=fmt_fn, key=rating_key, label_visibility="collapsed"
                    )

                    if user_rating_input != previous_rating_in_buffer:
                        if user_rating_input is not None:
                            st.session_state.logged_in_user_ratings_buffer[movie_id_current] = user_rating_input
                        elif movie_id_current in st.session_state.logged_in_user_ratings_buffer and user_rating_input is None:
                            del st.session_state.logged_in_user_ratings_buffer[movie_id_current]
    st.markdown("<hr style='margin-top: 0.5rem; margin-bottom: 1rem;'>", unsafe_allow_html=True)


# --- Session State & Sidebar ---
if 'last_processed_radio_selection' not in st.session_state:st.session_state.last_processed_radio_selection = None
if 'active_page' not in st.session_state: st.session_state.active_page = "general"
if 'current_user_id' not in st.session_state: st.session_state.current_user_id = None
if 'new_user_ratings' not in st.session_state: st.session_state.new_user_ratings = {}
if 'new_user_name_input' not in st.session_state: st.session_state.new_user_name_input = ''
if 'last_selected_user_id' not in st.session_state: st.session_state.last_selected_user_id = None
if 'instant_reco_model_new_user' not in st.session_state: st.session_state.instant_reco_model_new_user = None
if 'new_user_id_generated' not in st.session_state: st.session_state.new_user_id_generated = None
if 'logged_in_user_ratings_buffer' not in st.session_state:st.session_state.logged_in_user_ratings_buffer = {}

if 'search_input_value' not in st.session_state: st.session_state.search_input_value = ""
if 'active_search_query' not in st.session_state: st.session_state.active_search_query = ""

# --- Sidebar setup ---
st.sidebar.header("Filters and Options")
all_genres_list_sidebar = ["All Genres"]
if not df_items_global_app.empty and hasattr(C, 'GENRES_COL') and C.GENRES_COL in df_items_global_app.columns:
    try:
        genres_series = df_items_global_app[C.GENRES_COL].fillna('').astype(str)
        s_genres = genres_series.str.split('|').explode()
        unique_sidebar_genres = sorted([ g.strip() for g in s_genres.unique() if g.strip() and g.strip().lower() != '(no genres listed)' ])
        if unique_sidebar_genres: all_genres_list_sidebar.extend(unique_sidebar_genres)
    except Exception as e_g_sb: print(f"Sidebar error (genre list): {e_g_sb}"); st.sidebar.error("Error loading genres.")
selected_genre_sidebar = st.sidebar.selectbox("Filter by genre:", all_genres_list_sidebar, key="genre_filter_sb")

slider_min, slider_max, current_slider_val = 1900, pd.Timestamp.now().year, (1900, pd.Timestamp.now().year)
if not df_items_global_app.empty and hasattr(C, 'RELEASE_YEAR_COL') and C.RELEASE_YEAR_COL in df_items_global_app.columns:
    valid_years = pd.to_numeric(df_items_global_app[C.RELEASE_YEAR_COL], errors='coerce').dropna()
    if not valid_years.empty:
        calc_min, calc_max = int(valid_years.min()), int(valid_years.max())
        if calc_min <= calc_max and calc_min > 1800: slider_min = calc_min
        if calc_max <= pd.Timestamp.now().year + 5: slider_max = calc_max
        current_slider_val = (slider_min, slider_max)
if slider_max < slider_min: slider_max = slider_min
selected_year_range_sidebar = st.sidebar.slider("Filter by year:", min_value=slider_min, max_value=slider_max, value=current_slider_val, key="year_filter_sb")

st.sidebar.markdown("---")
# Search bar removed from sidebar here

st.sidebar.header("👤 User Space")
user_opts = ["Explore Movies"]
can_sel_existing = not df_ratings_global_app.empty and hasattr(C, 'USER_ID_COL') and \
                   C.USER_ID_COL in df_ratings_global_app.columns and \
                   not df_ratings_global_app[C.USER_ID_COL].empty
if can_sel_existing: user_opts.append("Log In (Existing Profile)")
user_opts.append("Create New Profile")

idx_radio = 0
page_for_radio_display = st.session_state.active_page
current_radio_option_to_reflect = st.session_state.last_processed_radio_selection

if st.session_state.active_page == "search_results":
    if current_radio_option_to_reflect == "Explore Movies": page_for_radio_display = "general"
    elif current_radio_option_to_reflect == "Create New Profile": page_for_radio_display = "new_user_profiling"
    elif current_radio_option_to_reflect == "Log In (Existing Profile)": page_for_radio_display = "user_specific"
    else: page_for_radio_display = "general" # Default if radio state is somehow lost

if page_for_radio_display == "general":
    idx_radio = user_opts.index("Explore Movies") if "Explore Movies" in user_opts else 0
elif page_for_radio_display in ["new_user_profiling", "new_user_instant_recs"]:
    idx_radio = user_opts.index("Create New Profile") if "Create New Profile" in user_opts else 0
elif page_for_radio_display == "user_specific" and "Log In (Existing Profile)" in user_opts:
    idx_radio = user_opts.index("Log In (Existing Profile)")

user_sel_opt = st.sidebar.radio("Choose an option:", user_opts, key="user_sel_main_radio", index=idx_radio)

user_has_made_new_radio_choice = False
if st.session_state.last_processed_radio_selection != user_sel_opt:
    user_has_made_new_radio_choice = True
    st.session_state.last_processed_radio_selection = user_sel_opt

if user_has_made_new_radio_choice:
    intended_page_from_radio = st.session_state.active_page
    intended_uid_from_radio = st.session_state.current_user_id

    if user_sel_opt == "Explore Movies":
        intended_page_from_radio, intended_uid_from_radio = "general", None
    elif user_sel_opt == "Create New Profile":
        intended_page_from_radio = "new_user_profiling"
        intended_uid_from_radio = "new_user_temp" # Indicates a new user flow
        if st.session_state.active_page not in ["new_user_profiling", "new_user_instant_recs"]:
            st.session_state.new_user_ratings, st.session_state.new_user_name_input = {}, ''
            st.session_state.instant_reco_model_new_user, st.session_state.new_user_id_generated = None, None
    elif user_sel_opt == "Log In (Existing Profile)" and can_sel_existing:
        intended_page_from_radio = "user_specific"
        if st.session_state.current_user_id is None or st.session_state.current_user_id == "new_user_temp":
            uids_list_for_default = sorted(df_ratings_global_app[C.USER_ID_COL].unique()) if not df_ratings_global_app.empty else []
            last_id_sel = st.session_state.last_selected_user_id
            intended_uid_from_radio = last_id_sel if last_id_sel in uids_list_for_default else (uids_list_for_default[0] if uids_list_for_default else None)

    if st.session_state.active_page != intended_page_from_radio or \
       st.session_state.current_user_id != intended_uid_from_radio:
        st.session_state.active_page = intended_page_from_radio
        st.session_state.current_user_id = intended_uid_from_radio

        # If navigating away from search results via radio, clear search
        if intended_page_from_radio != "search_results":
            st.session_state.search_input_value = ""
            st.session_state.active_search_query = ""
        st.rerun()


# --- MODIFIED: Search bar in the main page ---
search_placeholder = st.empty() # Placeholder for the search bar container

with search_placeholder.container(): # Use a container to manage search bar visibility if needed later
    # This search bar will always be visible at the top of the main content area
    # Adjust its placement or add conditional visibility if needed for specific pages like new_user_profiling
    
    # Only show search bar if not in new user profiling or instant recs page
    if st.session_state.active_page not in ["new_user_profiling", "new_user_instant_recs"]:
        st.markdown("---") # Visual separator
        # Use columns to constrain width of search bar
        search_col_1, search_col_2, search_col_3 = st.columns([0.2, 0.6, 0.2])
        with search_col_2:
            search_text_input_main = st.text_input(
                "🔍 Search for a movie by title:",
                value=st.session_state.search_input_value, # Use session state to preserve value across reruns
                key="movie_search_main_key",
                help="Enter part of the title and press Enter."
            )

        if search_text_input_main != st.session_state.search_input_value: # If text_input changed
            st.session_state.search_input_value = search_text_input_main # Update session state value

            if st.session_state.search_input_value: # If there's a new search query
                st.session_state.active_search_query = st.session_state.search_input_value
                if st.session_state.active_page != "search_results":
                    st.session_state.active_page = "search_results"
                st.rerun() # Rerun to either switch to search page or update results on search page
            
            # If search input is cleared *while on the search results page*
            elif not st.session_state.search_input_value and st.session_state.active_page == "search_results":
                st.session_state.active_search_query = ""
                # Determine where to go back: to 'general' or user's previous context based on radio
                # This logic is similar to how radio button changes page
                current_radio_selection = st.session_state.get('last_processed_radio_selection', "Explore Movies")
                if current_radio_selection == "Explore Movies":
                    st.session_state.active_page = "general"
                elif current_radio_selection == "Log In (Existing Profile)":
                    st.session_state.active_page = "user_specific" # uid should persist or be re-evaluated
                elif current_radio_selection == "Create New Profile": # Unlikely from search results, but handle
                    st.session_state.active_page = "new_user_profiling"
                else:
                    st.session_state.active_page = "general" # Fallback
                st.rerun()
        st.markdown("---") # Visual separator

uid_for_reco = None
user_profiles_map = {}

if st.session_state.active_page == "user_specific":
    if user_sel_opt == "Log In (Existing Profile)" and can_sel_existing:
        user_profiles_path = C.DATA_PATH / getattr(C, 'USER_PROFILES_FILENAME', 'user_profiles.csv')
        if os.path.exists(user_profiles_path):
            try:
                df_profiles = pd.read_csv(user_profiles_path)
                if 'userId' in df_profiles.columns and 'userName' in df_profiles.columns:
                     user_profiles_map = pd.Series(df_profiles.userName.values, index=df_profiles.userId).to_dict()
            except Exception as e_pf: print(f"Error loading user_profiles.csv for sorting: {e_pf}")

        disp_opts_func = lambda uid_val: f"{user_profiles_map.get(uid_val, 'User')} (ID: {uid_val})"
        uids_from_ratings = df_ratings_global_app[C.USER_ID_COL].unique()
        user_sort_list = []
        for uid_val_loop in uids_from_ratings:
            actual_name = user_profiles_map.get(uid_val_loop)
            has_profile_name_sort_key = 0 if actual_name else 1
            display_name_for_sort = f"{actual_name if actual_name else 'User'} (ID: {uid_val_loop})"
            user_sort_list.append({'uid': uid_val_loop, 'sort_key': has_profile_name_sort_key, 'display_text': display_name_for_sort})

        user_sort_list.sort(key=lambda x: (x['sort_key'], x['display_text']))
        uids_avail = [user['uid'] for user in user_sort_list]

        if uids_avail:
            current_selection_uid = st.session_state.current_user_id
            if current_selection_uid not in uids_avail:
                current_selection_uid = uids_avail[0]
                st.session_state.current_user_id = current_selection_uid
            idx_sel_box = uids_avail.index(current_selection_uid)

            uid_sel_box_val = st.sidebar.selectbox(
                f"Profile ID:",
                options=uids_avail,
                format_func=disp_opts_func,
                index=idx_sel_box,
                key="uid_sel_box"
            )
            if st.session_state.current_user_id != uid_sel_box_val:
                st.session_state.current_user_id = uid_sel_box_val
                st.session_state.last_selected_user_id = uid_sel_box_val
                st.rerun()
            uid_for_reco = st.session_state.current_user_id
        else:
            st.sidebar.warning("No user ratings available to select a profile.")
            uid_for_reco = None
    elif st.session_state.current_user_id not in [None, "new_user_temp"]:
        uid_for_reco = st.session_state.current_user_id


# --- Main Display Logic ---
if st.session_state.active_page == "general":
    st.header("General Recommendations and Discoveries")
    yr_min, yr_max = selected_year_range_sidebar[0], selected_year_range_sidebar[1]
    genre_f_general = selected_genre_sidebar if selected_genre_sidebar != "All Genres" else None
    genre_suffix = f" : {genre_f_general}" if genre_f_general else ""
    top_tmdb_movies = get_top_overall_movies_tmdb(genre_filter=genre_f_general, year_min_filter=yr_min, year_max_filter=yr_max)
    display_movie_carousel("top_tmdb", f"🏆 Top Overall Ratings{genre_suffix}", top_tmdb_movies)
    top_documentaries = get_top_genre_movies_tmdb(genre="Documentary", year_min_filter=yr_min, year_max_filter=yr_max)
    display_movie_carousel("top_documentaries", "📹 Must-Watch Documentaries", top_documentaries)
    hidden_gems = get_hidden_gems_movies(genre_filter=genre_f_general, year_min_filter=yr_min, year_max_filter=yr_max)
    display_movie_carousel("hidden_gems", f"💎 Hidden Gems{genre_suffix}", hidden_gems)

elif st.session_state.active_page == "user_specific" and uid_for_reco is not None:
    user_display_name_map_val = user_profiles_map.get(uid_for_reco, f"User {uid_for_reco}")
    st.header(f"Recommendations For You, {user_display_name_map_val}")

    yr_min_p, yr_max_p = selected_year_range_sidebar[0], selected_year_range_sidebar[1]
    genre_f_perso = selected_genre_sidebar if selected_genre_sidebar != "All Genres" else None

    models_p_dir = str(C.DATA_PATH / 'recs')
    avail_model_files = [f for f in os.listdir(models_p_dir) if f.endswith('.p') and not 'personalized' in f.lower()] if os.path.exists(models_p_dir) and os.path.isdir(models_p_dir) else []

    if not avail_model_files: st.error(f"No general pre-trained models found in {models_p_dir}.")
    else:
        user_profile_for_titles = explanations.get_user_profile_for_explanation(uid_for_reco, top_n_movies=2, min_rating=3.5)
        model_types_config = [
            ("content_based", "content_based", "Content-Based Suggestions"),
            ("user_based", "user_based", "Liked by Similar Profiles"),
            ("svd", "svd", "Algorithmic Discoveries For You")
        ]
        for model_key, file_keyword, fallback_carousel_title in model_types_config:
            m_file = next((mfile for mfile in avail_model_files if file_keyword in mfile.lower() and 'final' in mfile.lower()), None)
            if not m_file: m_file = next((mfile for mfile in avail_model_files if file_keyword in mfile.lower()), None)
            if m_file:
                recs_data = recommender.get_top_n_recommendations(
                    uid_for_reco, m_file, n=N_RECOS_PERSONNALISEES_TOTAL_FETCH,
                    filter_genre=genre_f_perso, filter_year_range=(yr_min_p, yr_max_p)
                )
                carousel_title_final = fallback_carousel_title
                if model_key == "content_based":
                    if user_profile_for_titles:
                        anchor_movie = user_profile_for_titles[0]
                        carousel_title_final = f"Because you liked {anchor_movie['title']} ({anchor_movie['rating']:.1f}/5):"
                elif model_key == "user_based":
                    if len(user_profile_for_titles) >= 2:
                        movie1, movie2 = user_profile_for_titles[0], user_profile_for_titles[1]
                        carousel_title_final = f"Fans of {movie1['title']} and {movie2['title']} also enjoy:"
                    elif len(user_profile_for_titles) == 1:
                        movie1 = user_profile_for_titles[0]
                        carousel_title_final = f"Fans of {movie1['title']} ({movie1['rating']:.1f}/5) also enjoy:"
                elif model_key == "svd":
                    # MODIFIED: SVD explanation changed
                    carousel_title_final = "Based on your personal behaviour..."

                carousel_id_perso = f"{model_key}_{str(uid_for_reco).replace('.', '_')}"
                display_movie_carousel(
                    carousel_id_perso, carousel_title_final, recs_data,
                    enable_rating_for_user_id=uid_for_reco, is_personalized=True
                )
            else: st.warning(f"No model of type '{file_keyword}' found.")

        if st.session_state.logged_in_user_ratings_buffer:
            st.markdown("---")
            num_buffered_ratings = len(st.session_state.logged_in_user_ratings_buffer)
            cols_save_button = st.columns([0.3, 0.4, 0.3])
            with cols_save_button[1]:
                 if st.button(f"✔️ Save my {num_buffered_ratings} new rating(s)", key="save_logged_in_ratings_final", use_container_width=True):
                    ratings_to_save_list = []
                    current_ts = int(time.time())
                    user_id_to_save = uid_for_reco
                    for movie_id_key, rating_val_key in st.session_state.logged_in_user_ratings_buffer.items():
                        ratings_to_save_list.append({
                            C.USER_ID_COL: user_id_to_save, C.ITEM_ID_COL: movie_id_key,
                            C.RATING_COL: rating_val_key, C.TIMESTAMP_COL: current_ts
                        })
                    if ratings_to_save_list:
                        df_new_ratings_to_save_out = pd.DataFrame(ratings_to_save_list)
                        pending_ratings_filepath = C.EVIDENCE_PATH / getattr(C, 'NEW_RATINGS_PENDING_FILENAME', 'new_ratings_pending.csv')
                        file_exists_pending = os.path.exists(pending_ratings_filepath)
                        try:
                            df_new_ratings_to_save_out.to_csv(pending_ratings_filepath, mode='a', header=not file_exists_pending, index=False)
                            st.success(f"{len(ratings_to_save_list)} rating(s) saved!")
                            st.session_state.logged_in_user_ratings_buffer = {}
                            st.rerun()
                        except Exception as e_save_rating:
                            st.error(f"Error saving ratings: {e_save_rating}")

elif st.session_state.active_page == "new_user_profiling":
    search_placeholder.empty() # Hide search bar on this page
    st.header("👤 Create Your Taste Profile")
    st.write("To help us understand your preferences, please enter your name and rate some movies.")
    new_user_name = st.text_input("What is your name?", st.session_state.get('new_user_name_input', ''))
    st.session_state.new_user_name_input = new_user_name

    movies_for_profiling_pool_initial = df_items_global_app.copy() if not df_items_global_app.empty else pd.DataFrame()
    sample_size = 20
    min_prefs_needed = 5
    movies_to_display_df = pd.DataFrame()

    if not movies_for_profiling_pool_initial.empty:
        num_candidate_popular_movies = 500
        movies_per_genre_target = 1
        candidate_pool_df = pd.DataFrame()

        if C.VOTE_COUNT_COL in movies_for_profiling_pool_initial.columns:
            movies_for_profiling_pool_initial[C.VOTE_COUNT_COL] = pd.to_numeric(movies_for_profiling_pool_initial[C.VOTE_COUNT_COL], errors='coerce').fillna(0)
            candidate_pool_df = movies_for_profiling_pool_initial.sort_values(by=C.VOTE_COUNT_COL, ascending=False).head(num_candidate_popular_movies)

        if candidate_pool_df.empty:
            if C.POPULARITY_COL in movies_for_profiling_pool_initial.columns and not movies_for_profiling_pool_initial[C.POPULARITY_COL].isnull().all():
                movies_for_profiling_pool_initial[C.POPULARITY_COL] = pd.to_numeric(movies_for_profiling_pool_initial[C.POPULARITY_COL], errors='coerce').fillna(0)
                candidate_pool_df = movies_for_profiling_pool_initial.sort_values(by=C.POPULARITY_COL, ascending=False).head(num_candidate_popular_movies)
            elif len(movies_for_profiling_pool_initial) > 0:
                candidate_pool_df = movies_for_profiling_pool_initial.sample(n=min(num_candidate_popular_movies, len(movies_for_profiling_pool_initial)), random_state=42)

        selected_movies_for_profiling_list = []
        selected_movie_ids = set()

        if not candidate_pool_df.empty and C.GENRES_COL in candidate_pool_df.columns:
            genres_df_for_selection = candidate_pool_df.copy()
            genres_df_for_selection['genre_list'] = genres_df_for_selection[C.GENRES_COL].astype(str).str.split('|')
            genres_exploded_df = genres_df_for_selection.explode('genre_list')
            genres_exploded_df['genre_list'] = genres_exploded_df['genre_list'].str.strip()
            genres_exploded_df = genres_exploded_df[genres_exploded_df['genre_list'].notna() & (genres_exploded_df['genre_list'] != '') & (genres_exploded_df['genre_list'].str.lower() != '(no genres listed)')]
            unique_genres_in_pool = genres_exploded_df['genre_list'].unique()
            random.shuffle(unique_genres_in_pool)

            for genre in unique_genres_in_pool:
                if len(selected_movies_for_profiling_list) >= sample_size: break
                movies_in_genre_from_exploded = genres_exploded_df[genres_exploded_df['genre_list'] == genre]
                sort_col_for_genre_picking = C.VOTE_COUNT_COL if C.VOTE_COUNT_COL in movies_in_genre_from_exploded else C.POPULARITY_COL
                if sort_col_for_genre_picking in movies_in_genre_from_exploded:
                    movies_in_genre_sorted = movies_in_genre_from_exploded.sort_values(by=sort_col_for_genre_picking, ascending=False)
                else: movies_in_genre_sorted = movies_in_genre_from_exploded

                added_for_this_genre = 0
                for _, movie_data_from_exploded in movies_in_genre_sorted.iterrows():
                    movie_id = movie_data_from_exploded[C.ITEM_ID_COL]
                    if movie_id not in selected_movie_ids:
                        full_movie_row_df = candidate_pool_df[candidate_pool_df[C.ITEM_ID_COL] == movie_id]
                        if not full_movie_row_df.empty:
                            selected_movies_for_profiling_list.append(full_movie_row_df.iloc[0].to_dict())
                            selected_movie_ids.add(movie_id)
                            added_for_this_genre += 1
                            if added_for_this_genre >= movies_per_genre_target or len(selected_movies_for_profiling_list) >= sample_size: break

        if len(selected_movies_for_profiling_list) < sample_size and not candidate_pool_df.empty:
            remaining_candidates = candidate_pool_df[~candidate_pool_df[C.ITEM_ID_COL].isin(selected_movie_ids)]
            needed = sample_size - len(selected_movies_for_profiling_list)
            for _, movie_row_to_add in remaining_candidates.head(needed).iterrows():
                if movie_row_to_add[C.ITEM_ID_COL] not in selected_movie_ids:
                    selected_movies_for_profiling_list.append(movie_row_to_add.to_dict())
                    selected_movie_ids.add(movie_row_to_add[C.ITEM_ID_COL])

        if selected_movies_for_profiling_list:
            movies_to_display_df = pd.DataFrame(selected_movies_for_profiling_list)
            if len(movies_to_display_df) > sample_size:
                movies_to_display_df = movies_to_display_df.sample(n=sample_size, random_state=43)
        else:
            st.warning("Diversified movie selection could not be performed. Using global popularity selection mode.")
            if hasattr(C, 'POPULARITY_COL') and C.POPULARITY_COL in movies_for_profiling_pool_initial.columns and not movies_for_profiling_pool_initial[C.POPULARITY_COL].isnull().all():
                movies_for_profiling_pool_initial[C.POPULARITY_COL] = pd.to_numeric(movies_for_profiling_pool_initial[C.POPULARITY_COL], errors='coerce').fillna(0)
                movies_to_display_df_temp = movies_for_profiling_pool_initial.sort_values(by=C.POPULARITY_COL, ascending=False).head(150)
                if len(movies_to_display_df_temp) >= sample_size: movies_to_display_df = movies_to_display_df_temp.sample(n=sample_size, random_state=42)
            if movies_to_display_df.empty:
                if len(movies_for_profiling_pool_initial) >= sample_size: movies_to_display_df = movies_for_profiling_pool_initial.sample(n=sample_size, random_state=42)
                elif not movies_for_profiling_pool_initial.empty : movies_to_display_df = movies_for_profiling_pool_initial.copy()

    if movies_to_display_df.empty: st.error("Could not load movies for profiling.")
    else:
        movies_to_display_final_unique = movies_to_display_df.drop_duplicates(subset=[C.ITEM_ID_COL])
        if len(movies_to_display_final_unique) > sample_size:
             movies_to_display_final_unique = movies_to_display_final_unique.sample(n=sample_size, random_state=44)

        with st.form(key="new_user_profiling_form"):
            st.subheader("Rate the following movies")
            st.caption("Leave as 'No rating' if you don't know the movie or have no opinion.")
            for _, row in movies_to_display_final_unique.iterrows():
                movie_id, title = row[C.ITEM_ID_COL], row[C.LABEL_COL]
                rating_key = f"rating_select_{movie_id}"
                current_rating = st.session_state.new_user_ratings.get(movie_id)
                col1, col2 = st.columns([3,2])
                with col1:
                    st.markdown(f"**{title}**")
                    if hasattr(C, 'GENRES_COL') and C.GENRES_COL in row and pd.notna(row[C.GENRES_COL]):
                        genres_display_text_profiling = str(row[C.GENRES_COL]).replace('|', ', ')
                        st.caption(f"Genres: {genres_display_text_profiling}")
                with col2:
                    opts = [None, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
                    fmt_fn = lambda x: "No rating" if x is None else f"{x} ★"
                    idx = opts.index(current_rating) if current_rating in opts else 0
                    new_r = st.selectbox(f"Rating for {title}:", opts, index=idx, format_func=fmt_fn, key=rating_key, label_visibility="collapsed")
                    if new_r is not None: st.session_state.new_user_ratings[movie_id] = new_r
                    elif movie_id in st.session_state.new_user_ratings: del st.session_state.new_user_ratings[movie_id]
            form_submitted = st.form_submit_button("✔️ Save and See Initial Suggestions")

        final_prefs = st.session_state.new_user_ratings.copy()
        num_total_prefs = len(final_prefs)
        st.info(f"You have provided {num_total_prefs} rating(s). At least {min_prefs_needed} are required.")
        if form_submitted:
            if not new_user_name.strip(): st.warning("Please enter your name.")
            elif num_total_prefs < min_prefs_needed: st.warning(f"Please rate at least {min_prefs_needed} movies.")
            else:
                try:
                    current_ratings_df_for_id = df_ratings_global_app
                    new_user_id_val = (current_ratings_df_for_id[C.USER_ID_COL].max() + 1) if not current_ratings_df_for_id.empty and C.USER_ID_COL in current_ratings_df_for_id else 1
                    st.session_state.new_user_id_generated = new_user_id_val
                    ratings_to_save_list = []
                    current_ts = int(time.time())
                    for movie_id_key, rating_val_key in final_prefs.items():
                        ratings_to_save_list.append({C.USER_ID_COL: new_user_id_val, C.ITEM_ID_COL: movie_id_key, C.RATING_COL: rating_val_key, C.TIMESTAMP_COL: current_ts})
                    if ratings_to_save_list:
                        df_new_ratings_to_save_out = pd.DataFrame(ratings_to_save_list)
                        pending_ratings_filepath = C.EVIDENCE_PATH / getattr(C, 'NEW_RATINGS_PENDING_FILENAME', 'new_ratings_pending.csv')
                        file_exists_pending = os.path.exists(pending_ratings_filepath)
                        df_new_ratings_to_save_out.to_csv(pending_ratings_filepath, mode='a', header=not file_exists_pending, index=False)
                        user_profiles_filepath = C.DATA_PATH / getattr(C, 'USER_PROFILES_FILENAME', 'user_profiles.csv')
                        user_profile_data_out = pd.DataFrame([{'userId': new_user_id_val, 'userName': new_user_name.strip()}])
                        file_exists_profiles = os.path.exists(user_profiles_filepath)
                        user_profile_data_out.to_csv(user_profiles_filepath, mode='a', header=not file_exists_profiles, index=False)
                        st.success(f"Profile for {new_user_name.strip()} (ID: {new_user_id_val}) saved with {len(ratings_to_save_list)} ratings.")

                    st.info("Calculating your first suggestions...")
                    actual_cb_features_instant = []
                    if hasattr(C, 'GENRES_COL') and C.GENRES_COL in models_df_items_global.columns : actual_cb_features_instant.append("Genre_binary")
                    if hasattr(C, 'RELEASE_YEAR_COL') and C.RELEASE_YEAR_COL in models_df_items_global.columns: actual_cb_features_instant.append("Year_of_release")
                    if not actual_cb_features_instant:
                        st.warning("Not enough features available for Content-Based instant suggestions.")
                        st.session_state.active_page = "general"; st.rerun()
                    else:
                        cb_model_instant = ContentBased(features_methods=actual_cb_features_instant, regressor_method='linear')
                        reader = Reader(rating_scale=C.RATINGS_SCALE if hasattr(C, 'RATINGS_SCALE') else (0.5, 5.0))
                        instant_user_id_for_train = -1
                        ratings_for_instant_model_df = pd.DataFrame([{'userId': instant_user_id_for_train, 'movieId': mid, 'rating': rval} for mid, rval in final_prefs.items()])
                        data_instant = Dataset.load_from_df(ratings_for_instant_model_df, reader)
                        trainset_instant = data_instant.build_full_trainset()
                        cb_model_instant.fit(trainset_instant)
                        st.session_state.instant_reco_model_new_user = cb_model_instant
                        st.session_state.active_page = "new_user_instant_recs"
                        st.rerun()
                except Exception as e_profile_processing:
                    st.error(f"Error creating your profile: {e_profile_processing}")
                    st.exception(e_profile_processing)

elif st.session_state.active_page == "new_user_instant_recs":
    search_placeholder.empty() # Hide search bar on this page
    st.header("🎉 Your First Movie Suggestions!")
    st.caption("Based on the ratings you just provided.")
    model_instance = st.session_state.get('instant_reco_model_new_user')
    new_user_ratings_keys = st.session_state.get('new_user_ratings', {}).keys()
    generated_user_id_for_pred = -1
    if model_instance and models_df_items_global is not None and not models_df_items_global.empty:
        all_movie_ids_global = models_df_items_global[C.ITEM_ID_COL].unique()
        movies_to_predict_ids = [mid for mid in all_movie_ids_global if mid not in new_user_ratings_keys]
        if not movies_to_predict_ids: st.info("No other movies to suggest at the moment.")
        else:
            preds_list_instant = []
            sample_size_instant_pred = min(len(movies_to_predict_ids), 200)
            for item_id_to_predict in random.sample(movies_to_predict_ids, sample_size_instant_pred):
                try:
                    prediction = model_instance.predict(uid=generated_user_id_for_pred, iid=item_id_to_predict)
                    preds_list_instant.append({C.ITEM_ID_COL: prediction.iid, 'estimated_score': prediction.est})
                except: continue
            if preds_list_instant:
                recs_instant_df_raw = pd.DataFrame(preds_list_instant).sort_values(by='estimated_score', ascending=False).head(N_INSTANT_RECOS_NEW_USER)
                if not recs_instant_df_raw.empty and not df_items_global_app.empty:
                    final_recs_instant_df = pd.merge(
                        recs_instant_df_raw,
                        df_items_global_app[[C.ITEM_ID_COL, C.LABEL_COL, C.GENRES_COL, C.RELEASE_YEAR_COL, C.VOTE_AVERAGE_COL, C.TMDB_ID_COL]],
                        on=C.ITEM_ID_COL, how='left'
                    )
                    display_movie_carousel("instant_recs_new_user", "Quick suggestions for you:", final_recs_instant_df, is_personalized=True)
                elif not recs_instant_df_raw.empty: st.write(recs_instant_df_raw)
                else: st.info("Could not generate instant suggestions.")
            else: st.info("No instant suggestions could be generated.")
    else: st.warning("The instant suggestion model is not available or item data is missing.")
    if st.button("Explore other movies"):
        st.session_state.active_page = "general"
        st.session_state.new_user_ratings, st.session_state.new_user_name_input = {}, ''
        st.session_state.instant_reco_model_new_user, st.session_state.new_user_id_generated = None, None
        st.session_state.search_input_value = ""
        st.session_state.active_search_query = ""
        st.rerun()

elif st.session_state.active_page == "search_results":
    query = st.session_state.active_search_query # Query comes from main search bar state
    st.header(f"🎬 Search Results for: \"{query}\"")

    if not query: # Should not happen if search bar logic is correct, but as a fallback
        st.warning("Please enter a term in the search bar above.")
        if st.button("Back to Home"):
            st.session_state.active_page = "general"
            st.session_state.search_input_value = ""
            st.session_state.active_search_query = ""
            st.rerun()
    else:
        results_df = df_items_global_app[
            df_items_global_app[C.LABEL_COL].str.contains(re.escape(query), case=False, na=False)
        ]

        if results_df.empty:
            st.info(f"No movies found containing \"{query}\".")
        else:
            st.markdown(f"**{len(results_df)} movie(s) found:**")

            for _, movie_data in results_df.iterrows():
                movie_id = movie_data[C.ITEM_ID_COL]
                title = movie_data[C.LABEL_COL]
                genres_text_search = str(movie_data.get(C.GENRES_COL, "N/A")).replace('|', ', ')
                year_val = movie_data.get(C.RELEASE_YEAR_COL)
                year_display = int(year_val) if pd.notna(year_val) and year_val != 0 else "N/A"

                with st.container(border=True):
                    cols_info, cols_rating_form = st.columns([0.6, 0.4])

                    with cols_info:
                        st.subheader(f"{title}")
                        st.caption(f"Genres: {genres_text_search} | Year: {year_display}")
                        tmdb_id_val = movie_data.get(C.TMDB_ID_COL)
                        if pd.notna(tmdb_id_val):
                            try:
                                title_link = f"https://www.themoviedb.org/movie/{int(tmdb_id_val)}"
                                st.markdown(f"<a href='{title_link}' target='_blank' style='font-size:0.9em;'>View on TMDB 🔗</a>", unsafe_allow_html=True) # Link color styled by CSS
                            except ValueError: pass

                    with cols_rating_form:
                        st.markdown("##### Rate this movie:")

                        rating_key_sr = f"search_rating_{movie_id}_{re.sub(r'[^a-zA-Z0-9]', '', query)}"
                        userid_key_sr = f"search_userid_{movie_id}_{re.sub(r'[^a-zA-Z0-9]', '', query)}"
                        button_key_sr = f"search_save_{movie_id}_{re.sub(r'[^a-zA-Z0-9]', '', query)}"

                        rating_opts_sr = [None, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
                        fmt_fn_sr = lambda x: "Rate" if x is None else f"{x} ★"

                        selected_rating_sr = st.selectbox("Your rating:", options=rating_opts_sr, index=0,
                                                        format_func=fmt_fn_sr, key=rating_key_sr, label_visibility="collapsed")

                        user_id_for_rating_sr = None
                        current_session_user_id_sr = st.session_state.get('current_user_id')
                        is_user_identified = current_session_user_id_sr and \
                                             current_session_user_id_sr != "new_user_temp" and \
                                             (st.session_state.active_page == "user_specific" or \
                                              st.session_state.get('last_processed_radio_selection') == "Log In (Existing Profile)" or \
                                              (st.session_state.get('new_user_id_generated') is not None and \
                                               st.session_state.new_user_id_generated == current_session_user_id_sr))
                        
                        if is_user_identified:
                            st.caption(f"As User ID: {current_session_user_id_sr}")
                            user_id_for_rating_sr = current_session_user_id_sr
                        else:
                            # If user is not logged in or is a 'new_user_temp', prompt for ID.
                            st.caption("Log in or create a profile to save ratings easily, or enter your User ID manually.")
                            user_id_for_rating_input_val_sr = st.text_input("Your User ID (if known):", key=userid_key_sr,
                                                                            help="Enter your User ID (numeric) to save the rating.")
                            if user_id_for_rating_input_val_sr:
                                user_id_for_rating_sr = user_id_for_rating_input_val_sr.strip()

                        if st.button("💾 Save", key=button_key_sr, use_container_width=True):
                            if selected_rating_sr is None:
                                st.warning("Please select a rating.", icon="⚠️")
                            elif not user_id_for_rating_sr:
                                st.warning("Please provide a User ID or log in.", icon="⚠️")
                            else:
                                try:
                                    user_id_to_save_sr = int(user_id_for_rating_sr)
                                    rating_to_save_single_sr = {
                                        C.USER_ID_COL: user_id_to_save_sr, C.ITEM_ID_COL: movie_id,
                                        C.RATING_COL: selected_rating_sr, C.TIMESTAMP_COL: int(time.time())
                                    }
                                    df_single_rating_sr = pd.DataFrame([rating_to_save_single_sr])
                                    pending_ratings_filepath_sr = C.EVIDENCE_PATH / getattr(C, 'NEW_RATINGS_PENDING_FILENAME', 'new_ratings_pending.csv')
                                    file_exists_sr = os.path.exists(pending_ratings_filepath_sr)
                                    df_single_rating_sr.to_csv(pending_ratings_filepath_sr, mode='a', header=not file_exists_sr, index=False)
                                    st.success(f"Rating ({selected_rating_sr}★) for '{title}' (User ID: {user_id_to_save_sr}) saved!")
                                except ValueError:
                                    st.error("User ID must be an integer.", icon="❌")
                                except Exception as e_save_sr:
                                    st.error(f"Saving error: {e_save_sr}", icon="❌")
                st.markdown("---")
else: # Fallback for undefined active_page
    search_placeholder.empty() # Ensure search is not shown if page state is invalid
    valid_pages = ["general", "user_specific", "new_user_profiling", "new_user_instant_recs", "search_results"]
    if st.session_state.active_page not in valid_pages:
        st.session_state.active_page = "general"
        st.session_state.search_input_value = ""
        st.session_state.active_search_query = ""
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("Recommender Systems Project MLSMM2156")