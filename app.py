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
        st.error("Erreur critique : Les données des films ou des évaluations n'ont pas pu être chargées.")
        st.stop()
except Exception as e_load:
    st.error(f"Erreur fatale lors du chargement initial des données: {e_load}")
    st.stop()

st.set_page_config(page_title="Recommandation de Films", layout="wide")
st.title("🎬 Système de Recommandation de Films")

# --- Fonctions de récupération de données (inchangées par rapport à la dernière version fournie) ---
@st.cache_data
def get_top_overall_movies_tmdb(n=N_RECOS_PERSONNALISEES_TOTAL_FETCH, genre_filter=None, year_min_filter=None, year_max_filter=None):
    if df_items_global_app.empty or not hasattr(C, 'VOTE_AVERAGE_COL') or C.VOTE_AVERAGE_COL not in df_items_global_app.columns:
        return pd.DataFrame()
    items_to_consider = df_items_global_app.copy()
    if genre_filter and genre_filter != "Tous les genres" and hasattr(C, 'GENRES_COL') and C.GENRES_COL in items_to_consider.columns:
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
def get_hidden_gems_movies(n=N_RECOS_PERSONNALISEES_TOTAL_FETCH, genre_filter=None, year_min_filter=None, year_max_filter=None, min_vote_average=6.5, min_votes=20):
    if df_items_global_app.empty or not hasattr(C, 'VOTE_COUNT_COL') or C.VOTE_COUNT_COL not in df_items_global_app.columns \
       or not hasattr(C, 'VOTE_AVERAGE_COL') or C.VOTE_AVERAGE_COL not in df_items_global_app.columns:
        return pd.DataFrame()
    items_to_consider = df_items_global_app.copy()
    if genre_filter and genre_filter != "Tous les genres" and hasattr(C, 'GENRES_COL') and C.GENRES_COL in items_to_consider.columns:
        items_to_consider = items_to_consider[
            items_to_consider[C.GENRES_COL].astype(str).str.contains(re.escape(genre_filter), case=False, na=False, regex=True)
        ]
        if items_to_consider.empty: return pd.DataFrame()
    items_to_consider[C.VOTE_COUNT_COL] = pd.to_numeric(items_to_consider[C.VOTE_COUNT_COL], errors='coerce').fillna(0)
    items_to_consider[C.VOTE_AVERAGE_COL] = pd.to_numeric(items_to_consider[C.VOTE_AVERAGE_COL], errors='coerce').fillna(0)
    items_to_consider = items_to_consider[items_to_consider[C.VOTE_AVERAGE_COL] >= min_vote_average]
    items_to_consider = items_to_consider[items_to_consider[C.VOTE_COUNT_COL] >= min_votes]
    if year_min_filter is not None and year_max_filter is not None and hasattr(C, 'RELEASE_YEAR_COL') and C.RELEASE_YEAR_COL in items_to_consider.columns:
        items_to_consider[C.RELEASE_YEAR_COL] = pd.to_numeric(items_to_consider[C.RELEASE_YEAR_COL], errors='coerce').fillna(0)
        items_to_consider = items_to_consider[
            (items_to_consider[C.RELEASE_YEAR_COL] >= year_min_filter) &
            (items_to_consider[C.RELEASE_YEAR_COL] <= year_max_filter)
        ]
    if items_to_consider.empty: return pd.DataFrame()
    hidden_gems_df = items_to_consider.sort_values(by=[C.VOTE_COUNT_COL, C.VOTE_AVERAGE_COL], ascending=[True, False]).head(n)
    cols_out = [C.ITEM_ID_COL, C.LABEL_COL, C.GENRES_COL, C.RELEASE_YEAR_COL, C.VOTE_AVERAGE_COL, C.VOTE_COUNT_COL, C.TMDB_ID_COL]
    return hidden_gems_df[[col for col in cols_out if col in hidden_gems_df.columns]]


# --- Fonctions d'affichage des bandeaux (AVEC PAGINATION HORIZONTALE et couleur ROUGE VIF) ---
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
        st.subheader(carousel_title)

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
                # Construire le contenu HTML de la carte
                card_content_html = ""
                title_text_plain = str(movie_data.get(C.LABEL_COL, "Titre Inconnu"))
                tmdb_id_val = movie_data.get(C.TMDB_ID_COL)

                if pd.notna(tmdb_id_val):
                    try:
                        title_link = f"https://www.themoviedb.org/movie/{int(tmdb_id_val)}"
                        card_content_html += f"<h6><a href='{title_link}' target='_blank' style='color: white; text-decoration: none; font-weight: bold;'>{title_text_plain}</a></h6>"
                    except ValueError:
                        card_content_html += f"<h6 style='font-weight: bold; color: white;'>{title_text_plain}</h6>"
                else:
                    card_content_html += f"<h6 style='font-weight: bold; color: white;'>{title_text_plain}</h6>"

                genres_val = str(movie_data.get(C.GENRES_COL, "N/A"))
                year_val = movie_data.get(C.RELEASE_YEAR_COL)
                year_display = int(year_val) if pd.notna(year_val) and year_val != 0 else "N/A"
                card_content_html += f"<p style='font-size: 0.8em; color: #f0f0f0; margin-bottom: 5px;'>{genres_val} | {year_display}</p>"

                tmdb_avg_val = movie_data.get(C.VOTE_AVERAGE_COL)
                if pd.notna(tmdb_avg_val):
                    try: card_content_html += f"<small style='color: #f0f0f0;'>TMDB: {pd.to_numeric(tmdb_avg_val, errors='coerce'):.1f}/10</small><br>"
                    except: pass
                
                if is_personalized and 'estimated_score' in movie_data and pd.notna(movie_data['estimated_score']):
                    display_score = movie_data['estimated_score']
                    if hasattr(C, 'RATINGS_SCALE') and (C.RATINGS_SCALE == (1,5) or C.RATINGS_SCALE == (0.5, 5.0)) : display_score *=2
                    card_content_html += f"<small style='color: white;'>Pour vous: <strong>{display_score:.1f}/10</strong></small>"

                # Div principal de la carte avec style rouge vif
                # Le contenu textuel est injecté, puis le selectbox est ajouté en dessous (hors du div principal stylisé, mais dans le même conteneur de carte)
                st.markdown(
                    f"<div style='background-color: #E50914; color: white; border-radius: 8px; "
                    f"padding: 15px; height: 300px; /* Hauteur pour le contenu textuel */ "
                    f"box-shadow: 3px 3px 8px rgba(0,0,0,0.4); overflow-y: auto; margin-bottom: 8px;'>"
                    f"{card_content_html}"
                    f"</div>",
                    unsafe_allow_html=True
                )
                
                # Widget de notation (placé en dessous du bloc rouge)
                movie_id_current = movie_data.get(C.ITEM_ID_COL)
                if enable_rating_for_user_id is not None and movie_id_current is not None:
                    rating_opts = [None, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
                    fmt_fn = lambda x: "Notez" if x is None else f"{x} ★"
                    clean_carousel_id_for_key = re.sub(r'\W+', '', carousel_id.lower())[:15]
                    # Clé de widget unique
                    rating_key = f"rating_{clean_carousel_id_for_key}_{str(movie_id_current)}_{current_page}_{str(enable_rating_for_user_id)}"
                    
                    current_buffered_rating = st.session_state.logged_in_user_ratings_buffer.get(movie_id_current)
                    idx_rating = rating_opts.index(current_buffered_rating) if current_buffered_rating in rating_opts else 0
                    
                    previous_rating_in_buffer = st.session_state.logged_in_user_ratings_buffer.get(movie_id_current)

                    user_rating_input = st.selectbox(
                        label="Votre note :", options=rating_opts, index=idx_rating,
                        format_func=fmt_fn, key=rating_key, label_visibility="collapsed"
                    )
                    
                    if user_rating_input != previous_rating_in_buffer: # Mise à jour du buffer seulement si changement
                        if user_rating_input is not None:
                            st.session_state.logged_in_user_ratings_buffer[movie_id_current] = user_rating_input
                        elif movie_id_current in st.session_state.logged_in_user_ratings_buffer and user_rating_input is None:
                            del st.session_state.logged_in_user_ratings_buffer[movie_id_current]
                        # Pas de st.rerun() ici pour éviter la redirection après chaque note.
                        # Le widget selectbox se met à jour visuellement, le buffer est mis à jour en arrière-plan.
                        # L'utilisateur cliquera sur le bouton "Enregistrer" pour sauvegarder toutes les notes du buffer.
    st.markdown("<hr style='margin-top: 0.5rem; margin-bottom: 1rem;'>", unsafe_allow_html=True)


# --- Session State & Sidebar --- (Identique)
if 'active_page' not in st.session_state: st.session_state.active_page = "general"
if 'current_user_id' not in st.session_state: st.session_state.current_user_id = None
if 'new_user_ratings' not in st.session_state: st.session_state.new_user_ratings = {}
if 'new_user_name_input' not in st.session_state: st.session_state.new_user_name_input = ''
if 'last_selected_user_id' not in st.session_state: st.session_state.last_selected_user_id = None
if 'instant_reco_model_new_user' not in st.session_state: st.session_state.instant_reco_model_new_user = None
if 'new_user_id_generated' not in st.session_state: st.session_state.new_user_id_generated = None
if 'logged_in_user_ratings_buffer' not in st.session_state:st.session_state.logged_in_user_ratings_buffer = {}

st.sidebar.header("Filtres et Options")
all_genres_list_sidebar = ["Tous les genres"]
if not df_items_global_app.empty and hasattr(C, 'GENRES_COL') and C.GENRES_COL in df_items_global_app.columns:
    try:
        genres_series = df_items_global_app[C.GENRES_COL].fillna('').astype(str)
        s_genres = genres_series.str.split('|').explode()
        unique_sidebar_genres = sorted([ g.strip() for g in s_genres.unique() if g.strip() and g.strip().lower() != '(no genres listed)' ])
        if unique_sidebar_genres: all_genres_list_sidebar.extend(unique_sidebar_genres)
    except Exception as e_g_sb: print(f"Erreur sidebar (liste genres): {e_g_sb}"); st.sidebar.error("Erreur chargement genres.")
selected_genre_sidebar = st.sidebar.selectbox("Filtrer par genre :", all_genres_list_sidebar, key="genre_filter_sb")

slider_min, slider_max, current_slider_val = 1900, pd.Timestamp.now().year, (1900, pd.Timestamp.now().year)
if not df_items_global_app.empty and hasattr(C, 'RELEASE_YEAR_COL') and C.RELEASE_YEAR_COL in df_items_global_app.columns:
    valid_years = pd.to_numeric(df_items_global_app[C.RELEASE_YEAR_COL], errors='coerce').dropna()
    if not valid_years.empty:
        calc_min, calc_max = int(valid_years.min()), int(valid_years.max())
        if calc_min <= calc_max and calc_min > 1800: slider_min = calc_min
        if calc_max <= pd.Timestamp.now().year + 5: slider_max = calc_max
        current_slider_val = (slider_min, slider_max)
if slider_max < slider_min: slider_max = slider_min
selected_year_range_sidebar = st.sidebar.slider("Filtrer par année :", min_value=slider_min, max_value=slider_max, value=current_slider_val, key="year_filter_sb")

st.sidebar.markdown("---")
st.sidebar.header("👤 Espace Utilisateur")
user_opts = ["Explorer les Films"]
can_sel_existing = not df_ratings_global_app.empty and hasattr(C, 'USER_ID_COL') and \
                   C.USER_ID_COL in df_ratings_global_app.columns and \
                   not df_ratings_global_app[C.USER_ID_COL].empty
if can_sel_existing: user_opts.append("Se connecter (Profil Existant)")
user_opts.append("Créer un Nouveau Profil")
idx_radio = 0
if st.session_state.active_page == "general": idx_radio = user_opts.index("Explorer les Films") if "Explorer les Films" in user_opts else 0
elif st.session_state.active_page in ["new_user_profiling", "new_user_instant_recs"]: idx_radio = user_opts.index("Créer un Nouveau Profil") if "Créer un Nouveau Profil" in user_opts else 0
elif st.session_state.active_page == "user_specific" and "Se connecter (Profil Existant)" in user_opts: idx_radio = user_opts.index("Se connecter (Profil Existant)")
user_sel_opt = st.sidebar.radio("Choisissez une option :", user_opts, key="user_sel_main_radio", index=idx_radio)

orig_page, orig_uid = st.session_state.active_page, st.session_state.current_user_id
if user_sel_opt == "Explorer les Films":
    st.session_state.active_page, st.session_state.current_user_id = "general", None
elif user_sel_opt == "Créer un Nouveau Profil":
    if st.session_state.active_page != "new_user_instant_recs": st.session_state.active_page = "new_user_profiling"
    st.session_state.current_user_id = "new_user_temp"
    if orig_page not in ["new_user_profiling", "new_user_instant_recs"]:
        st.session_state.new_user_ratings, st.session_state.new_user_name_input = {}, ''
        st.session_state.instant_reco_model_new_user, st.session_state.new_user_id_generated = None, None
elif user_sel_opt == "Se connecter (Profil Existant)" and can_sel_existing:
    st.session_state.active_page = "user_specific"
    uids_list = sorted(df_ratings_global_app[C.USER_ID_COL].unique()) if not df_ratings_global_app.empty else []
    last_id_sel = st.session_state.last_selected_user_id
    if st.session_state.current_user_id is None or st.session_state.current_user_id == "new_user_temp":
        st.session_state.current_user_id = last_id_sel if last_id_sel in uids_list else (uids_list[0] if uids_list else None)
if st.session_state.active_page != orig_page or st.session_state.current_user_id != orig_uid: st.rerun()

uid_for_reco = None
if st.session_state.active_page == "user_specific": # Logique de sélection User ID identique
    if user_sel_opt == "Se connecter (Profil Existant)" and can_sel_existing:
        uids_avail = sorted(df_ratings_global_app[C.USER_ID_COL].unique()) if not df_ratings_global_app.empty else []
        user_profiles_map = {}
        user_profiles_path = C.DATA_PATH / getattr(C, 'USER_PROFILES_FILENAME', 'user_profiles.csv')
        if os.path.exists(user_profiles_path):
            try:
                df_profiles = pd.read_csv(user_profiles_path)
                if 'userId' in df_profiles.columns and 'userName' in df_profiles.columns:
                     user_profiles_map = pd.Series(df_profiles.userName.values, index=df_profiles.userId).to_dict()
            except Exception as e_pf: print(f"Erreur chargement user_profiles.csv: {e_pf}")
        if uids_avail:
            disp_opts_func = lambda uid_val: f"{user_profiles_map.get(uid_val, 'Utilisateur')} (ID: {uid_val})"
            current_selection_uid = st.session_state.current_user_id
            idx_sel_box = uids_avail.index(current_selection_uid) if current_selection_uid in uids_avail else 0
            if not uids_avail: st.sidebar.warning("Aucun utilisateur existant à sélectionner.")
            else:
                if current_selection_uid not in uids_avail:
                    current_selection_uid = uids_avail[0]
                    st.session_state.current_user_id = current_selection_uid
                uid_sel_box_val = st.sidebar.selectbox(f"Profil ID:", options=uids_avail, format_func=disp_opts_func, index=idx_sel_box, key="uid_sel_box")
                if st.session_state.current_user_id != uid_sel_box_val:
                    st.session_state.current_user_id = uid_sel_box_val
                    st.session_state.last_selected_user_id = uid_sel_box_val
                    st.rerun()
            uid_for_reco = st.session_state.current_user_id
        else: st.sidebar.warning("Aucun profil utilisateur existant.")
    elif st.session_state.current_user_id not in [None, "new_user_temp"]: uid_for_reco = st.session_state.current_user_id


# --- Logique d'Affichage Principal (avec bandeaux paginés et titres modifiés) ---
if st.session_state.active_page == "general":
    st.header("🌟 Explorer les Films")
    yr_min, yr_max = selected_year_range_sidebar[0], selected_year_range_sidebar[1]
    genre_f_general = selected_genre_sidebar if selected_genre_sidebar != "Tous les genres" else None

    genre_suffix = f" : {genre_f_general}" if genre_f_general else ""
    top_tmdb_movies = get_top_overall_movies_tmdb(genre_filter=genre_f_general, year_min_filter=yr_min, year_max_filter=yr_max)
    display_movie_carousel("top_tmdb", f"🏆 Top Évaluations Générales{genre_suffix}", top_tmdb_movies)

    top_documentaries = get_top_genre_movies_tmdb(genre="Documentary", year_min_filter=yr_min, year_max_filter=yr_max)
    display_movie_carousel("top_documentaries", "📹 Documentaires Incontournables", top_documentaries)

    hidden_gems = get_hidden_gems_movies(genre_filter=genre_f_general, year_min_filter=yr_min, year_max_filter=yr_max)
    display_movie_carousel("hidden_gems", f"💎 Potentiel Inexploré{genre_suffix}", hidden_gems)

elif st.session_state.active_page == "user_specific" and uid_for_reco is not None:
    user_display_name = f"Utilisateur {uid_for_reco}"
    user_profiles_path_main = C.DATA_PATH / getattr(C, 'USER_PROFILES_FILENAME', 'user_profiles.csv')
    if os.path.exists(user_profiles_path_main):
        try:
            df_profiles_main = pd.read_csv(user_profiles_path_main)
            if 'userId' in df_profiles_main.columns and 'userName' in df_profiles_main.columns:
                profile_entry = df_profiles_main[df_profiles_main['userId'].astype(type(uid_for_reco)) == uid_for_reco]
                if not profile_entry.empty: user_display_name = profile_entry['userName'].iloc[0]
        except Exception as e_pf_main: print(f"Erreur lecture user_profiles.csv pour nom affichage: {e_pf_main}")
    st.header(f"Recommandations Pour Vous, {user_display_name} (ID: {uid_for_reco})")
    yr_min_p, yr_max_p = selected_year_range_sidebar[0], selected_year_range_sidebar[1]
    genre_f_perso = selected_genre_sidebar if selected_genre_sidebar != "Tous les genres" else None

    models_p_dir = str(C.DATA_PATH / 'recs')
    avail_model_files = [f for f in os.listdir(models_p_dir) if f.endswith('.p') and not 'personalized' in f.lower()] if os.path.exists(models_p_dir) and os.path.isdir(models_p_dir) else []
    
    if not avail_model_files: st.error(f"Aucun modèle général pré-entraîné trouvé dans {models_p_dir}.")
    else:
        user_profile_for_titles = explanations.get_user_profile_for_explanation(uid_for_reco, top_n_movies=2, min_rating=3.5)
        model_types_config = [
            ("content_based", "content_based", "Suggestions Basées sur le Contenu"),
            ("user_based", "user_based", "Aimé par des Profils Similaires"),
            ("svd", "svd", "Découvertes Algorithmiques Pour Vous") # Titre SVD mis à jour
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
                        carousel_title_final = f"Parce que vous avez aimé {anchor_movie['title']} ({anchor_movie['rating']:.1f}/5) :"
                elif model_key == "user_based":
                    if len(user_profile_for_titles) >= 2:
                        movie1, movie2 = user_profile_for_titles[0], user_profile_for_titles[1]
                        carousel_title_final = f"Les fans de {movie1['title']} et {movie2['title']} apprécient aussi :"
                    elif len(user_profile_for_titles) == 1:
                        movie1 = user_profile_for_titles[0]
                        carousel_title_final = f"Les fans de {movie1['title']} ({movie1['rating']:.1f}/5) apprécient aussi :"
                elif model_key == "svd":
                    if not recs_data.empty and 'explanation' in recs_data.columns and pd.notna(recs_data['explanation'].iloc[0]):
                        first_explanation = str(recs_data['explanation'].iloc[0])
                        if "dans la lignée de" in first_explanation.lower() and len(first_explanation) < 70 :
                             carousel_title_final = first_explanation
                        elif user_profile_for_titles:
                            anchor_movie = user_profile_for_titles[0]
                            carousel_title_final = f"Inspiré par vos goûts pour {anchor_movie['title']} :"
                    elif user_profile_for_titles:
                        anchor_movie = user_profile_for_titles[0]
                        carousel_title_final = f"Dans un style similaire à {anchor_movie['title']} :"
                
                carousel_id_perso = f"{model_key}_{str(uid_for_reco).replace('.', '_')}" # Assurer ID valide pour session_state
                display_movie_carousel(
                    carousel_id_perso, carousel_title_final, recs_data,
                    enable_rating_for_user_id=uid_for_reco, is_personalized=True
                )
            else: st.warning(f"Aucun modèle de type '{file_keyword}' trouvé.")

        if st.session_state.logged_in_user_ratings_buffer:
            st.markdown("---")
            num_buffered_ratings = len(st.session_state.logged_in_user_ratings_buffer)
            cols_save_button = st.columns([0.3, 0.4, 0.3])
            with cols_save_button[1]:
                 if st.button(f"✔️ Enregistrer mes {num_buffered_ratings} nouvelle(s) note(s)", key="save_logged_in_ratings_final", use_container_width=True):
                    ratings_to_save_list = [] # Logique de sauvegarde identique
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
                            st.success(f"{len(ratings_to_save_list)} note(s) enregistrée(s) !")
                            st.session_state.logged_in_user_ratings_buffer = {}
                            st.rerun()
                        except Exception as e_save_rating:
                            st.error(f"Erreur sauvegarde notes : {e_save_rating}")

# --- Section Nouveau Utilisateur (Profilage et Recos Instantanées) --- (Identique à la version précédente)
elif st.session_state.active_page == "new_user_profiling":
    st.header("👤 Créez votre profil de goûts")
    st.write("Pour nous aider à comprendre vos préférences, veuillez indiquer votre nom et noter quelques films.")
    new_user_name = st.text_input("Quel est votre nom ?", st.session_state.get('new_user_name_input', ''))
    st.session_state.new_user_name_input = new_user_name
    movies_for_profiling_pool = df_items_global_app.copy() if not df_items_global_app.empty else pd.DataFrame()
    sample_size = 20; min_prefs_needed = 5
    movies_to_display_df = pd.DataFrame()
    if not movies_for_profiling_pool.empty:
        if hasattr(C, 'POPULARITY_COL') and C.POPULARITY_COL in movies_for_profiling_pool.columns and not movies_for_profiling_pool[C.POPULARITY_COL].isnull().all():
            movies_for_profiling_pool[C.POPULARITY_COL] = pd.to_numeric(movies_for_profiling_pool[C.POPULARITY_COL], errors='coerce').fillna(0)
            movies_to_display_df_temp = movies_for_profiling_pool.sort_values(by=C.POPULARITY_COL, ascending=False).head(150)
            if len(movies_to_display_df_temp) >= sample_size: movies_to_display_df = movies_to_display_df_temp.sample(n=sample_size, random_state=42)
        if movies_to_display_df.empty:
            if len(movies_for_profiling_pool) >= sample_size: movies_to_display_df = movies_for_profiling_pool.sample(n=sample_size, random_state=42)
            else: movies_to_display_df = movies_for_profiling_pool.copy()
    if movies_to_display_df.empty: st.error("Impossible de charger des films pour le profilage.")
    else:
        with st.form(key="new_user_profiling_form"):
            st.subheader("Notez les films suivants")
            st.caption("Laissez sur 'Pas de note' si vous ne connaissez pas ou n'avez pas d'avis.")
            for _, row in movies_to_display_df.iterrows():
                movie_id, title = row[C.ITEM_ID_COL], row[C.LABEL_COL]
                rating_key = f"rating_select_{movie_id}"
                current_rating = st.session_state.new_user_ratings.get(movie_id)
                col1, col2 = st.columns([3,2])
                with col1:
                    st.write(f"**{title}**")
                    if hasattr(C, 'GENRES_COL') and C.GENRES_COL in row and pd.notna(row[C.GENRES_COL]): st.caption(f"Genres: {row[C.GENRES_COL]}")
                with col2:
                    opts = [None, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
                    fmt_fn = lambda x: "Pas de note" if x is None else f"{x} ★"
                    idx = opts.index(current_rating) if current_rating in opts else 0
                    new_r = st.selectbox(f"Note pour {title}:", opts, index=idx, format_func=fmt_fn, key=rating_key, label_visibility="collapsed")
                    if new_r is not None: st.session_state.new_user_ratings[movie_id] = new_r
                    elif movie_id in st.session_state.new_user_ratings: del st.session_state.new_user_ratings[movie_id]
            form_submitted = st.form_submit_button("✔️ Enregistrer et voir premières suggestions")
        final_prefs = st.session_state.new_user_ratings.copy()
        num_total_prefs = len(final_prefs)
        st.info(f"Vous avez fourni {num_total_prefs} note(s). Au moins {min_prefs_needed} sont requises.")
        if form_submitted:
            if not new_user_name.strip(): st.warning("Veuillez entrer votre nom.")
            elif num_total_prefs < min_prefs_needed: st.warning(f"Veuillez noter au moins {min_prefs_needed} films.")
            else:
                try:
                    current_ratings_df_for_id = df_ratings_global_app
                    new_user_id_val = (current_ratings_df_for_id[C.USER_ID_COL].max() + 1) if not current_ratings_df_for_id.empty else 1
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
                        st.success(f"Profil pour {new_user_name.strip()} (ID: {new_user_id_val}) sauvegardé.")
                    st.info("Calcul de vos premières suggestions...")
                    cb_features_instant = ["Genre_binary", "Year_of_release"]
                    actual_cb_features_instant = []
                    if hasattr(C, 'GENRES_COL') and C.GENRES_COL in models_df_items_global.columns : actual_cb_features_instant.append("Genre_binary")
                    if hasattr(C, 'RELEASE_YEAR_COL') and C.RELEASE_YEAR_COL in models_df_items_global.columns: actual_cb_features_instant.append("Year_of_release")
                    if not actual_cb_features_instant:
                        st.warning("Pas assez de features disponibles pour les suggestions instantanées.")
                        st.session_state.active_page = "general"; st.rerun()
                    else:
                        cb_model_instant = ContentBased(features_methods=actual_cb_features_instant, regressor_method='linear')
                        reader = Reader(rating_scale=C.RATINGS_SCALE)
                        instant_user_id_for_train = -1
                        ratings_for_instant_model_df = pd.DataFrame([{
                            C.USER_ID_COL: instant_user_id_for_train, C.ITEM_ID_COL: mid, C.RATING_COL: rval
                        } for mid, rval in final_prefs.items()])
                        data_instant = Dataset.load_from_df(ratings_for_instant_model_df, reader)
                        trainset_instant = data_instant.build_full_trainset()
                        cb_model_instant.fit(trainset_instant)
                        st.session_state.instant_reco_model_new_user = cb_model_instant
                        st.session_state.active_page = "new_user_instant_recs"
                        st.rerun()
                except Exception as e_profile_processing:
                    st.error(f"Erreur lors de la création de votre profil : {e_profile_processing}")

elif st.session_state.active_page == "new_user_instant_recs":
    st.header("🎉 Vos Premières Suggestions de Films !")
    st.caption("Basées sur les notes que vous venez de donner.")
    model_instance = st.session_state.get('instant_reco_model_new_user')
    new_user_ratings_keys = st.session_state.get('new_user_ratings', {}).keys()
    generated_user_id_for_pred = -1
    if model_instance and models_df_items_global is not None and not models_df_items_global.empty:
        all_movie_ids_global = models_df_items_global[C.ITEM_ID_COL].unique()
        movies_to_predict_ids = [mid for mid in all_movie_ids_global if mid not in new_user_ratings_keys]
        if not movies_to_predict_ids: st.info("Pas d'autres films à suggérer pour le moment.")
        else:
            preds_list_instant = []
            sample_size_instant_pred = min(len(movies_to_predict_ids), 200)
            for item_id_to_predict in random.sample(movies_to_predict_ids, sample_size_instant_pred):
                try:
                    prediction = model_instance.predict(uid=generated_user_id_for_pred, iid=item_id_to_predict)
                    preds_list_instant.append({C.ITEM_ID_COL: prediction.iid, 'estimated_score': prediction.est})
                except: continue
            if preds_list_instant:
                recs_instant_df_raw = pd.DataFrame(preds_list_instant).sort_values(by='estimated_score', ascending=False).head(N_RECOS_PERSONNALISEES_TOTAL_FETCH)
                if not recs_instant_df_raw.empty and not df_items_global_app.empty:
                    final_recs_instant_df = pd.merge(
                        recs_instant_df_raw,
                        df_items_global_app[[C.ITEM_ID_COL, C.LABEL_COL, C.GENRES_COL, C.RELEASE_YEAR_COL, C.VOTE_AVERAGE_COL, C.TMDB_ID_COL]],
                        on=C.ITEM_ID_COL, how='left'
                    )
                    display_movie_carousel("instant_recs_new_user", "Suggestions rapides pour vous :", final_recs_instant_df, is_personalized=True)
                elif not recs_instant_df_raw.empty: st.write(recs_instant_df_raw)
                else: st.info("Impossible de générer des suggestions instantanées.")
            else: st.info("Aucune suggestion instantanée n'a pu être générée.")
    else: st.warning("Le modèle de suggestion instantanée n'est pas disponible.")
    if st.button("Explorer d'autres films"):
        st.session_state.active_page = "general"
        st.session_state.new_user_ratings, st.session_state.new_user_name_input = {}, ''
        st.session_state.instant_reco_model_new_user, st.session_state.new_user_id_generated = None, None
        st.rerun()
else:
    if st.session_state.active_page not in ["general", "user_specific", "new_user_profiling", "new_user_instant_recs"]:
        st.session_state.active_page = "general"; st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("Projet Recommender Systems MLSMM2156")