# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import time

# Modules du projet
import constants as C_module
C = C_module.Constant()
import content
import recommender
import profile_manager

<<<<<<< Updated upstream
# --- Fonction d'aide pour la sérialisation JSON ---
def convert_to_native_python_types(data):
    if isinstance(data, dict): return {k: convert_to_native_python_types(v) for k, v in data.items()}
    if isinstance(data, list): return [convert_to_native_python_types(i) for i in data]
    if isinstance(data, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)): return int(data)
    if isinstance(data, (np.float_, np.float16, np.float32, np.float64)): return float(data)
    if isinstance(data, (np.bool_)): return bool(data)
    if isinstance(data, pd.Timestamp): return data.isoformat()
    return data
=======
# --- Constantes ---
N_RECOS_PERSONNALISEES_TOTAL_FETCH = 50
N_INSTANT_RECOS_NEW_USER = 10
CARDS_PER_ROW = 5
PROFILING_SAMPLE_SIZE = 20 
PROFILING_MOVIES_PER_GENRE = 2 
PROFILING_MIN_VOTES_THRESHOLD = 500 
>>>>>>> Stashed changes

# --- Chargement des données globales ---
df_items_global = pd.DataFrame()
df_ratings_global = pd.DataFrame() 
try:
    from models import df_items_global as df_items_model_load, df_ratings_global as df_ratings_model_load
    df_items_global = df_items_model_load
    df_ratings_global = df_ratings_model_load
    if df_items_global.empty: from loaders import load_items; df_items_global = load_items()
    if df_ratings_global.empty and C.USER_ID_COL in df_ratings_global.columns : 
        from loaders import load_ratings; df_ratings_global = load_ratings()
except ImportError:
    print(f"app.py: ERREUR Import depuis models.py. Chargement direct des données.")
    from loaders import load_items, load_ratings
    df_items_global = load_items(); df_ratings_global = load_ratings()
if df_items_global is None or df_items_global.empty : st.error("ERREUR CRITIQUE: Impossible de charger df_items_global."); st.stop()
if df_ratings_global is None: df_ratings_global = pd.DataFrame()

<<<<<<< Updated upstream
N_TOP_GENERAL = 30; N_RECOS_PER_MODEL_TYPE = 20
st.set_page_config(page_title="Recommandations de Films", layout="wide")
st.title("🎬 Système de Recommandation de Films Simplifié")

# --- Initialisation de l'état de session ---
default_page = "general_tops"
if 'active_page' not in st.session_state: st.session_state.active_page = default_page
if 'current_user_prenom' not in st.session_state: st.session_state.current_user_prenom = None
if 'current_user_id_persistent' not in st.session_state: st.session_state.current_user_id_persistent = None
if 'selected_movielens_user_id' not in st.session_state: st.session_state.selected_movielens_user_id = None
if 'new_user_prenom' not in st.session_state: st.session_state.new_user_prenom = ""
if 'new_user_ratings_input' not in st.session_state: st.session_state.new_user_ratings_input = {}
if 'confirm_delete_action' not in st.session_state: st.session_state.confirm_delete_action = None
if 'source_page_for_recs' not in st.session_state: st.session_state.source_page_for_recs = None # Pour savoir d'où on vient pour la page de recos

# --- Fonctions de récupération de données (Tops Généraux) ---
=======

st.set_page_config(page_title="Recommandation de Films", layout="wide", initial_sidebar_state="expanded")

# --- STYLING GLOBAL ---
st.markdown("""
<style>
    html, body, [class*="st-"] { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .main .block-container {
        padding-top: 2rem; padding-bottom: 2rem; padding-left: 3rem; padding-right: 3rem;
        background-color: #f0f2f6;
    }
    [data-testid="stSidebar"] { background-color: #e8eaed; padding: 10px; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #333; }
    h1, h2 { color: #2c3e50; }
    h3 { color: #34495e; margin-top: 2rem; margin-bottom: 1rem;
         border-bottom: 2px solid #bdc3c7; padding-bottom: 0.5rem; }
    div[data-testid="stHorizontalBlock"] > div[data-testid="stButton"] button {
        background-color: #7f8c8d; color: white; border-radius: 5px; border: none; padding: 5px 10px;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="stButton"] button:hover { background-color: #95a5a6; }
    div[data-testid="stHorizontalBlock"] > div[data-testid="stButton"] button:disabled {
        background-color: #bdc3c7; color: #7f8c8d;
    }
    div[data-testid="stSelectbox"] > div { border-radius: 5px; }
    a { text-decoration: none; }
</style>
""", unsafe_allow_html=True)

st.title("🎬 Système de Recommandation de Films")


# --- Fonctions de récupération de données (inchangées) ---
>>>>>>> Stashed changes
@st.cache_data
def get_top_overall_movies_tmdb(n=N_TOP_GENERAL, year_min_filter=None, year_max_filter=None):
    # ... (Identique)
    if df_items_global.empty or not hasattr(C, 'VOTE_AVERAGE_COL') or C.VOTE_AVERAGE_COL not in df_items_global.columns: return pd.DataFrame()
    items_to_consider = df_items_global.copy(); items_to_consider[C.VOTE_AVERAGE_COL] = pd.to_numeric(items_to_consider[C.VOTE_AVERAGE_COL], errors='coerce')
    vote_count_col_to_use = 'temp_vote_count_overall'; min_tmdb_votes_threshold = 100
    if hasattr(C, 'VOTE_COUNT_COL') and C.VOTE_COUNT_COL in items_to_consider.columns: items_to_consider[C.VOTE_COUNT_COL] = pd.to_numeric(items_to_consider[C.VOTE_COUNT_COL], errors='coerce').fillna(0); vote_count_col_to_use = C.VOTE_COUNT_COL
    else: items_to_consider[vote_count_col_to_use] = min_tmdb_votes_threshold 
    if year_min_filter is not None and year_max_filter is not None and hasattr(C, 'RELEASE_YEAR_COL') and C.RELEASE_YEAR_COL in items_to_consider.columns:
        items_to_consider[C.RELEASE_YEAR_COL] = pd.to_numeric(items_to_consider[C.RELEASE_YEAR_COL], errors='coerce').fillna(0)
        items_to_consider = items_to_consider[(items_to_consider[C.RELEASE_YEAR_COL] >= year_min_filter) & (items_to_consider[C.RELEASE_YEAR_COL] <= year_max_filter)]
    if items_to_consider.empty: return pd.DataFrame()
    qualified_movies = items_to_consider[items_to_consider[vote_count_col_to_use] >= min_tmdb_votes_threshold]
    top_movies_source = qualified_movies if not qualified_movies.empty else items_to_consider
    top_movies_df = top_movies_source.sort_values(by=C.VOTE_AVERAGE_COL, ascending=False).head(n)
    cols_out = [C.ITEM_ID_COL, C.LABEL_COL]; 
    if hasattr(C, 'GENRES_COL') and C.GENRES_COL in top_movies_df.columns: cols_out.append(C.GENRES_COL)
    if hasattr(C, 'RELEASE_YEAR_COL') and C.RELEASE_YEAR_COL in top_movies_df.columns: cols_out.append(C.RELEASE_YEAR_COL)
    if hasattr(C, 'VOTE_AVERAGE_COL') and C.VOTE_AVERAGE_COL in top_movies_df.columns: cols_out.append(C.VOTE_AVERAGE_COL)
    if vote_count_col_to_use == C.VOTE_COUNT_COL and hasattr(C, 'VOTE_COUNT_COL') and C.VOTE_COUNT_COL in top_movies_df.columns: cols_out.append(C.VOTE_COUNT_COL)
    return top_movies_df[[col for col in cols_out if col in top_movies_df.columns]]

@st.cache_data
def get_top_genre_movies_tmdb(genre, n=N_TOP_GENERAL, year_min_filter=None, year_max_filter=None):
    # ... (Identique) ...
    if df_items_global.empty or not hasattr(C, 'GENRES_COL') or C.GENRES_COL not in df_items_global.columns or not hasattr(C, 'VOTE_AVERAGE_COL') or C.VOTE_AVERAGE_COL not in df_items_global.columns: return pd.DataFrame()
    genre_movies_df_initial = df_items_global[df_items_global[C.GENRES_COL].astype(str).str.contains(re.escape(genre), case=False, na=False, regex=True)]
    if genre_movies_df_initial.empty: return pd.DataFrame()
    items_to_consider = genre_movies_df_initial.copy(); items_to_consider[C.VOTE_AVERAGE_COL] = pd.to_numeric(items_to_consider[C.VOTE_AVERAGE_COL], errors='coerce')
    vote_count_col_to_use = 'temp_vote_count_genre'; min_tmdb_votes_threshold_genre = 50
    if hasattr(C, 'VOTE_COUNT_COL') and C.VOTE_COUNT_COL in items_to_consider.columns: items_to_consider[C.VOTE_COUNT_COL] = pd.to_numeric(items_to_consider[C.VOTE_COUNT_COL], errors='coerce').fillna(0); vote_count_col_to_use = C.VOTE_COUNT_COL
    else: items_to_consider[vote_count_col_to_use] = min_tmdb_votes_threshold_genre
    if year_min_filter is not None and year_max_filter is not None and hasattr(C, 'RELEASE_YEAR_COL') and C.RELEASE_YEAR_COL in items_to_consider.columns:
        items_to_consider[C.RELEASE_YEAR_COL] = pd.to_numeric(items_to_consider[C.RELEASE_YEAR_COL], errors='coerce').fillna(0)
        items_to_consider = items_to_consider[(items_to_consider[C.RELEASE_YEAR_COL] >= year_min_filter) & (items_to_consider[C.RELEASE_YEAR_COL] <= year_max_filter)]
    if items_to_consider.empty: return pd.DataFrame()
    qualified_movies = items_to_consider[items_to_consider[vote_count_col_to_use] >= min_tmdb_votes_threshold_genre]
    top_movies_source = qualified_movies if not qualified_movies.empty else items_to_consider
    top_genre_df = top_movies_source.sort_values(by=C.VOTE_AVERAGE_COL, ascending=False).head(n)
    cols_out = [C.ITEM_ID_COL, C.LABEL_COL] 
    if hasattr(C, 'GENRES_COL') and C.GENRES_COL in top_genre_df.columns: cols_out.append(C.GENRES_COL)
    if hasattr(C, 'RELEASE_YEAR_COL') and C.RELEASE_YEAR_COL in top_genre_df.columns: cols_out.append(C.RELEASE_YEAR_COL)
    if hasattr(C, 'VOTE_AVERAGE_COL') and C.VOTE_AVERAGE_COL in top_genre_df.columns: cols_out.append(C.VOTE_AVERAGE_COL)
    if vote_count_col_to_use == C.VOTE_COUNT_COL and hasattr(C, 'VOTE_COUNT_COL') and C.VOTE_COUNT_COL in top_genre_df.columns: cols_out.append(C.VOTE_COUNT_COL)
    return top_genre_df[[col for col in cols_out if col in top_genre_df.columns]]

<<<<<<< Updated upstream
# --- Fonctions d'affichage ---
def display_movie_cards(df_to_display):
    # ... (Identique) ...
    if df_to_display.empty: return
    item_id_col, label_col, genres_col, year_col = C.ITEM_ID_COL, C.LABEL_COL, C.GENRES_COL, C.RELEASE_YEAR_COL
    tmdb_id_col, vote_avg_col_orig = C.TMDB_ID_COL, C.VOTE_AVERAGE_COL 
    can_make_tmdb_links = (hasattr(C, 'TMDB_ID_COL') and tmdb_id_col in df_items_global.columns and item_id_col in df_items_global.columns)
    for _, row in df_to_display.iterrows():
        with st.container(): 
            title_display = str(row.get(label_col, "Titre Inconnu"))
            movie_id_current = row.get(item_id_col)
            if can_make_tmdb_links and pd.notna(movie_id_current):
                tmdb_id_series = df_items_global.loc[df_items_global[item_id_col] == movie_id_current, tmdb_id_col]
                if not tmdb_id_series.empty and pd.notna(tmdb_id_series.iloc[0]):
                    try: tmdb_id_val_int = int(tmdb_id_series.iloc[0]); title_display = f"[{title_display}](https://www.themoviedb.org/movie/{tmdb_id_val_int})"
                    except ValueError: pass 
            col_info, col_pred_score, col_global_score = st.columns([6, 2, 2]) 
            with col_info:
                st.markdown(f"**{title_display}**", unsafe_allow_html=True)
                genres_val_display = str(row.get(genres_col, "N/A")) 
                year_val = row.get(year_col); year_display = int(year_val) if pd.notna(year_val) and year_val != 0 and str(year_val) != "0" else "N/A"
                st.caption(f"Genres: {genres_val_display} | Année: {year_display}")
            with col_pred_score:
                if 'estimated_score' in row and pd.notna(row['estimated_score']): score_display = row['estimated_score']; st.markdown(f"<div style='font-size: small; text-align: center;'>Prédiction:<br><b>{score_display:.1f}/10</b></div>", unsafe_allow_html=True)
            with col_global_score:
                tmdb_avg_val_display = row.get('tmdb_vote_average', row.get(vote_avg_col_orig))
                if pd.notna(tmdb_avg_val_display):
                    try: numeric_avg_val = float(tmdb_avg_val_display); st.markdown(f"<div style='font-size: small; text-align: center;'>Note Globale:<br><b>{numeric_avg_val:.2f}/10</b></div>", unsafe_allow_html=True)
                    except ValueError: st.markdown(f"<div style='font-size: small; text-align: center;'>Note Globale:<br>N/A</div>", unsafe_allow_html=True)
            st.divider()

def display_movie_recommendations_section(recs_df, title="Recommandations", page_size=N_RECOS_PER_MODEL_TYPE):
    # ... (Identique) ...
    st.subheader(title)
    if recs_df.empty: st.info("Aucune recommandation à afficher pour cette sélection."); return
    display_df_all = recs_df.copy() 
    if 'estimated_score' in display_df_all.columns:
        display_df_all['estimated_score'] = pd.to_numeric(display_df_all['estimated_score'], errors='coerce')
        if C.RATINGS_SCALE[1] == 5.0: display_df_all['estimated_score'] = display_df_all['estimated_score'] * 2.0
        display_df_all['estimated_score'] = display_df_all['estimated_score'].round(1)
    vote_avg_col_name_display = 'tmdb_vote_average' 
    if hasattr(C, 'VOTE_AVERAGE_COL') and C.VOTE_AVERAGE_COL in display_df_all.columns and vote_avg_col_name_display not in display_df_all.columns: display_df_all = display_df_all.rename(columns={C.VOTE_AVERAGE_COL: vote_avg_col_name_display})
    if vote_avg_col_name_display in display_df_all.columns: display_df_all[vote_avg_col_name_display] = pd.to_numeric(display_df_all[vote_avg_col_name_display], errors='coerce').round(2)    
    display_movie_cards(display_df_all.head(page_size))

# --- Sidebar ---
# ... (Filtres globaux identiques) ...
st.sidebar.header("Filtres Globaux")
all_genres_list_sb = ["Tous les genres"]; unique_genres_for_ui_sb = []
if not df_items_global.empty and hasattr(C, 'GENRES_COL') and C.GENRES_COL in df_items_global.columns:
    try:
        s_genres_exploded_sb = df_items_global[C.GENRES_COL].dropna().astype(str).str.split('|').explode()
        unique_genres_for_ui_sb = sorted([g.strip() for g in s_genres_exploded_sb.unique() if g.strip() and g.strip().lower() not in ['', '(no genres listed)']])
        if unique_genres_for_ui_sb: all_genres_list_sb.extend(unique_genres_for_ui_sb)
    except Exception as e_g_sb_filter: print(f"Erreur sidebar (filtre genres): {e_g_sb_filter}")
selected_genre_filter_sb = st.sidebar.selectbox("Filtrer par genre :", all_genres_list_sb, key="genre_filter_sb_main_v9")
slider_min_year_sb, slider_max_year_sb = 1900, pd.Timestamp.now().year; current_slider_val_year_sb = (slider_min_year_sb, slider_max_year_sb)
if not df_items_global.empty and hasattr(C, 'RELEASE_YEAR_COL') and C.RELEASE_YEAR_COL in df_items_global.columns:
    valid_years_sb = pd.to_numeric(df_items_global[C.RELEASE_YEAR_COL], errors='coerce').replace(0, pd.NA).dropna()
    if not valid_years_sb.empty:
        calc_min_sb, calc_max_sb = int(valid_years_sb.min()), int(valid_years_sb.max())
        if calc_min_sb > 1800 and calc_max_sb >= calc_min_sb : slider_min_year_sb, slider_max_year_sb = calc_min_sb, calc_max_sb; current_slider_val_year_sb = (slider_min_year_sb, slider_max_year_sb) 
if slider_max_year_sb < slider_min_year_sb: slider_max_year_sb = slider_min_year_sb
selected_year_range_filter_sb = st.sidebar.slider("Filtrer par année de sortie :", min_value=slider_min_year_sb, max_value=slider_max_year_sb, value=current_slider_val_year_sb, key="year_filter_sb_main_v9")

st.sidebar.markdown("---"); st.sidebar.header("👤 Mode d'Exploration")
nav_options_map_sb = {
    "Recommandations Générales (Tops)": "general_tops",
    "Créer Mon Profil Utilisateur": "new_user_creation_form",
    "Mes Profils Personnels (par Prénom)": "select_personal_profile_area",
    "Explorer Utilisateurs MovieLens (par ID)": "select_movielens_user_area"
}
nav_options_display_sb = list(nav_options_map_sb.keys())
# Conditionner les options de navigation
if not profile_manager.get_all_mapped_prenoms(): # Utiliser une fonction qui liste les prénoms mappés
    if "Mes Profils Personnels (par Prénom)" in nav_options_display_sb:
        nav_options_display_sb.remove("Mes Profils Personnels (par Prénom)")
if df_ratings_global.empty or C.USER_ID_COL not in df_ratings_global.columns or df_ratings_global[C.USER_ID_COL].nunique() == 0:
    if "Explorer Utilisateurs MovieLens (par ID)" in nav_options_display_sb:
        nav_options_display_sb.remove("Explorer Utilisateurs MovieLens (par ID)")

active_page_for_nav_index = st.session_state.get('active_page', "general_tops")
# Si on est sur la page d'affichage, le mode de la sidebar doit correspondre à la source
if active_page_for_nav_index == "display_user_recommendations":
    source_page_map_key = st.session_state.get("source_page_for_recs") # Ex: "select_personal_profile_area"
    for display_name_map, page_key_map_val in nav_options_map_sb.items():
        if page_key_map_val == source_page_map_key:
            active_page_for_nav_index = page_key_map_val # Pour que le selectbox de la sidebar soit correct
            break
try: nav_index_sb = nav_options_display_sb.index(next(k for k, v in nav_options_map_sb.items() if v == active_page_for_nav_index))
except (StopIteration, ValueError): nav_index_sb = 0

selected_main_nav_option_sb = st.sidebar.selectbox("Choisissez une section :", nav_options_display_sb, index=nav_index_sb, key="main_nav_selectbox_v9")

new_target_page_main = nav_options_map_sb.get(selected_main_nav_option_sb)
if new_target_page_main and new_target_page_main != st.session_state.active_page:
    st.session_state.active_page = new_target_page_main
    # Reset user selection if changing main mode
    if new_target_page_main != "display_user_recommendations": # Ne pas reset si on est déjà sur la page de recos
        st.session_state.current_user_prenom = None; st.session_state.current_user_id_persistent = None
        st.session_state.selected_movielens_user_id = None
    if new_target_page_main == "new_user_creation_form": # Reset spécifique pour le formulaire
        st.session_state.new_user_prenom = ""; st.session_state.new_user_ratings_input = {}
    st.rerun()

# --- Logique d'Affichage Principal ---
if st.session_state.active_page == "general_tops":
    # ... (Identique)
    st.header("🏆 Tops des Films"); yr_min, yr_max = selected_year_range_filter_sb[0], selected_year_range_filter_sb[1]
    df_to_show_tops = pd.DataFrame()
    if selected_genre_filter_sb == "Tous les genres": df_to_show_tops = get_top_overall_movies_tmdb(n=N_TOP_GENERAL, year_min_filter=yr_min, year_max_filter=yr_max)
    else: df_to_show_tops = get_top_genre_movies_tmdb(genre=selected_genre_filter_sb, n=N_TOP_GENERAL, year_min_filter=yr_min, year_max_filter=yr_max)
    if df_to_show_tops.empty: st.info(f"Aucun film trouvé pour les critères : Genre '{selected_genre_filter_sb}', Années {yr_min}-{yr_max}.")
    else: display_movie_cards(df_to_show_tops)

elif st.session_state.active_page == "new_user_creation_form":
    # ... (Identique à la version précédente)
    st.header("👤 Créez Votre Profil Utilisateur"); st.write("Notez quelques films pour que nous puissions vous ajouter à notre base et générer des recommandations après le prochain entraînement des modèles.")
    st.session_state.new_user_prenom = st.text_input("Votre Prénom (optionnel, pour affichage) :", value=st.session_state.new_user_prenom, key="new_user_prenom_input_v9")
    movies_to_rate_df = pd.DataFrame(); 
    if not df_items_global.empty: sample_size = min(C.NEW_USER_MOVIES_TO_RATE_COUNT, len(df_items_global)); movies_to_rate_df = df_items_global.sample(n=sample_size, random_state=42) if len(df_items_global) >= sample_size else df_items_global.copy()
    st.subheader(f"Veuillez noter au moins {C.NEW_USER_MIN_RATINGS_FOR_SAVE} films parmi les suivants :")
    if movies_to_rate_df.empty: st.error("Impossible de charger des films pour la notation.")
    else:
        with st.form(key="new_user_ratings_form_v9"): 
            for idx, row in movies_to_rate_df.iterrows():
                movie_id = row[C.ITEM_ID_COL]; title = row.get(C.LABEL_COL, "Titre Inconnu")
                if movie_id not in st.session_state.new_user_ratings_input: st.session_state.new_user_ratings_input[movie_id] = "Pas de note"
                rating_options = ["Pas de note"] + [str(r) for r in np.arange(C.RATINGS_SCALE[0], C.RATINGS_SCALE[1] + 0.5, 0.5)]; current_rating_val = st.session_state.new_user_ratings_input[movie_id]
                st.session_state.new_user_ratings_input[movie_id] = st.selectbox(f"{title}", options=rating_options, index=rating_options.index(current_rating_val) if current_rating_val in rating_options else 0, key=f"new_rating_{movie_id}_v9")
            submit_new_user_ratings = st.form_submit_button("✔️ Enregistrer mes notes")
        if submit_new_user_ratings:
            user_ratings_list_for_save = []; new_user_id = profile_manager.get_next_available_user_id(); current_timestamp = int(time.time())
            for movie_id_rated, rating_str in st.session_state.new_user_ratings_input.items():
                if rating_str != "Pas de note": user_ratings_list_for_save.append({C.USER_ID_COL: new_user_id, C.ITEM_ID_COL: movie_id_rated, C.RATING_COL: float(rating_str), C.TIMESTAMP_COL: current_timestamp})
            if len(user_ratings_list_for_save) < C.NEW_USER_MIN_RATINGS_FOR_SAVE: st.warning(f"Veuillez noter au moins {C.NEW_USER_MIN_RATINGS_FOR_SAVE} films.")
=======
@st.cache_data
def get_hidden_gems_movies(n=N_RECOS_PERSONNALISEES_TOTAL_FETCH, genre_filter=None, year_min_filter=None, year_max_filter=None,
                           min_vote_average_initial=6.0, min_votes_initial=10,
                           num_novel_candidates=100, excluded_genre="Documentary"):
    if df_items_global_app.empty or not hasattr(C, 'VOTE_COUNT_COL') or C.VOTE_COUNT_COL not in df_items_global_app.columns \
       or not hasattr(C, 'VOTE_AVERAGE_COL') or C.VOTE_AVERAGE_COL not in df_items_global_app.columns:
        return pd.DataFrame()
    items_to_consider = df_items_global_app.copy()
    if genre_filter and genre_filter != "Tous les genres" and hasattr(C, 'GENRES_COL') and C.GENRES_COL in items_to_consider.columns:
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
        by=[C.VOTE_COUNT_COL, C.VOTE_AVERAGE_COL], ascending=[True, False]
    ).head(num_novel_candidates)
    if novel_candidates_df.empty: return pd.DataFrame()
    top_hidden_gems_df = novel_candidates_df.sort_values(
        by=C.VOTE_AVERAGE_COL, ascending=False
    ).head(n)
    cols_out = [C.ITEM_ID_COL, C.LABEL_COL, C.GENRES_COL, C.RELEASE_YEAR_COL, C.VOTE_AVERAGE_COL, C.VOTE_COUNT_COL, C.TMDB_ID_COL]
    return top_hidden_gems_df[[col for col in cols_out if col in top_hidden_gems_df.columns]]

@st.cache_data
def get_diverse_movies_for_profiling(df_all_items, total_movies_to_select=PROFILING_SAMPLE_SIZE,
                                     movies_per_genre_target=PROFILING_MOVIES_PER_GENRE,
                                     min_votes_threshold=PROFILING_MIN_VOTES_THRESHOLD):
    if df_all_items.empty: return pd.DataFrame()
    required_cols = [C.GENRES_COL, C.VOTE_COUNT_COL, C.VOTE_AVERAGE_COL, C.ITEM_ID_COL, C.LABEL_COL]
    if not all(col in df_all_items.columns for col in required_cols):
        return df_all_items.sample(min(total_movies_to_select, len(df_all_items)), random_state=42)
    df_items = df_all_items.copy()
    df_items[C.VOTE_COUNT_COL] = pd.to_numeric(df_items[C.VOTE_COUNT_COL], errors='coerce').fillna(0)
    df_items[C.VOTE_AVERAGE_COL] = pd.to_numeric(df_items[C.VOTE_AVERAGE_COL], errors='coerce').fillna(0)
    df_items_filtered = df_items[df_items[C.VOTE_COUNT_COL] >= min_votes_threshold]
    if len(df_items_filtered) < total_movies_to_select :
        df_items_filtered = df_items.sort_values(by=C.VOTE_COUNT_COL, ascending=False).head(total_movies_to_select * 5)
        if df_items_filtered.empty: return df_items.sample(min(total_movies_to_select, len(df_items)), random_state=43)
    all_genres_lists = df_items_filtered[C.GENRES_COL].astype(str).str.split('|')
    excluded_profiling_genres = {'(no genres listed)', '', 'imax', 'film-noir'}
    unique_genres = sorted(list(set(
        genre.strip() for sublist in all_genres_lists for genre in sublist
        if genre.strip().lower() not in excluded_profiling_genres )))
    if not unique_genres: return df_items_filtered.sort_values(by=C.VOTE_COUNT_COL, ascending=False).head(total_movies_to_select)
    selected_movie_ids = set(); final_movie_list = []
    random.shuffle(unique_genres)
    for genre in unique_genres:
        if len(selected_movie_ids) >= total_movies_to_select: break
        genre_movies = df_items_filtered[
            df_items_filtered[C.GENRES_COL].astype(str).str.contains(re.escape(genre), case=False, na=False) &
            ~df_items_filtered[C.ITEM_ID_COL].isin(selected_movie_ids) ]
        genre_movies_sorted = genre_movies.sort_values(by=[C.VOTE_COUNT_COL, C.VOTE_AVERAGE_COL], ascending=[False, False])
        added_count = 0
        for _, movie_row in genre_movies_sorted.iterrows():
            if len(selected_movie_ids) < total_movies_to_select and added_count < movies_per_genre_target:
                selected_movie_ids.add(movie_row[C.ITEM_ID_COL]); final_movie_list.append(movie_row); added_count += 1
            else: break
    final_movie_list_df = pd.DataFrame(final_movie_list)
    if len(selected_movie_ids) < total_movies_to_select:
        remaining_needed = total_movies_to_select - len(selected_movie_ids)
        additional_movies = df_items_filtered[~df_items_filtered[C.ITEM_ID_COL].isin(selected_movie_ids)]
        additional_movies_sorted = additional_movies.sort_values(by=[C.VOTE_COUNT_COL, C.VOTE_AVERAGE_COL], ascending=[False, False])
        final_movie_list_df = pd.concat([final_movie_list_df, additional_movies_sorted.head(remaining_needed)], ignore_index=True)
    if not final_movie_list_df.empty and C.ITEM_ID_COL in final_movie_list_df.columns:
        final_movie_list_df = final_movie_list_df.drop_duplicates(subset=[C.ITEM_ID_COL], keep='first')
    return final_movie_list_df.head(total_movies_to_select)

# --- Fonctions d'affichage des bandeaux ---
def display_movie_carousel(carousel_id, carousel_title, movies_df,
                           enable_rating_for_user_id=None,
                           num_cards_to_show_at_once=CARDS_PER_ROW,
                           is_personalized=False):
    if movies_df.empty: return
    if f'{carousel_id}_page' not in st.session_state: st.session_state[f'{carousel_id}_page'] = 0
    current_page = st.session_state[f'{carousel_id}_page']
    total_movies = len(movies_df)
    total_pages = math.ceil(total_movies / num_cards_to_show_at_once)
    start_index = current_page * num_cards_to_show_at_once
    end_index = start_index + num_cards_to_show_at_once
    movies_to_display_on_page = movies_df.iloc[start_index:end_index]
    title_col_ratio = 0.80; button_col_ratio = (1.0 - title_col_ratio) / 2
    if total_pages > 1: col_title, col_prev, col_next = st.columns([title_col_ratio, button_col_ratio, button_col_ratio])
    else: col_title, col_prev, col_next = st, None, None
    with col_title: st.subheader(carousel_title)
    if total_pages > 1 and col_prev is not None and col_next is not None:
        with col_prev:
            if st.button("⬅️", key=f"prev_{carousel_id}", use_container_width=True, disabled=(current_page == 0)):
                st.session_state[f'{carousel_id}_page'] -= 1; st.rerun()
        with col_next:
            if st.button("➡️", key=f"next_{carousel_id}", use_container_width=True, disabled=(current_page >= total_pages - 1)):
                st.session_state[f'{carousel_id}_page'] += 1; st.rerun()
    elif col_prev is not None and col_next is not None: # Cache les boutons si une seule page
         with col_prev: st.empty()
         with col_next: st.empty()
    if not movies_to_display_on_page.empty:
        num_actual_cards_on_page = len(movies_to_display_on_page)
        cols_cards = st.columns(num_actual_cards_on_page)
        for idx, (_, movie_data) in enumerate(movies_to_display_on_page.iterrows()):
            with cols_cards[idx]:
                card_content_html = ""
                title_text_plain = str(movie_data.get(C.LABEL_COL, "Titre Inconnu"))
                tmdb_id_val = movie_data.get(C.TMDB_ID_COL)
                if pd.notna(tmdb_id_val):
                    try:
                        title_link = f"https://www.themoviedb.org/movie/{int(tmdb_id_val)}"
                        card_content_html += f"<h6><a href='{title_link}' target='_blank' style='color: white; text-decoration: none; font-weight: bold;'>{title_text_plain}</a></h6>"
                    except ValueError: card_content_html += f"<h6 style='font-weight: bold; color: white;'>{title_text_plain}</h6>"
                else: card_content_html += f"<h6 style='font-weight: bold; color: white;'>{title_text_plain}</h6>"
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
                st.markdown(
                    f"<div style='background-color: #CD5C5C; color: white; border-radius: 8px; "
                    f"padding: 15px; height: 300px; box-shadow: 3px 3px 8px rgba(0,0,0,0.4); "
                    f"overflow-y: auto; margin-bottom: 8px; display: flex; flex-direction: column; justify-content: flex-start;'>"
                    f"<div>{card_content_html}</div></div>", unsafe_allow_html=True)
                movie_id_current = movie_data.get(C.ITEM_ID_COL)
                if enable_rating_for_user_id is not None and movie_id_current is not None:
                    rating_opts = [None, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
                    fmt_fn = lambda x: "Notez" if x is None else f"{x} ★"
                    clean_carousel_id_for_key = re.sub(r'\W+', '', carousel_id.lower())[:15]
                    rating_key = f"rating_{clean_carousel_id_for_key}_{str(movie_id_current)}_{current_page}_{str(enable_rating_for_user_id)}"
                    current_buffered_rating = st.session_state.logged_in_user_ratings_buffer.get(movie_id_current)
                    idx_rating = rating_opts.index(current_buffered_rating) if current_buffered_rating in rating_opts else 0
                    previous_rating_in_buffer = st.session_state.logged_in_user_ratings_buffer.get(movie_id_current)
                    user_rating_input = st.selectbox(label="Votre note :", options=rating_opts, index=idx_rating,
                                                     format_func=fmt_fn, key=rating_key, label_visibility="collapsed")
                    if user_rating_input != previous_rating_in_buffer:
                        if user_rating_input is not None: st.session_state.logged_in_user_ratings_buffer[movie_id_current] = user_rating_input
                        elif movie_id_current in st.session_state.logged_in_user_ratings_buffer and user_rating_input is None:
                            del st.session_state.logged_in_user_ratings_buffer[movie_id_current]
    st.markdown("<hr style='margin-top: 0.5rem; margin-bottom: 1rem;'>", unsafe_allow_html=True)

# --- Session State (Initialisation) ---
if 'active_page' not in st.session_state: st.session_state.active_page = "general"
if 'current_user_id' not in st.session_state: st.session_state.current_user_id = None
if 'new_user_ratings' not in st.session_state: st.session_state.new_user_ratings = {}
if 'new_user_name_input' not in st.session_state: st.session_state.new_user_name_input = ''
if 'last_selected_user_id' not in st.session_state: st.session_state.last_selected_user_id = None
if 'instant_reco_model_new_user' not in st.session_state: st.session_state.instant_reco_model_new_user = None
if 'new_user_id_generated' not in st.session_state: st.session_state.new_user_id_generated = None
if 'logged_in_user_ratings_buffer' not in st.session_state:st.session_state.logged_in_user_ratings_buffer = {}
if 'active_search_query' not in st.session_state: st.session_state.active_search_query = "" # Pour stocker la requête de recherche active
if 'search_results_df' not in st.session_state: st.session_state.search_results_df = pd.DataFrame()

# --- Sidebar ---
st.sidebar.header("Navigation & Filtres")
search_input_val = st.sidebar.text_input("Rechercher un film :", key="sidebar_search_field",
                                         value=st.session_state.active_search_query if st.session_state.active_page == "search_results" else "")
if st.sidebar.button("Rechercher", key="search_trigger_button"):
    if search_input_val.strip():
        st.session_state.active_search_query = search_input_val.strip()
        search_term_cleaned = re.escape(st.session_state.active_search_query.lower())
        results = df_items_global_app[
            df_items_global_app[C.LABEL_COL].str.lower().str.contains(search_term_cleaned, na=False)]
        st.session_state.search_results_df = results
        st.session_state.active_page = "search_results"
        st.rerun()
    else: # Recherche vide
        st.session_state.active_search_query = ""
        st.session_state.search_results_df = pd.DataFrame()
        if st.session_state.active_page == "search_results": # Si on était sur la page de recherche, retour à general
            st.session_state.active_page = "general"; st.rerun()

st.sidebar.markdown("---")
all_genres_list_sidebar = ["Tous les genres"]
if not df_items_global_app.empty and hasattr(C, 'GENRES_COL') and C.GENRES_COL in df_items_global_app.columns:
    try:
        genres_series = df_items_global_app[C.GENRES_COL].fillna('').astype(str)
        s_genres = genres_series.str.split('|').explode()
        unique_sidebar_genres = sorted([ g.strip() for g in s_genres.unique() if g.strip() and g.strip().lower() != '(no genres listed)' ])
        if unique_sidebar_genres: all_genres_list_sidebar.extend(unique_sidebar_genres)
    except Exception as e_g_sb: print(f"Erreur sidebar (liste genres): {e_g_sb}")
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
can_sel_existing = not df_ratings_global_app.empty and hasattr(C, 'USER_ID_COL') and C.USER_ID_COL in df_ratings_global_app.columns and not df_ratings_global_app[C.USER_ID_COL].empty
if can_sel_existing: user_opts.append("Se connecter (Profil Existant)")
user_opts.append("Créer un Nouveau Profil")
idx_radio = 0
if st.session_state.active_page == "general": idx_radio = user_opts.index("Explorer les Films") if "Explorer les Films" in user_opts else 0
elif st.session_state.active_page == "search_results": idx_radio = user_opts.index("Explorer les Films") # Garder l'option générale sélectionnée
elif st.session_state.active_page in ["new_user_profiling", "new_user_instant_recs"]: idx_radio = user_opts.index("Créer un Nouveau Profil") if "Créer un Nouveau Profil" in user_opts else 0
elif st.session_state.active_page == "user_specific" and "Se connecter (Profil Existant)" in user_opts: idx_radio = user_opts.index("Se connecter (Profil Existant)")
user_sel_opt = st.sidebar.radio("Choisissez une option :", user_opts, key="user_sel_main_radio", index=idx_radio)

orig_page, orig_uid = st.session_state.active_page, st.session_state.current_user_id
page_changed_by_radio = False
if user_sel_opt == "Explorer les Films" and st.session_state.active_page != "general":
    st.session_state.active_page = "general"; st.session_state.current_user_id = None; page_changed_by_radio = True
elif user_sel_opt == "Créer un Nouveau Profil":
    if st.session_state.active_page not in ["new_user_profiling", "new_user_instant_recs"]:
        st.session_state.active_page = "new_user_profiling"; page_changed_by_radio = True
    st.session_state.current_user_id = "new_user_temp"
    if orig_page not in ["new_user_profiling", "new_user_instant_recs"]:
        st.session_state.new_user_ratings, st.session_state.new_user_name_input = {}, ''
        st.session_state.instant_reco_model_new_user, st.session_state.new_user_id_generated = None, None
elif user_sel_opt == "Se connecter (Profil Existant)" and can_sel_existing and st.session_state.active_page != "user_specific":
    st.session_state.active_page = "user_specific"; page_changed_by_radio = True
    uids_list = sorted(df_ratings_global_app[C.USER_ID_COL].unique()) if not df_ratings_global_app.empty else []
    last_id_sel = st.session_state.last_selected_user_id
    if st.session_state.current_user_id is None or st.session_state.current_user_id == "new_user_temp":
        st.session_state.current_user_id = last_id_sel if last_id_sel in uids_list else (uids_list[0] if uids_list else None)

if page_changed_by_radio or (st.session_state.active_page == "user_specific" and st.session_state.current_user_id != orig_uid) :
    if st.session_state.active_page != "search_results": # Ne pas effacer la recherche si on change d'utilisateur en étant sur la page de recherche
        st.session_state.active_search_query = ""
        st.session_state.search_results_df = pd.DataFrame()
    st.rerun()

uid_for_reco = None
if st.session_state.active_page == "user_specific":
    if user_sel_opt == "Se connecter (Profil Existant)" and can_sel_existing: # Logique de sélection de profil inchangée
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
>>>>>>> Stashed changes
            else:
                df_new_ratings_to_append = pd.DataFrame(user_ratings_list_for_save)
                df_new_ratings_to_append[C.USER_ID_COL] = df_new_ratings_to_append[C.USER_ID_COL].astype(int); df_new_ratings_to_append[C.ITEM_ID_COL] = df_new_ratings_to_append[C.ITEM_ID_COL].astype(int)
                df_new_ratings_to_append[C.RATING_COL] = df_new_ratings_to_append[C.RATING_COL].astype(float); df_new_ratings_to_append[C.TIMESTAMP_COL] = df_new_ratings_to_append[C.TIMESTAMP_COL].astype(int)
                save_ratings_success = profile_manager.append_ratings_to_csv(df_new_ratings_to_append)
                if save_ratings_success:
                    user_prenom_for_map = st.session_state.new_user_prenom.strip()
                    if user_prenom_for_map: profile_manager.add_user_prenom_mapping(user_prenom_for_map, new_user_id)
                    st.success(f"Votre profil a été créé avec l'UserID: {new_user_id}. Prénom associé : '{user_prenom_for_map if user_prenom_for_map else 'Non spécifié'}'")
                    st.info("Vos notes ont été ajoutées à la base de données générale. Un ré-entraînement des modèles via `training.py` est nécessaire."); 
                    st.session_state.new_user_prenom = ""; st.session_state.new_user_ratings_input = {}
                else: st.error("Une erreur est survenue lors de la sauvegarde de vos notes.")

elif st.session_state.active_page == "select_personal_profile_area":
    st.header("Mes Profils Personnels (par Prénom)")
    mapped_prenoms = profile_manager.get_all_mapped_prenoms() # Liste de (userID, Prénom)
    if not mapped_prenoms:
        st.info("Aucun profil personnel avec prénom mappé n'a été créé via l'application.")
        if st.button("Créer un nouveau profil", key="goto_create_from_perso_empty"):
            st.session_state.active_page = "new_user_creation_form"; st.rerun()
    else:
        prenom_options_main = [""] + [p[1] for p in mapped_prenoms]
        selected_perso_prenom_main = st.selectbox("Choisissez votre profil :", prenom_options_main, index=0, key="main_select_perso_profile_v2")
        if selected_perso_prenom_main:
            user_id_for_selection = profile_manager.get_user_id_by_prenom(selected_perso_prenom_main)
            if st.button(f"Voir les recommandations pour {selected_perso_prenom_main}", key=f"view_recs_perso_btn_{selected_perso_prenom_main.replace(' ','_')}"):
                if user_id_for_selection is not None:
                    st.session_state.selected_display_user_id = user_id_for_selection
                    st.session_state.selected_display_user_prenom = selected_perso_prenom_main
                    st.session_state.active_page = "display_user_recommendations"
                    st.session_state.source_page_for_recs = "select_personal_profile_area"
                    st.rerun()
                else: st.error(f"UserID non trouvé pour {selected_perso_prenom_main}")

<<<<<<< Updated upstream
            # Option de suppression de mappage
            st.markdown("---")
            if st.button(f"Supprimer le mappage pour {selected_perso_prenom_main}", key=f"del_map_btn_main_{selected_perso_prenom_main.replace(' ','_')}"):
                st.session_state.confirm_delete_action = selected_perso_prenom_main; st.rerun()
    
    if st.session_state.confirm_delete_action: # Logique de confirmation de suppression de mappage
        prenom_to_delete_map = st.session_state.confirm_delete_action
        st.warning(f"Supprimer le mappage pour {prenom_to_delete_map} ? Ses notes resteront dans ratings.csv.")
        col_confirm_map, col_cancel_map = st.columns(2)
        with col_confirm_map:
            if st.button("Oui, supprimer mappage", key=f"confirm_del_map_yes_main_{prenom_to_delete_map.replace(' ','_')}"):
                success_del_map = profile_manager.delete_user_prenom_mapping(prenom_to_delete_map)
                st.session_state.confirm_delete_action = None 
                if success_del_map: st.success(f"Mappage pour {prenom_to_delete_map} supprimé.")
                else: st.error(f"Échec suppression mappage pour {prenom_to_delete_map}.")
                st.rerun()
        with col_cancel_map:
            if st.button("Annuler", key=f"cancel_del_map_no_main_{prenom_to_delete_map.replace(' ','_')}"):
                st.session_state.confirm_delete_action = None; st.info("Suppression annulée."); st.rerun()

elif st.session_state.active_page == "select_movielens_user_area":
    st.header("Explorer Utilisateurs MovieLens (par ID)")
    if df_ratings_global.empty or C.USER_ID_COL not in df_ratings_global.columns: st.error("Données ratings MovieLens non disponibles.")
    else:
        movielens_user_ids_main = sorted(pd.to_numeric(df_ratings_global[C.USER_ID_COL], errors='coerce').dropna().unique().astype(int))
        if not movielens_user_ids_main: st.info("Aucun UserID MovieLens trouvé.")
        else:
            min_ml_uid_main, max_ml_uid_main = min(movielens_user_ids_main), max(movielens_user_ids_main)
            default_ml_id_main = movielens_user_ids_main[0] # Prend le premier comme défaut

            selected_ml_id_main_area = st.number_input(
                f"Entrez un UserID MovieLens ({min_ml_uid_main}-{max_ml_uid_main}) :",
                min_value=min_ml_uid_main, max_value=max_ml_uid_main,
                value=st.session_state.get('selected_movielens_user_id_input_val', default_ml_id_main), # Mémoriser la dernière entrée
                step=1, key="main_input_ml_user_v2"
            )
            st.session_state.selected_movielens_user_id_input_val = selected_ml_id_main_area # Mémoriser pour la prochaine fois

            if st.button(f"Voir les recommandations pour UserID {selected_ml_id_main_area}", key=f"view_recs_ml_btn_main_{selected_ml_id_main_area}"):
                if selected_ml_id_main_area in movielens_user_ids_main:
                    st.session_state.selected_display_user_id = selected_ml_id_main_area
                    st.session_state.selected_display_user_prenom = f"MovieLens UserID {selected_ml_id_main_area}"
                    st.session_state.active_page = "display_user_recommendations"
                    st.session_state.source_page_for_recs = "select_movielens_user_area"
                    st.rerun()
                else: st.error(f"L'UserID {selected_ml_id_main_area} n'est pas valide.")

elif st.session_state.active_page == "display_user_recommendations":
    # ... (Identique à la version précédente, s'assure d'utiliser N_RECOS_PER_MODEL_TYPE)
    uid_to_display = st.session_state.get('selected_display_user_id')
    prenom_to_display = st.session_state.get('selected_display_user_prenom', f"UserID {uid_to_display}")
    if uid_to_display is None: # ... (Logique de fallback identique)
        st.warning("Aucun utilisateur sélectionné."); source_page = st.session_state.get("source_page_for_recs")
        btn_label_back = "Retour à la sélection de Profil Personnel" if source_page == "select_personal_profile_area" else \
                         "Retour à la sélection d'Utilisateur MovieLens" if source_page == "select_movielens_user_area" else "Retour à l'accueil"
        target_back_page = source_page if source_page else "general_tops"
        if st.button(btn_label_back, key="recs_back_btn_v2"): st.session_state.active_page = target_back_page; st.rerun()
        st.stop()
    st.header(f"Recommandations pour : {prenom_to_display} (ID: {uid_to_display})")
    st.caption("Ces recommandations sont basées sur les modèles généraux. Ré-exécutez `training.py` après avoir ajouté de nouveaux utilisateurs/notes.")
    general_models_to_show = {}; # ... (Logique de détection des modèles généraux identique)
    if C.MODELS_RECS_PATH.exists():
        for f_model in os.listdir(C.MODELS_RECS_PATH):
            if f_model.endswith('.p') and "_personalized" not in f_model.lower(): # Exclure tous les modèles personnalisés
                model_key = os.path.splitext(f_model)[0].lower()
                if "svd" in model_key and "SVD (Général)" not in general_models_to_show: general_models_to_show["SVD (Général)"] = f_model
                elif "content_based" in model_key and "Contenu Similaire (Général)" not in general_models_to_show: general_models_to_show["Contenu Similaire (Général)"] = f_model
                elif "user_based" in model_key and "Utilisateurs Similaires (Général)" not in general_models_to_show: general_models_to_show["Utilisateurs Similaires (Général)"] = f_model
    if not general_models_to_show: st.error("Aucun fichier modèle général trouvé. Exécutez `training.py`.")
    else: # ... (Logique d'affichage identique, utilisant N_RECOS_PER_MODEL_TYPE)
        yr_min_all, yr_max_all = selected_year_range_filter_sb[0], selected_year_range_filter_sb[1]; genre_f_all = selected_genre_filter_sb if selected_genre_filter_sb != "Tous les genres" else None
        any_recs_shown_all = False
        for model_name_display, model_file_name_gen in general_models_to_show.items():
            if os.path.exists(C.MODELS_RECS_PATH / model_file_name_gen):
                st.markdown("---"); recs_data_all = recommender.get_top_n_recommendations(user_id=uid_to_display,model_filename=model_file_name_gen,n=N_RECOS_PER_MODEL_TYPE, filter_genre=genre_f_all,filter_year_range=(yr_min_all, yr_max_all))
                display_movie_recommendations_section(recs_data_all, title=model_name_display, page_size=N_RECOS_PER_MODEL_TYPE)
                any_recs_shown_all = True
            else: st.warning(f"Modèle général '{model_name_display}' (fichier: {model_file_name_gen}) non trouvé.")
        if not any_recs_shown_all: st.info("Aucun modèle général n'a pu générer de recommandations.")
=======

# --- Logique d'Affichage Principal ---
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
            ("svd", "svd", "Découvertes Algorithmiques Pour Vous")
        ]
        for model_key, file_keyword, fallback_carousel_title in model_types_config:
            m_file = next((mfile for mfile in avail_model_files if file_keyword in mfile.lower() and 'final' in mfile.lower()), None)
            if not m_file: m_file = next((mfile for mfile in avail_model_files if file_keyword in mfile.lower()), None)
            if m_file:
                recs_data = recommender.get_top_n_recommendations(uid_for_reco, m_file, n=N_RECOS_PERSONNALISEES_TOTAL_FETCH,
                                                                  filter_genre=genre_f_perso, filter_year_range=(yr_min_p, yr_max_p))
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
                carousel_id_perso = f"{model_key}_{str(uid_for_reco).replace('.', '_')}"
                display_movie_carousel(carousel_id_perso, carousel_title_final, recs_data,
                                       enable_rating_for_user_id=uid_for_reco, is_personalized=True)
            else: st.warning(f"Aucun modèle de type '{file_keyword}' trouvé.")
        if st.session_state.logged_in_user_ratings_buffer:
            st.markdown("---"); num_buffered_ratings = len(st.session_state.logged_in_user_ratings_buffer)
            cols_save_button = st.columns([0.3, 0.4, 0.3])
            with cols_save_button[1]:
                 if st.button(f"✔️ Enregistrer mes {num_buffered_ratings} nouvelle(s) note(s)", key="save_logged_in_ratings_final", use_container_width=True):
                    ratings_to_save_list = [] # Logique de sauvegarde identique
                    current_ts = int(time.time()); user_id_to_save = uid_for_reco
                    for movie_id_key, rating_val_key in st.session_state.logged_in_user_ratings_buffer.items():
                        ratings_to_save_list.append({ C.USER_ID_COL: user_id_to_save, C.ITEM_ID_COL: movie_id_key,C.RATING_COL: rating_val_key, C.TIMESTAMP_COL: current_ts })
                    if ratings_to_save_list:
                        df_new_ratings_to_save_out = pd.DataFrame(ratings_to_save_list)
                        pending_ratings_filepath = C.EVIDENCE_PATH / getattr(C, 'NEW_RATINGS_PENDING_FILENAME', 'new_ratings_pending.csv')
                        file_exists_pending = os.path.exists(pending_ratings_filepath)
                        try:
                            df_new_ratings_to_save_out.to_csv(pending_ratings_filepath, mode='a', header=not file_exists_pending, index=False)
                            st.success(f"{len(ratings_to_save_list)} note(s) enregistrée(s) !")
                            st.session_state.logged_in_user_ratings_buffer = {}; st.rerun()
                        except Exception as e_save_rating: st.error(f"Erreur sauvegarde notes : {e_save_rating}")

elif st.session_state.active_page == "new_user_profiling":
    st.header("👤 Créez votre profil de goûts") # ... (code identique)
    st.write("Pour nous aider à comprendre vos préférences, veuillez indiquer votre nom et noter quelques films connus de genres variés.")
    new_user_name = st.text_input("Quel est votre nom ?", st.session_state.get('new_user_name_input', ''))
    st.session_state.new_user_name_input = new_user_name
    movies_to_display_df = get_diverse_movies_for_profiling(df_items_global_app, total_movies_to_select=PROFILING_SAMPLE_SIZE,
                                                            movies_per_genre_target=PROFILING_MOVIES_PER_GENRE,
                                                            min_votes_threshold=PROFILING_MIN_VOTES_THRESHOLD)
    if movies_to_display_df.empty: st.error("Impossible de charger une sélection diversifiée de films pour le profilage.")
    else:
        with st.form(key="new_user_profiling_form"):
            st.subheader("Notez les films suivants") # ... (reste du formulaire identique)
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
        num_total_prefs = len(final_prefs); min_prefs_needed = 5
        st.info(f"Vous avez fourni {num_total_prefs} note(s). Au moins {min_prefs_needed} sont requises.")
        if form_submitted: # ... (logique de soumission identique)
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
                        st.warning("Pas assez de features pour suggestions instantanées.")
                        st.session_state.active_page = "general"; st.rerun()
                    else:
                        cb_model_instant = ContentBased(features_methods=actual_cb_features_instant, regressor_method='linear')
                        reader = Reader(rating_scale=C.RATINGS_SCALE)
                        instant_user_id_for_train = -1
                        ratings_for_instant_model_df = pd.DataFrame([{ C.USER_ID_COL: instant_user_id_for_train, C.ITEM_ID_COL: mid, C.RATING_COL: rval } for mid, rval in final_prefs.items()])
                        data_instant = Dataset.load_from_df(ratings_for_instant_model_df, reader)
                        trainset_instant = data_instant.build_full_trainset()
                        cb_model_instant.fit(trainset_instant)
                        st.session_state.instant_reco_model_new_user = cb_model_instant
                        st.session_state.active_page = "new_user_instant_recs"; st.rerun()
                except Exception as e_profile_processing: st.error(f"Erreur création profil : {e_profile_processing}")


elif st.session_state.active_page == "new_user_instant_recs": # Logique identique
    st.header("🎉 Vos Premières Suggestions de Films !") # ... (code identique)
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
                    final_recs_instant_df = pd.merge(recs_instant_df_raw,
                                                     df_items_global_app[[C.ITEM_ID_COL, C.LABEL_COL, C.GENRES_COL, C.RELEASE_YEAR_COL, C.VOTE_AVERAGE_COL, C.TMDB_ID_COL]],
                                                     on=C.ITEM_ID_COL, how='left')
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

elif st.session_state.active_page == "search_results":
    st.header(f"🔍 Résultats pour : \"{st.session_state.active_search_query}\"")
    results_df = st.session_state.search_results_df
    if not results_df.empty:
        search_carousel_id = "search_results_carousel_" + re.sub(r'\W+', '', st.session_state.active_search_query.lower())[:20]
        display_movie_carousel(search_carousel_id, f"Résultats de la recherche ({len(results_df)} films trouvés)", results_df,
                               enable_rating_for_user_id=uid_for_reco, is_personalized=False)
        if uid_for_reco is not None and st.session_state.logged_in_user_ratings_buffer:
            st.markdown("---")
            num_buffered_ratings = len(st.session_state.logged_in_user_ratings_buffer)
            cols_save_button = st.columns([0.3, 0.4, 0.3])
            with cols_save_button[1]:
                 if st.button(f"✔️ Enregistrer mes {num_buffered_ratings} nouvelle(s) note(s)", key="save_search_ratings", use_container_width=True):
                    ratings_to_save_list = []; current_ts = int(time.time()); user_id_to_save = uid_for_reco
                    for movie_id_key, rating_val_key in st.session_state.logged_in_user_ratings_buffer.items():
                        ratings_to_save_list.append({ C.USER_ID_COL: user_id_to_save, C.ITEM_ID_COL: movie_id_key, C.RATING_COL: rating_val_key, C.TIMESTAMP_COL: current_ts })
                    if ratings_to_save_list:
                        df_new_ratings_to_save_out = pd.DataFrame(ratings_to_save_list)
                        pending_ratings_filepath = C.EVIDENCE_PATH / getattr(C, 'NEW_RATINGS_PENDING_FILENAME', 'new_ratings_pending.csv')
                        file_exists_pending = os.path.exists(pending_ratings_filepath)
                        try:
                            df_new_ratings_to_save_out.to_csv(pending_ratings_filepath, mode='a', header=not file_exists_pending, index=False)
                            st.success(f"{len(ratings_to_save_list)} note(s) enregistrée(s) !")
                            st.session_state.logged_in_user_ratings_buffer = {}; st.rerun()
                        except Exception as e_save_rating: st.error(f"Erreur sauvegarde notes : {e_save_rating}")
    else: st.info("Aucun film ne correspond à votre recherche.")
    if st.button("Retour à l'exploration générale", key="back_to_general_from_search"):
        st.session_state.active_page = "general"
        st.session_state.active_search_query = ""
        st.session_state.search_results_df = pd.DataFrame()
        st.rerun()

else:
    if st.session_state.active_page not in ["general", "user_specific", "new_user_profiling", "new_user_instant_recs", "search_results"]:
        st.session_state.active_page = "general"; st.rerun()
>>>>>>> Stashed changes

st.sidebar.markdown("---")
st.sidebar.info("Projet Recommender Systems")