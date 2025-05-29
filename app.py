import streamlit as st
import pandas as pd
import os
import re
import random
import time

import constants as C_module
C = C_module.Constant()
import content
import recommender
from loaders import load_items, load_ratings
from models import ContentBased, df_items_global as models_df_items_global
from surprise import Dataset, Reader 


# --- Constantes pour le nombre de recommandations ---
N_TOP_GENERAL = 30
N_RECOS_PERSONNALISEES_INITIAL_DISPLAY = 10
N_RECOS_PERSONNALISEES_PER_PAGE = 10
N_RECOS_PERSONNALISEES_TOTAL_FETCH = 30
N_INSTANT_RECOS_NEW_USER = 10 # Nombre de recos instantanées

# --- Chargement des données ---
try:
    df_items_global_app = load_items() # Renommé pour clarté, utilisé par l'UI générale
    df_ratings_global_app = load_ratings() # Renommé pour clarté
    if df_items_global_app.empty:
        print("app.py: ERREUR CRITIQUE - df_items_global_app est vide.")
    if df_ratings_global_app.empty:
        print("app.py: ERREUR CRITIQUE - df_ratings_global_app est vide.")
except Exception as e_load:
    print(f"app.py: ERREUR FATALE lors du chargement initial des données: {e_load}")
    _cols_items = [getattr(C, col_attr, col_attr.lower()) for col_attr in ['ITEM_ID_COL', 'LABEL_COL', 'GENRES_COL', 'RELEASE_YEAR_COL', 'TMDB_ID_COL', 'VOTE_AVERAGE_COL', 'VOTE_COUNT_COL'] if hasattr(C, col_attr)]
    df_items_global_app = pd.DataFrame(columns=_cols_items)
    _cols_ratings = [getattr(C, col_attr, col_attr.lower()) for col_attr in ['USER_ID_COL', 'ITEM_ID_COL', 'RATING_COL'] if hasattr(C, col_attr)]
    df_ratings_global_app = pd.DataFrame(columns=_cols_ratings)
    st.error("Erreur critique lors du chargement des données initiales.")


st.set_page_config(page_title="TMDB Recommender Assistance", layout="wide")
st.title("🎬 TMDB Recommender System Assistance")

# --- Fonctions de récupération de données (get_top_overall_movies_tmdb, get_top_genre_movies_tmdb) ---
# (Utilisent df_items_global_app)
@st.cache_data
def get_top_overall_movies_tmdb(n=N_TOP_GENERAL, year_min_filter=None, year_max_filter=None):
    if df_items_global_app.empty or not hasattr(C, 'VOTE_AVERAGE_COL') or C.VOTE_AVERAGE_COL not in df_items_global_app.columns:
        return pd.DataFrame()
    items_to_consider = df_items_global_app.copy()
    items_to_consider[C.VOTE_AVERAGE_COL] = pd.to_numeric(items_to_consider[C.VOTE_AVERAGE_COL], errors='coerce')
    vote_count_col_to_use = 'temp_vote_count_overall'
    if hasattr(C, 'VOTE_COUNT_COL') and C.VOTE_COUNT_COL in items_to_consider.columns:
        items_to_consider[C.VOTE_COUNT_COL] = pd.to_numeric(items_to_consider[C.VOTE_COUNT_COL], errors='coerce').fillna(0)
        vote_count_col_to_use = C.VOTE_COUNT_COL
    else: items_to_consider[vote_count_col_to_use] = 100 
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
    cols_out = [C.ITEM_ID_COL, C.LABEL_COL]
    if hasattr(C, 'GENRES_COL') and C.GENRES_COL in top_movies_df.columns: cols_out.append(C.GENRES_COL)
    if hasattr(C, 'RELEASE_YEAR_COL') and C.RELEASE_YEAR_COL in top_movies_df.columns: cols_out.append(C.RELEASE_YEAR_COL)
    if hasattr(C, 'VOTE_AVERAGE_COL') and C.VOTE_AVERAGE_COL in top_movies_df.columns: cols_out.append(C.VOTE_AVERAGE_COL)
    if vote_count_col_to_use == C.VOTE_COUNT_COL and hasattr(C, 'VOTE_COUNT_COL') and C.VOTE_COUNT_COL in top_movies_df.columns:
        cols_out.append(C.VOTE_COUNT_COL)
    return top_movies_df[[col for col in cols_out if col in top_movies_df.columns]]

@st.cache_data
def get_top_genre_movies_tmdb(genre, n=N_TOP_GENERAL, year_min_filter=None, year_max_filter=None):
    if df_items_global_app.empty or not hasattr(C, 'GENRES_COL') or C.GENRES_COL not in df_items_global_app.columns or \
       not hasattr(C, 'VOTE_AVERAGE_COL') or C.VOTE_AVERAGE_COL not in df_items_global_app.columns:
        return pd.DataFrame()
    genre_movies_df = df_items_global_app[
        df_items_global_app[C.GENRES_COL].astype(str).str.contains(re.escape(genre), case=False, na=False, regex=True)
    ]
    if genre_movies_df.empty: return pd.DataFrame()
    items_to_consider = genre_movies_df.copy()
    items_to_consider[C.VOTE_AVERAGE_COL] = pd.to_numeric(items_to_consider[C.VOTE_AVERAGE_COL], errors='coerce')
    vote_count_col_to_use = 'temp_vote_count_genre'
    if hasattr(C, 'VOTE_COUNT_COL') and C.VOTE_COUNT_COL in items_to_consider.columns:
        items_to_consider[C.VOTE_COUNT_COL] = pd.to_numeric(items_to_consider[C.VOTE_COUNT_COL], errors='coerce').fillna(0)
        vote_count_col_to_use = C.VOTE_COUNT_COL
    else: items_to_consider[vote_count_col_to_use] = 50 
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
    top_genre_df = top_movies_source.sort_values(by=C.VOTE_AVERAGE_COL, ascending=False).head(n)
    cols_out = [C.ITEM_ID_COL, C.LABEL_COL]
    if hasattr(C, 'GENRES_COL') and C.GENRES_COL in top_genre_df.columns: cols_out.append(C.GENRES_COL)
    if hasattr(C, 'RELEASE_YEAR_COL') and C.RELEASE_YEAR_COL in top_genre_df.columns: cols_out.append(C.RELEASE_YEAR_COL)
    if hasattr(C, 'VOTE_AVERAGE_COL') and C.VOTE_AVERAGE_COL in top_genre_df.columns: cols_out.append(C.VOTE_AVERAGE_COL)
    if vote_count_col_to_use == C.VOTE_COUNT_COL and hasattr(C, 'VOTE_COUNT_COL') and C.VOTE_COUNT_COL in top_genre_df.columns:
        cols_out.append(C.VOTE_COUNT_COL)
    return top_genre_df[[col for col in cols_out if col in top_genre_df.columns]]

# --- Fonctions d'affichage (display_movie_cards, display_movie_recommendations_section) ---
# Dans app.py

# Assurez-vous que pandas est importé : import pandas as pd

def display_movie_cards(df_to_display, enable_rating_for_user_id=None):
    if df_to_display.empty: return
    item_id_col, label_col, genres_col, year_col, tmdb_id_col, vote_avg_col = \
        C.ITEM_ID_COL, C.LABEL_COL, C.GENRES_COL, C.RELEASE_YEAR_COL, C.TMDB_ID_COL, C.VOTE_AVERAGE_COL
    
    can_make_links = (hasattr(C, 'TMDB_ID_COL') and tmdb_id_col in df_items_global_app.columns and \
                      item_id_col in df_items_global_app.columns and not df_items_global_app[tmdb_id_col].isnull().all())

    rating_opts = [None, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    fmt_fn = lambda x: "Pas de note" if x is None else f"{x} ★"

    for _, row in df_to_display.iterrows():
        with st.container(): # Utiliser st.container pour un meilleur regroupement visuel par carte
            st.markdown("---") # Ligne de séparation fine avant chaque carte

            title_text_plain = str(row.get(label_col, "Titre Inconnu")) 
            title_display = title_text_plain
            movie_id_current = row.get(item_id_col)

            if can_make_links and pd.notna(movie_id_current):
                # Logique pour créer le lien TMDB (inchangée)
                tmdb_id_series = df_items_global_app.loc[df_items_global_app[item_id_col] == movie_id_current, tmdb_id_col]
                if not tmdb_id_series.empty:
                    tmdb_id_val = tmdb_id_series.iloc[0]
                    if pd.notna(tmdb_id_val):
                        try: title_display = f"[{title_text_plain}](https://themoviedb.org/movie/{int(tmdb_id_val)})"
                        except ValueError: pass 
            
            # Affichage du titre et des informations de base
            st.markdown(f"##### {title_display}", unsafe_allow_html=True) # Titre un peu plus grand
            
            genres_val = str(row.get(genres_col, "N/A"))
            year_val = row.get(year_col)
            year_display = int(year_val) if pd.notna(year_val) and year_val != 0 else "N/A"
            st.caption(f"Genres: {genres_val} | Année: {year_display}")

            # Colonnes pour les notes et l'input de notation
            col_scores, col_rating_widget = st.columns([3, 2]) # Ajuster les ratios au besoin

            with col_scores:
                if 'estimated_score' in row and pd.notna(row['estimated_score']):
                    st.markdown(f"Prédiction pour vous : **{row['estimated_score']:.1f}/10**")
                
                tmdb_avg_val_display = row.get('tmdb_vote_average', row.get(vote_avg_col)) # Utilise la colonne renommée ou l'originale
                if pd.notna(tmdb_avg_val_display):
                    try:
                        st.markdown(f"Note globale TMDB : **{pd.to_numeric(tmdb_avg_val_display, errors='coerce'):.2f}/10**")
                    except: pass
            
            if enable_rating_for_user_id is not None and movie_id_current is not None:
                with col_rating_widget:
                    # st.write("Votre note:") # Peut-être redondant si le selectbox est clair
                    rating_key = f"rating_logged_in_user_{movie_id_current}_{enable_rating_for_user_id}" 
                    current_buffered_rating = st.session_state.logged_in_user_ratings_buffer.get(movie_id_current)
                    idx = rating_opts.index(current_buffered_rating) if current_buffered_rating in rating_opts else 0
                    
                    user_rating_input = st.selectbox(
                        label=f"Notez '{title_text_plain}' :", 
                        options=rating_opts,
                        index=idx,
                        format_func=fmt_fn,
                        key=rating_key,
                        label_visibility="collapsed" # ou "visible" si vous préférez un label au-dessus
                    )
                    if user_rating_input is not None: 
                        st.session_state.logged_in_user_ratings_buffer[movie_id_current] = user_rating_input
                    elif movie_id_current in st.session_state.logged_in_user_ratings_buffer: 
                        del st.session_state.logged_in_user_ratings_buffer[movie_id_current]
            
            # AJOUT DE L'AFFICHAGE DE L'EXPLICATION
            if 'explanation' in row and pd.notna(row['explanation']):
                st.markdown(f"<div style='font-size: 0.9em; color: #555; margin-top: 5px;'>💡 <i>{row['explanation']}</i></div>", unsafe_allow_html=True)
            

def display_movie_recommendations_section(recs_df, title="Recommandations", page_size=N_RECOS_PERSONNALISEES_PER_PAGE, enable_rating_for_user_id=None): # Ajout du paramètre
    st.subheader(title)
    if recs_df.empty:
        st.info("Aucun film à afficher pour cette sélection.")
        return
    
    display_df_all = recs_df.copy()
    if 'estimated_score' in display_df_all.columns:
        display_df_all['estimated_score'] = pd.to_numeric(display_df_all['estimated_score'], errors='coerce') * 2 
        display_df_all['estimated_score'] = display_df_all['estimated_score'].round(1)
    
    tmdb_avg_col_name_display = 'tmdb_vote_average' 
    if hasattr(C, 'VOTE_AVERAGE_COL') and C.VOTE_AVERAGE_COL in display_df_all.columns and tmdb_avg_col_name_display not in display_df_all.columns:
        display_df_all = display_df_all.rename(columns={C.VOTE_AVERAGE_COL: tmdb_avg_col_name_display})
    if tmdb_avg_col_name_display in display_df_all.columns:
         display_df_all[tmdb_avg_col_name_display] = pd.to_numeric(display_df_all[tmdb_avg_col_name_display], errors='coerce').round(2)
    
    # MODIFICATION ICI : passer enable_rating_for_user_id
    display_movie_cards(display_df_all.head(page_size), enable_rating_for_user_id=enable_rating_for_user_id)
    
    remaining_recs = display_df_all.iloc[page_size:]
    num_remaining_total = len(remaining_recs)
    idx = 0
    while idx < num_remaining_total:
        start_item_num = page_size + idx + 1
        end_idx_chunk = min(idx + page_size, num_remaining_total)
        chunk_to_display = remaining_recs.iloc[idx:end_idx_chunk]
        end_item_num = page_size + end_idx_chunk
        if not chunk_to_display.empty:
            with st.expander(f"Voir plus ({start_item_num} - {end_item_num} sur {len(display_df_all)})..."):
                # MODIFICATION ICI : passer enable_rating_for_user_id
                display_movie_cards(chunk_to_display, enable_rating_for_user_id=enable_rating_for_user_id)
        idx = end_idx_chunk

# --- Session State & Sidebar ---
if 'active_page' not in st.session_state: st.session_state.active_page = "general"
if 'current_user_id' not in st.session_state: st.session_state.current_user_id = None
if 'new_user_ratings' not in st.session_state: st.session_state.new_user_ratings = {}
if 'new_user_name_input' not in st.session_state: st.session_state.new_user_name_input = ''
if 'last_selected_user_id' not in st.session_state: st.session_state.last_selected_user_id = None
if 'instant_reco_model_new_user' not in st.session_state: st.session_state.instant_reco_model_new_user = None
if 'new_user_id_generated' not in st.session_state: st.session_state.new_user_id_generated = None
if 'logged_in_user_ratings_buffer' not in st.session_state:st.session_state.logged_in_user_ratings_buffer = {}

# --- Sidebar ---
st.sidebar.header("Filtres et Options")
all_genres_list_sidebar = ["Tous les genres"]
if not df_items_global_app.empty and hasattr(C, 'GENRES_COL') and C.GENRES_COL in df_items_global_app.columns:
    try:
        genres_series = df_items_global_app[C.GENRES_COL].fillna('').astype(str)
        s_genres = genres_series.str.split('|').explode()
        unique_sidebar_genres = sorted([
            g.strip() for g in s_genres.unique() if g.strip() and g.strip().lower() != '(no genres listed)'
        ])
        if unique_sidebar_genres: all_genres_list_sidebar.extend(unique_sidebar_genres)
    except Exception as e_g_sb: print(f"Erreur sidebar (liste genres): {e_g_sb}"); st.sidebar.error("Erreur chargement genres.")
selected_genre = st.sidebar.selectbox("Filtrer par genre :", all_genres_list_sidebar, key="genre_filter_sb")

slider_min, slider_max, current_slider_val = 1900, pd.Timestamp.now().year, (1900, pd.Timestamp.now().year)
if not df_items_global_app.empty and hasattr(C, 'RELEASE_YEAR_COL') and C.RELEASE_YEAR_COL in df_items_global_app.columns:
    valid_years = pd.to_numeric(df_items_global_app[C.RELEASE_YEAR_COL], errors='coerce').dropna()
    if not valid_years.empty:
        calc_min, calc_max = int(valid_years.min()), int(valid_years.max())
        if calc_min <= calc_max and calc_min > 1800: slider_min = calc_min
        if calc_max <= pd.Timestamp.now().year + 5: slider_max = calc_max
        current_slider_val = (slider_min, slider_max)
if slider_max < slider_min: slider_max = slider_min
selected_year_range = st.sidebar.slider("Filtrer par année :", min_value=slider_min, max_value=slider_max, value=current_slider_val, key="year_filter_sb")

st.sidebar.markdown("---")
st.sidebar.header("👤 Espace Utilisateur")
user_opts = ["Voir les Tops Généraux"]
can_sel_existing = not df_ratings_global_app.empty and hasattr(C, 'USER_ID_COL') and \
                   C.USER_ID_COL in df_ratings_global_app.columns and \
                   not df_ratings_global_app[C.USER_ID_COL].empty
if can_sel_existing: user_opts.append("Se connecter (ID existant)")
user_opts.append("Créer un nouveau profil")

idx_radio = 0 # Logique de sélection de l'index du radio button
if st.session_state.active_page == "general": idx_radio = user_opts.index("Voir les Tops Généraux") if "Voir les Tops Généraux" in user_opts else 0
elif st.session_state.active_page in ["new_user_profiling", "new_user_instant_recs"]: idx_radio = user_opts.index("Créer un nouveau profil") if "Créer un nouveau profil" in user_opts else 0
elif st.session_state.active_page == "user_specific" and "Se connecter (ID existant)" in user_opts: idx_radio = user_opts.index("Se connecter (ID existant)")


user_sel_opt = st.sidebar.radio("Choisissez une option :", user_opts, key="user_sel_main_radio", index=idx_radio)

orig_page, orig_uid = st.session_state.active_page, st.session_state.current_user_id
rerun_needed = False
if user_sel_opt == "Voir les Tops Généraux":
    st.session_state.active_page, st.session_state.current_user_id = "general", None
elif user_sel_opt == "Créer un nouveau profil":
    # Aller à la page de profilage si on n'est pas déjà sur la page de recos instantanées
    if st.session_state.active_page != "new_user_instant_recs":
        st.session_state.active_page = "new_user_profiling"
    st.session_state.current_user_id = "new_user_temp" 
    if orig_page not in ["new_user_profiling", "new_user_instant_recs"]:
        st.session_state.new_user_ratings = {}
        st.session_state.new_user_name_input = ''
        st.session_state.instant_reco_model_new_user = None
        st.session_state.new_user_id_generated = None

elif user_sel_opt == "Se connecter (ID existant)" and can_sel_existing:
    st.session_state.active_page = "user_specific"
    uids_list = sorted(df_ratings_global_app[C.USER_ID_COL].unique()) if not df_ratings_global_app.empty else []
    last_id_sel = st.session_state.last_selected_user_id
    if st.session_state.current_user_id is None or st.session_state.current_user_id == "new_user_temp":
        st.session_state.current_user_id = last_id_sel if last_id_sel in uids_list else (uids_list[0] if uids_list else None)

if st.session_state.active_page != orig_page or st.session_state.current_user_id != orig_uid:
    rerun_needed = True
if rerun_needed: st.rerun()

uid_for_reco = None
if st.session_state.active_page == "user_specific":
    if user_sel_opt == "Se connecter (ID existant)" and can_sel_existing:
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
            if current_selection_uid not in uids_avail:
                current_selection_uid = uids_avail[0] if uids_avail else None
                st.session_state.current_user_id = current_selection_uid
            idx_sel_box = uids_avail.index(current_selection_uid) if current_selection_uid in uids_avail else 0
            uid_sel_box_val = st.sidebar.selectbox(f"Profil ID:", options=uids_avail, format_func=disp_opts_func, index=idx_sel_box, key="uid_sel_box")
            if st.session_state.current_user_id != uid_sel_box_val:
                st.session_state.current_user_id, st.session_state.last_selected_user_id = uid_sel_box_val, uid_sel_box_val
                st.rerun()
            uid_for_reco = st.session_state.current_user_id
        else: st.sidebar.warning("Aucun utilisateur existant.")
    elif st.session_state.current_user_id not in [None, "new_user_temp"]: uid_for_reco = st.session_state.current_user_id

# --- Logique d'Affichage Principal ---
if st.session_state.active_page == "general":
    st.header("🏆 Tops des Films")
    yr_min, yr_max = selected_year_range[0], selected_year_range[1]
    if selected_genre != "Tous les genres": df_to_show = get_top_genre_movies_tmdb(genre=selected_genre, n=N_TOP_GENERAL, year_min_filter=yr_min, year_max_filter=yr_max)
    else: df_to_show = get_top_overall_movies_tmdb(n=N_TOP_GENERAL, year_min_filter=yr_min, year_max_filter=yr_max)
    display_movie_cards(df_to_show)

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
    st.header(f"Recommandations Personnalisées pour {user_display_name} (ID: {uid_for_reco})")
    yr_min_p, yr_max_p = selected_year_range[0], selected_year_range[1]
    genre_f = selected_genre if selected_genre != "Tous les genres" else None

    models_p_dir = str(C.DATA_PATH / 'recs')
    avail_model_files = [f for f in os.listdir(models_p_dir) if f.endswith('.p') and not 'personalized' in f.lower()] if os.path.exists(models_p_dir) and os.path.isdir(models_p_dir) else []
    
    if not avail_model_files: st.error(f"Aucun modèle général pré-entraîné trouvé dans {models_p_dir}.")
    else:
        
        sel_model_types = st.multiselect(
            "Types de recommandations à afficher :",
            ["Utilisateurs Similaires", "Contenu Similaire", "Modèle Factorisation (SVD)"],
            default=["Utilisateurs Similaires", "Contenu Similaire"],
            key="sel_model_types_multi"
        )
        type_map = {
            "Utilisateurs Similaires": ("user_based", "❤️ Pourrait vous plaire (Utilisateurs similaires)"),
            "Contenu Similaire": ("content_based", "👍 Basé sur vos goûts (Contenu similaire)"),
            "Modèle Factorisation (SVD)": ("svd", "✨ Recommandations Générales (SVD)")
        }

        for mt in sel_model_types:

            pfix, title_str = type_map[mt]
            m_file = next((mfile for mfile in avail_model_files if pfix in mfile.lower() and 'final' in mfile.lower()), None)
            if not m_file: m_file = next((mfile for mfile in avail_model_files if pfix in mfile.lower()), None)
            
            if m_file:
                recs_data = recommender.get_top_n_recommendations(uid_for_reco, m_file, n=N_RECOS_PERSONNALISEES_TOTAL_FETCH, filter_genre=genre_f, filter_year_range=(yr_min_p, yr_max_p))
                # MODIFICATION ICI : passer l'ID de l'utilisateur pour activer la notation
                display_movie_recommendations_section(recs_data, title=title_str, enable_rating_for_user_id=uid_for_reco)
            else: st.warning(f"Aucun modèle général de type '{mt}' trouvé.")

        # NOUVELLE SECTION : BOUTON ET LOGIQUE POUR SAUVEGARDER LES NOTES DU BUFFER
        if st.session_state.logged_in_user_ratings_buffer: # Vérifie s'il y a des notes dans le buffer
            st.markdown("---") # Séparateur visuel
            num_buffered_ratings = len(st.session_state.logged_in_user_ratings_buffer)
            
            # Espace pour centrer le bouton ou l'afficher de manière plus visible
            col1_btn, col2_btn, col3_btn = st.columns([2,3,2])
            with col2_btn: # Colonne du milieu pour le bouton
                if st.button(f"✔️ Enregistrer mes {num_buffered_ratings} nouvelle(s) note(s)", key="save_logged_in_ratings", help="Vos notes seront ajoutées pour améliorer les futures recommandations après la mise à jour du système."):
                    ratings_to_save_list = []
                    current_ts = int(time.time())
                    
                    # Assurez-vous que uid_for_reco est bien l'ID de l'utilisateur actuel
                    user_id_to_save = uid_for_reco 

                    for movie_id_key, rating_val_key in st.session_state.logged_in_user_ratings_buffer.items():
                        ratings_to_save_list.append({
                            C.USER_ID_COL: user_id_to_save,
                            C.ITEM_ID_COL: movie_id_key,
                            C.RATING_COL: rating_val_key,
                            C.TIMESTAMP_COL: current_ts # Assurez-vous que C.TIMESTAMP_COL est défini dans constants.py
                        })
                    
                    if ratings_to_save_list:
                        df_new_ratings_to_save_out = pd.DataFrame(ratings_to_save_list)
                        pending_ratings_filepath = C.EVIDENCE_PATH / getattr(C, 'NEW_RATINGS_PENDING_FILENAME', 'new_ratings_pending.csv')
                        file_exists_pending = os.path.exists(pending_ratings_filepath)
                        
                        try:
                            df_new_ratings_to_save_out.to_csv(pending_ratings_filepath, mode='a', header=not file_exists_pending, index=False)
                            st.success(f"{len(ratings_to_save_list)} note(s) enregistrée(s) avec succès ! Elles seront prises en compte lors de la prochaine mise à jour des modèles.")
                            st.session_state.logged_in_user_ratings_buffer = {} # Vider le buffer après sauvegarde
                            st.rerun() # Optionnel : rafraîchir l'interface pour refléter que le buffer est vide
                        except Exception as e_save_rating:
                            st.error(f"Une erreur est survenue lors de la sauvegarde de vos notes : {e_save_rating}")
                            print(f"ERREUR App.py (sauvegarde notes utilisateur existant): {e_save_rating}")
                    else:
                        st.info("Aucune note à enregistrer pour le moment.") # Devrait être couvert par le if st.session_state.logged_in_user_ratings_buffer:
elif st.session_state.active_page == "new_user_profiling":
    st.header("👤 Créez votre profil de goûts")
    st.write("Pour nous aider à comprendre vos préférences, veuillez indiquer votre nom et noter quelques films.")
    new_user_name = st.text_input("Quel est votre nom ?", st.session_state.get('new_user_name_input', ''))
    st.session_state.new_user_name_input = new_user_name
    movies_for_profiling_pool = df_items_global_app.copy() if not df_items_global_app.empty else pd.DataFrame()
    sample_size = 20; min_prefs_needed = 5
    movies_to_display_df = pd.DataFrame()
    if not movies_for_profiling_pool.empty: # Sélection des films pour le questionnaire
        if hasattr(C, 'POPULARITY_COL') and C.POPULARITY_COL in movies_for_profiling_pool.columns and not movies_for_profiling_pool[C.POPULARITY_COL].isnull().all():
            movies_for_profiling_pool[C.POPULARITY_COL] = pd.to_numeric(movies_for_profiling_pool[C.POPULARITY_COL], errors='coerce').fillna(0)
            movies_to_display_df_temp = movies_for_profiling_pool.sort_values(by=C.POPULARITY_COL, ascending=False).head(150)
            if len(movies_to_display_df_temp) >= sample_size: movies_to_display_df = movies_to_display_df_temp.sample(n=sample_size, random_state=42)
        if movies_to_display_df.empty: 
            if len(movies_for_profiling_pool) >= sample_size: movies_to_display_df = movies_for_profiling_pool.sample(n=sample_size, random_state=42)
            else: movies_to_display_df = movies_for_profiling_pool.copy()
    
    if movies_to_display_df.empty: st.error("Impossible de charger des films pour le profilage.")
    else:
        with st.form(key="new_user_profiling_form"): # Formulaire de notation
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

        if form_submitted: # Après soumission du formulaire
            if not new_user_name.strip(): st.warning("Veuillez entrer votre nom.")
            elif num_total_prefs < min_prefs_needed: st.warning(f"Veuillez noter au moins {min_prefs_needed} films.")
            else: # Assez de notes et nom fourni
                try:
                    # 1. Sauvegarde pour traitement offline (comme avant)
                    current_ratings_df_for_id = df_ratings_global_app 
                    if current_ratings_df_for_id.empty:
                         current_ratings_df_for_id = load_ratings()
                         if current_ratings_df_for_id.empty: raise Exception("Ratings non chargeables pour ID.")
                    
                    new_user_id_val = current_ratings_df_for_id[C.USER_ID_COL].max() + 1 if not current_ratings_df_for_id.empty else 1
                    st.session_state.new_user_id_generated = new_user_id_val # Sauvegarder l'ID généré
                    
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
                        st.success(f"Profil pour {new_user_name.strip()} (ID: {new_user_id_val}) sauvegardé pour traitement hors ligne.")
                    
                    # 2. Entraînement du modèle ContentBased à la volée pour suggestions immédiates
                    st.info("Calcul de vos premières suggestions...")
                    cb_features_instant = ["Genre_binary", "Year_of_release"] # Choisir des features simples et rapides
                    # Vérifier la disponibilité des features pour le modèle CB instantané
                    actual_cb_features_instant = []
                    if hasattr(C, 'GENRES_COL') and C.GENRES_COL in models_df_items_global.columns : actual_cb_features_instant.append("Genre_binary")
                    if hasattr(C, 'RELEASE_YEAR_COL') and C.RELEASE_YEAR_COL in models_df_items_global.columns: actual_cb_features_instant.append("Year_of_release")
                    
                    if not actual_cb_features_instant:
                        st.warning("Pas assez de features disponibles pour les suggestions instantanées. Des recommandations plus complètes seront disponibles après la mise à jour du système.")
                        st.session_state.active_page = "general" # Rediriger
                        st.rerun()

                    else:
                        cb_model_instant = ContentBased(features_methods=actual_cb_features_instant, regressor_method='linear')
                        reader = Reader(rating_scale=C.RATINGS_SCALE) # Utiliser l'échelle définie
                        
                        # Créer un DataFrame pour le trainset Surprise avec le nouvel utilisateur temporaire
                        # L'ID utilisateur pour ce trainset instantané peut être un ID temporaire comme -1
                        # ou l'ID réel généré si ContentBased peut le gérer directement.
                        # Pour la simplicité de l'entraînement CB, on utilise un ID temporaire unique pour ce trainset.
                        instant_user_id_for_train = -1 
                        ratings_for_instant_model_df = pd.DataFrame([{
                            C.USER_ID_COL: instant_user_id_for_train, 
                            C.ITEM_ID_COL: mid, 
                            C.RATING_COL: rval
                        } for mid, rval in final_prefs.items()])

                        data_instant = Dataset.load_from_df(ratings_for_instant_model_df, reader)
                        trainset_instant = data_instant.build_full_trainset()
                        
                        cb_model_instant.fit(trainset_instant)
                        st.session_state.instant_reco_model_new_user = cb_model_instant
                        st.session_state.active_page = "new_user_instant_recs"
                        st.rerun()

                except Exception as e_profile_processing:
                    st.error(f"Erreur lors de la création de votre profil : {e_profile_processing}")
                    print(f"ERREUR App.py (new user processing): {e_profile_processing}")

elif st.session_state.active_page == "new_user_instant_recs":
    st.header("🎉 Vos Premières Suggestions de Films !")
    st.caption("Ces suggestions sont basées sur les quelques notes que vous venez de donner. Des recommandations plus personnalisées et issues de modèles plus complexes seront disponibles après la prochaine mise à jour de notre système.")
    
    model_instance = st.session_state.get('instant_reco_model_new_user')
    new_user_ratings_keys = st.session_state.get('new_user_ratings', {}).keys() # Films déjà notés par l'utilisateur
    generated_user_id_for_pred = -1 # Utiliser le même ID temporaire que pour l'entraînement du modèle instantané

    if model_instance and models_df_items_global is not None and not models_df_items_global.empty:
        # Obtenir tous les films, exclure ceux déjà notés dans le questionnaire
        all_movie_ids_global = models_df_items_global[C.ITEM_ID_COL].unique()
        movies_to_predict_ids = [mid for mid in all_movie_ids_global if mid not in new_user_ratings_keys]
        
        if not movies_to_predict_ids:
            st.info("Il semble que vous ayez noté tous les films de notre sélection initiale ou qu'il n'y ait pas d'autres films à suggérer pour le moment.")
        else:
            preds_list_instant = []
            for item_id_to_predict in random.sample(movies_to_predict_ids, min(len(movies_to_predict_ids), 200)): # Prédire sur un échantillon pour la vitesse
                try:
                    prediction = model_instance.predict(uid=generated_user_id_for_pred, iid=item_id_to_predict)
                    preds_list_instant.append({C.ITEM_ID_COL: prediction.iid, 'estimated_score': prediction.est})
                except Exception as e_pred_inst:
                    # print(f"Erreur prédiction instantanée pour {item_id_to_predict}: {e_pred_inst}")
                    continue
            
            if preds_list_instant:
                recs_instant_df = pd.DataFrame(preds_list_instant).sort_values(by='estimated_score', ascending=False).head(N_INSTANT_RECOS_NEW_USER)
                
                # Enrichir avec les détails des films
                if not recs_instant_df.empty and not df_items_global_app.empty:
                    final_recs_instant_df = pd.merge(recs_instant_df, 
                                                     df_items_global_app[[C.ITEM_ID_COL, C.LABEL_COL, C.GENRES_COL, C.RELEASE_YEAR_COL, C.VOTE_AVERAGE_COL, C.TMDB_ID_COL]], 
                                                     on=C.ITEM_ID_COL, 
                                                     how='left')
                    display_movie_recommendations_section(final_recs_instant_df, title="Suggestions rapides pour vous :")
                elif not recs_instant_df.empty: # Si df_items_global_app est vide mais on a des recos
                     st.write(recs_instant_df) # Afficher au moins les IDs et scores
                else:
                    st.info("Impossible de générer des suggestions instantanées pour le moment.")
            else:
                st.info("Aucune suggestion instantanée n'a pu être générée avec les notes fournies.")
    else:
        st.warning("Le modèle de suggestion instantanée n'est pas disponible. Veuillez réessayer de créer votre profil.")

    if st.button("Explorer d'autres films (Tops Généraux)"):
        st.session_state.active_page = "general"
        # Nettoyer les états de session spécifiques au nouveau profil si on quitte cette section
        st.session_state.new_user_ratings = {}
        st.session_state.new_user_name_input = ''
        st.session_state.instant_reco_model_new_user = None
        st.session_state.new_user_id_generated = None
        st.rerun()

else:
    if st.session_state.active_page not in ["general", "user_specific", "new_user_profiling", "new_user_instant_recs"]:
        pass # Évite les messages de debug pour les états non explicitement gérés

st.sidebar.markdown("---")
st.sidebar.info("Projet Recommender Systems MLSMM2156")