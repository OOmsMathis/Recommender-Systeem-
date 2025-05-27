# app.py
import streamlit as st
import pandas as pd
import os
import re
import random # Pour échantillonner les films de genre favori

import constants as C_module
C = C_module.Constant()
import content
import recommender

# --- Constantes pour le nombre de recommandations ---
N_TOP_GENERAL = 30
N_RECOS_PERSONNALISEES_INITIAL_DISPLAY = 10
N_RECOS_PERSONNALISEES_PER_PAGE = 10
N_RECOS_PERSONNALISEES_TOTAL_FETCH = 30

# --- Chargement des données ---
try:
    from models import df_ratings_global, df_items_global, ContentBased
    if df_items_global.empty:
        print("app.py: AVERTISSEMENT - df_items_global (de models.py) est vide.")
        from loaders import load_items
        df_items_global_direct = load_items()
        if not df_items_global_direct.empty: df_items_global = df_items_global_direct
        elif df_items_global.empty: print("app.py: ERREUR - df_items_global reste vide.")
    if df_ratings_global.empty:
        print("app.py: AVERTISSEMENT - df_ratings_global (de models.py) est vide.")
        from loaders import load_ratings
        df_ratings_global_direct = load_ratings()
        if not df_ratings_global_direct.empty: df_ratings_global = df_ratings_global_direct
        elif df_ratings_global.empty: print("app.py: ERREUR - df_ratings_global reste vide.")
except ImportError:
    print("app.py: ERREUR CRITIQUE - Import depuis models.py échoué.")
    try:
        from loaders import load_items, load_ratings
        df_items_global = load_items(); df_ratings_global = load_ratings()
        if df_items_global.empty or df_ratings_global.empty: raise Exception("Chargement direct via loaders a échoué.")
    except Exception as e_load:
        print(f"app.py: ERREUR FATALE lors du chargement direct des données: {e_load}")
        _cols_items = [getattr(C, col_attr, col_attr.lower()) for col_attr in ['ITEM_ID_COL', 'LABEL_COL', 'GENRES_COL', 'RELEASE_YEAR_COL', 'TMDB_ID_COL', 'VOTE_AVERAGE_COL', 'VOTE_COUNT_COL'] if hasattr(C, col_attr)]
        df_items_global = pd.DataFrame(columns=_cols_items)
        _cols_ratings = [getattr(C, col_attr, col_attr.lower()) for col_attr in ['USER_ID_COL', 'ITEM_ID_COL', 'RATING_COL'] if hasattr(C, col_attr)]
        df_ratings_global = pd.DataFrame(columns=_cols_ratings)
        st.error("Erreur critique lors du chargement des données initiales.")

st.set_page_config(page_title="TMDB Recommender Assistance", layout="wide")
st.title("🎬 TMDB Recommender System Assistance")

# --- Fonctions de récupération de données (inchangées) ---
@st.cache_data
def get_top_overall_movies_tmdb(n=N_TOP_GENERAL, year_min_filter=None, year_max_filter=None):
    if df_items_global.empty or not hasattr(C, 'VOTE_AVERAGE_COL') or C.VOTE_AVERAGE_COL not in df_items_global.columns:
        return pd.DataFrame()
    items_to_consider = df_items_global.copy()
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
    if hasattr(C, 'VOTE_AVERAGE_COL') and C.VOTE_AVERAGE_COL in top_movies_df.columns: cols_out.append(C.VOTE_AVERAGE_COL) # S'assurer que la constante existe
    if vote_count_col_to_use == C.VOTE_COUNT_COL and hasattr(C, 'VOTE_COUNT_COL') and C.VOTE_COUNT_COL in top_movies_df.columns:
        cols_out.append(C.VOTE_COUNT_COL)
    return top_movies_df[[col for col in cols_out if col in top_movies_df.columns]]


@st.cache_data
def get_top_genre_movies_tmdb(genre, n=N_TOP_GENERAL, year_min_filter=None, year_max_filter=None):
    if df_items_global.empty or not hasattr(C, 'GENRES_COL') or C.GENRES_COL not in df_items_global.columns or not hasattr(C, 'VOTE_AVERAGE_COL') or C.VOTE_AVERAGE_COL not in df_items_global.columns:
        return pd.DataFrame()
    genre_movies_df = df_items_global[df_items_global[C.GENRES_COL].str.contains(re.escape(genre), case=False, na=False, regex=True)]
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

# --- Fonctions d'affichage ---
def display_movie_cards(df_to_display):
    if df_to_display.empty: return
    item_id_col, label_col, genres_col, year_col, tmdb_id_col, vote_avg_col = C.ITEM_ID_COL, C.LABEL_COL, C.GENRES_COL, C.RELEASE_YEAR_COL, C.TMDB_ID_COL, C.VOTE_AVERAGE_COL
    can_make_links = (hasattr(C, 'TMDB_ID_COL') and tmdb_id_col in df_items_global.columns and item_id_col in df_items_global.columns)
    
    for _, row in df_to_display.iterrows():
        with st.container():
            title_display = str(row.get(label_col, "Titre Inconnu"))
            movie_id_current = row.get(item_id_col)

            if can_make_links and pd.notna(movie_id_current) and item_id_col in df_items_global and movie_id_current in df_items_global[item_id_col].values:
                tmdb_id_val = df_items_global.loc[df_items_global[item_id_col] == movie_id_current, tmdb_id_col].iloc[0]
                if pd.notna(tmdb_id_val):
                    try: title_display = f"[{title_display}](https://themoviedb.org/movie/{int(tmdb_id_val)})"
                    except: pass 
            
            col_info, col_pred_score, col_global_score = st.columns([6, 2, 2])
            with col_info:
                st.markdown(f"**{title_display}**", unsafe_allow_html=True)
                genres_val = str(row.get(genres_col, "N/A"))
                year_val = row.get(year_col)
                year_display = int(year_val) if pd.notna(year_val) and year_val != 0 else "N/A"
                st.caption(f"{genres_val} ({year_display})")
            with col_pred_score:
                if 'estimated_score' in row and pd.notna(row['estimated_score']):
                    st.markdown(f"<div style='font-size: small; text-align: center;'>Prédictions:<br><b>{row['estimated_score']:.1f}/10</b></div>", unsafe_allow_html=True)
            with col_global_score:
                tmdb_avg_val_display = row.get('tmdb_vote_average', row.get(vote_avg_col))
                if pd.notna(tmdb_avg_val_display):
                    st.markdown(f"<div style='font-size: small; text-align: center;'>Note Globale:<br><b>{pd.to_numeric(tmdb_avg_val_display, errors='coerce'):.2f}/10</b></div>", unsafe_allow_html=True)

def display_movie_recommendations_section(recs_df, title="Recommandations", initial_display_count=N_RECOS_PERSONNALISEES_INITIAL_DISPLAY, page_size=N_RECOS_PERSONNALISEES_PER_PAGE):
    st.subheader(title)
    if recs_df.empty:
        st.info("Aucun film à afficher pour cette sélection.")
        return
    
    display_df_all = recs_df.copy()
    if 'estimated_score' in display_df_all.columns:
        display_df_all['estimated_score'] = pd.to_numeric(display_df_all['estimated_score'], errors='coerce') * 2
        display_df_all['estimated_score'] = display_df_all['estimated_score'].round(1)
        
        if st.session_state.get('current_user_id') == "temp_newly_profiled" and title.startswith("Voici quelques films basés sur vos notes"):
            unique_pred_scores = display_df_all['estimated_score'].dropna().unique()
            if len(unique_pred_scores) == 1:
                st.caption(f"ℹ️ Les prédictions pour votre nouveau profil peuvent être uniformes ({unique_pred_scores[0]:.1f}/10). Cela est souvent dû au faible nombre de notes initiales et de préférences. Le modèle apprendra mieux avec plus d'interactions !")

    tmdb_avg_col_name_display = 'tmdb_vote_average' 
    if hasattr(C, 'VOTE_AVERAGE_COL') and C.VOTE_AVERAGE_COL in display_df_all.columns and tmdb_avg_col_name_display not in display_df_all.columns:
        display_df_all = display_df_all.rename(columns={C.VOTE_AVERAGE_COL: tmdb_avg_col_name_display})
    if tmdb_avg_col_name_display in display_df_all.columns:
         display_df_all[tmdb_avg_col_name_display] = pd.to_numeric(display_df_all[tmdb_avg_col_name_display], errors='coerce').round(2)
    
    display_movie_cards(display_df_all.head(page_size))
    remaining_recs = display_df_all.iloc[page_size:]
    num_remaining_total = len(remaining_recs)
    idx = 0; expander_idx = 0
    while idx < num_remaining_total:
        start_item_num = page_size + idx + 1
        end_idx_chunk = min(idx + page_size, num_remaining_total)
        chunk_to_display = remaining_recs.iloc[idx:end_idx_chunk]
        end_item_num = page_size + end_idx_chunk
        if not chunk_to_display.empty:
            with st.expander(f"Voir plus ({start_item_num} - {end_item_num} sur {len(display_df_all)})..."):
                display_movie_cards(chunk_to_display)
        idx = end_idx_chunk; expander_idx += 1

# --- Session State & Sidebar ---
if 'active_page' not in st.session_state: st.session_state.active_page = "general"
if 'current_user_id' not in st.session_state: st.session_state.current_user_id = None
if 'new_user_ratings' not in st.session_state: st.session_state.new_user_ratings = {}
if 'new_user_fav_movie_titles' not in st.session_state: st.session_state.new_user_fav_movie_titles = []
if 'new_user_disliked_movie_titles' not in st.session_state: st.session_state.new_user_disliked_movie_titles = []
if 'new_user_fav_genres' not in st.session_state: st.session_state.new_user_fav_genres = [] # Pour les genres favoris

if hasattr(C, 'MAXIME_USER_ID'): st.session_state.MAXIME_USER_ID = C.MAXIME_USER_ID
else: st.session_state.MAXIME_USER_ID = 0 
if 'last_selected_user_id' not in st.session_state: st.session_state.last_selected_user_id = None

# --- Sidebar ---
st.sidebar.header("Filtres et Options")
# (Filtres genre et année comme avant)
all_genres_list_sidebar = ["Tous les genres"]
if not df_items_global.empty and hasattr(C, 'GENRES_COL') and C.GENRES_COL in df_items_global.columns:
    try:
        s_genres = df_items_global[C.GENRES_COL].dropna().astype(str).str.split('|').explode()
        unique_sidebar_genres = sorted([g.strip() for g in s_genres.unique() if g.strip() and g.strip().lower() != '(no genres listed)'])
        if unique_sidebar_genres: all_genres_list_sidebar.extend(unique_sidebar_genres)
    except Exception as e_g_sb: print(f"Erreur sidebar (liste genres): {e_g_sb}")
selected_genre = st.sidebar.selectbox("Filtrer par genre :", all_genres_list_sidebar, key="genre_filter_sb")

# (Slider année comme avant)
slider_min, slider_max, current_slider_val = 1900, pd.Timestamp.now().year, (1900, pd.Timestamp.now().year)
if not df_items_global.empty and hasattr(C, 'RELEASE_YEAR_COL') and C.RELEASE_YEAR_COL in df_items_global.columns:
    valid_years = pd.to_numeric(df_items_global[C.RELEASE_YEAR_COL], errors='coerce').dropna()
    if not valid_years.empty:
        calc_min, calc_max = int(valid_years.min()), int(valid_years.max())
        if calc_min <= calc_max: slider_min, slider_max = calc_min, calc_max
        current_slider_val = (slider_min, slider_max)
if slider_max < slider_min: slider_max = slider_min # Correction
selected_year_range = st.sidebar.slider("Filtrer par année :", min_value=slider_min, max_value=slider_max, value=current_slider_val, key="year_filter_sb")


st.sidebar.markdown("---")
st.sidebar.header("👤 Espace Utilisateur")
# (Logique de user_options, current_radio_index, user_selection_option comme dans la dernière version fonctionnelle)
user_opts = ["Voir les Tops Généraux"]
known_users = getattr(C, 'KNOWN_USERS_MAP', {st.session_state.MAXIME_USER_ID: "Mon Profil (Maxime)"})
can_sel_existing = not df_ratings_global.empty and hasattr(C, 'USER_ID_COL') and C.USER_ID_COL in df_ratings_global.columns and not df_ratings_global[C.USER_ID_COL].empty

if can_sel_existing: user_opts.append("Se connecter (ID existant)")
if hasattr(st.session_state, 'MAXIME_USER_ID'):
    max_is_sel = st.session_state.MAXIME_USER_ID in known_users or \
                   (can_sel_existing and st.session_state.MAXIME_USER_ID in df_ratings_global[C.USER_ID_COL].unique())
    if max_is_sel and "Mon Profil (Maxime)" not in user_opts:
        idx = user_opts.index("Se connecter (ID existant)") + 1 if "Se connecter (ID existant)" in user_opts else len(user_opts)
        user_opts.insert(idx, "Mon Profil (Maxime)")
user_opts.append("Créer un nouveau profil")

idx_radio = 0
try:
    if st.session_state.active_page == "general": idx_radio = user_opts.index("Voir les Tops Généraux")
    elif st.session_state.active_page in ["new_user_profiling", "new_user_recs_ready"]: idx_radio = user_opts.index("Créer un nouveau profil")
    elif st.session_state.active_page == "user_specific":
        if st.session_state.current_user_id == st.session_state.MAXIME_USER_ID and "Mon Profil (Maxime)" in user_opts: idx_radio = user_opts.index("Mon Profil (Maxime)")
        elif st.session_state.current_user_id not in [None, "temp_newly_profiled", "new_user_temp"] and "Se connecter (ID existant)" in user_opts: idx_radio = user_opts.index("Se connecter (ID existant)")
except ValueError: idx_radio = 0

user_sel_opt = st.sidebar.radio("Choisissez une option :", user_opts, key="user_sel_main_radio", index=idx_radio)

orig_page, orig_uid = st.session_state.active_page, st.session_state.current_user_id
rerun_sb = False
if not (orig_page == "new_user_recs_ready" and user_sel_opt == "Créer un nouveau profil" and orig_uid == "temp_newly_profiled" ):
    if user_sel_opt == "Voir les Tops Généraux": st.session_state.active_page, st.session_state.current_user_id = "general", None
    elif user_sel_opt == "Mon Profil (Maxime)" and "Mon Profil (Maxime)" in user_opts: st.session_state.active_page, st.session_state.current_user_id = "user_specific", st.session_state.MAXIME_USER_ID
    elif user_sel_opt == "Créer un nouveau profil":
        st.session_state.active_page, st.session_state.current_user_id = "new_user_profiling", "new_user_temp"
        if orig_page != "new_user_profiling": # Reset only if changing to this page
            st.session_state.new_user_ratings, st.session_state.new_user_fav_movie_titles, st.session_state.new_user_disliked_movie_titles, st.session_state.new_user_fav_genres = {}, [], [], []
            if 'new_user_model_instance' in st.session_state: del st.session_state.new_user_model_instance
    elif user_sel_opt == "Se connecter (ID existant)" and can_sel_existing:
        st.session_state.active_page = "user_specific"
        uids_list = sorted(df_ratings_global[C.USER_ID_COL].unique()) if not df_ratings_global.empty else []
        last_id_sel = st.session_state.last_selected_user_id
        st.session_state.current_user_id = last_id_sel if last_id_sel in uids_list else (uids_list[0] if uids_list else None)
    if st.session_state.active_page != orig_page or st.session_state.current_user_id != orig_uid: rerun_sb = True
if rerun_sb: st.rerun()

uid_for_reco = None
if st.session_state.active_page == "user_specific":
    if user_sel_opt == "Se connecter (ID existant)" and can_sel_existing:
        uids_avail = sorted(df_ratings_global[C.USER_ID_COL].unique()) if not df_ratings_global.empty else []
        if uids_avail:
            disp_opts = {uid: f"{known_users.get(uid, '')} (ID: {uid})" if uid in known_users else uid for uid in uids_avail}
            idx_sel_box = uids_avail.index(st.session_state.current_user_id) if st.session_state.current_user_id in uids_avail else 0
            if st.session_state.current_user_id is None and uids_avail: st.session_state.current_user_id = uids_avail[0]; idx_sel_box = 0
            
            uid_sel_box_val = st.sidebar.selectbox(f"Profil ID:", options=uids_avail, format_func=lambda x: disp_opts.get(x,x), index=idx_sel_box, key="uid_sel_box")
            if st.session_state.current_user_id != uid_sel_box_val:
                st.session_state.current_user_id, st.session_state.last_selected_user_id = uid_sel_box_val, uid_sel_box_val
                st.rerun()
            uid_for_reco = st.session_state.current_user_id
        else: st.sidebar.warning("Aucun utilisateur existant.")
    elif st.session_state.current_user_id not in [None, "new_user_temp", "temp_newly_profiled"]: uid_for_reco = st.session_state.current_user_id


# --- Logique d'Affichage Principal ---
if st.session_state.active_page == "general":
    st.header("🏆 Tops des Films"); yr_min, yr_max = selected_year_range[0], selected_year_range[1]
    df_to_show = get_top_overall_movies_tmdb(n=N_TOP_GENERAL, year_min_filter=yr_min, year_max_filter=yr_max) if selected_genre == "Tous les genres" \
        else get_top_genre_movies_tmdb(genre=selected_genre, n=N_TOP_GENERAL, year_min_filter=yr_min, year_max_filter=yr_max)
    display_movie_cards(df_to_show)

elif st.session_state.active_page == "user_specific" and uid_for_reco is not None:
    # (Section pour utilisateurs existants/Maxime, comme avant, en utilisant uid_for_reco)
    user_disp_name = known_users.get(uid_for_reco, f"Utilisateur {uid_for_reco}")
    st.header(f"Recommandations Personnalisées pour {user_disp_name}"); yr_min_p, yr_max_p = selected_year_range[0], selected_year_range[1]
    genre_f = selected_genre if selected_genre != "Tous les genres" else None
    models_p_dir = str(C.DATA_PATH / 'recs')
    avail_model_files = [f for f in os.listdir(models_p_dir) if f.endswith('.p')] if os.path.exists(models_p_dir) and os.path.isdir(models_p_dir) else []
    if not avail_model_files: st.error(f"Aucun modèle pré-entraîné trouvé dans {models_p_dir}.")
    else:
        sel_model_types = st.multiselect("Types de recommandations :", ["Utilisateurs Similaires", "Contenu Similaire", "Modèle Général (SVD)"], default=["Utilisateurs Similaires", "Contenu Similaire"], key="sel_model_types_multi")
        
        type_map = {
            "Utilisateurs Similaires": ("user_based", "❤️ Pourrait vous plaire (Utilisateurs similaires)"),
            "Contenu Similaire": ("content_based", "👍 Basé sur vos goûts (Contenu similaire)"),
            "Modèle Général (SVD)": ("svd", "✨ Recommandations Générales (SVD)")
        }
        for mt in sel_model_types:
            pfix, title_str = type_map[mt]
            # Pour SVD général, s'assurer qu'il n'est pas personnalisé pour Maxime
            m_file = next((m for m in avail_model_files if pfix in m.lower() and not ('personalized' in m.lower() or 'maxime' in m.lower() or (hasattr(st.session_state, 'MAXIME_USER_ID') and str(st.session_state.MAXIME_USER_ID) in m))),None) if mt=="Modèle Général (SVD)" \
                else next((m for m in avail_model_files if pfix in m.lower()), None)

            if m_file:
                recs_data = recommender.get_top_n_recommendations(uid_for_reco, m_file, n=N_RECOS_PERSONNALISEES_TOTAL_FETCH, filter_genre=genre_f, filter_year_range=(yr_min_p, yr_max_p))
                display_movie_recommendations_section(recs_data, title=title_str)
            else: st.warning(f"Aucun modèle '{mt}' trouvé.")

        if hasattr(st.session_state, 'MAXIME_USER_ID') and uid_for_reco == st.session_state.MAXIME_USER_ID:
            max_model_f = next((m for m in avail_model_files if 'personalized' in m.lower() or 'maxime' in m.lower() or str(st.session_state.MAXIME_USER_ID) in m), None)
            if max_model_f:
                st.markdown("---"); recs_m = recommender.get_top_n_recommendations(uid_for_reco, max_model_f, n=N_RECOS_PERSONNALISEES_TOTAL_FETCH, filter_genre=genre_f, filter_year_range=(yr_min_p, yr_max_p))
                display_movie_recommendations_section(recs_m, title=f"🌟 Spécialement pour {known_users.get(st.session_state.MAXIME_USER_ID, 'Maxime')} (Modèle Personnalisé)")


elif st.session_state.active_page == "new_user_profiling":
    st.header("👤 Créez votre profil de goûts")
    st.write("Pour nous aider à comprendre vos préférences, veuillez noter quelques films et indiquer vos favoris/détestés ainsi que vos genres préférés.")
    
    # Préparation des films pour le profilage
    movies_for_profiling_pool = df_items_global.copy() if not df_items_global.empty else pd.DataFrame()
    sample_size = 20; min_prefs_needed = 5
    movies_to_display_df = pd.DataFrame() # Init

    if not movies_for_profiling_pool.empty:
        if hasattr(C, 'POPULARITY_COL') and C.POPULARITY_COL in movies_for_profiling_pool.columns and not movies_for_profiling_pool[C.POPULARITY_COL].isnull().all():
            movies_for_profiling_pool[C.POPULARITY_COL] = pd.to_numeric(movies_for_profiling_pool[C.POPULARITY_COL], errors='coerce').fillna(0)
            movies_to_display_df_temp = movies_for_profiling_pool.sort_values(by=C.POPULARITY_COL, ascending=False).head(150)
            if not movies_to_display_df_temp.empty:
                 movies_to_display_df = movies_to_display_df_temp.sample(n=min(sample_size, len(movies_to_display_df_temp)), random_state=42)
        if movies_to_display_df.empty: # Fallback si la popularité n'a pas aidé ou n'existe pas
            movies_to_display_df = movies_for_profiling_pool.sample(n=min(sample_size, len(movies_for_profiling_pool)), random_state=42)
    
    # Liste des genres pour la sélection par l'utilisateur
    profiling_genre_options = []
    if not df_items_global.empty and hasattr(C, 'GENRES_COL') and C.GENRES_COL in df_items_global.columns:
        try:
            s_genres_prof = df_items_global[C.GENRES_COL].dropna().astype(str).str.split('|').explode()
            profiling_genre_options = sorted([g.strip() for g in s_genres_prof.unique() if g.strip() and g.strip().lower() != '(no genres listed)'])
        except Exception as e_prof_g: print(f"Erreur profilage (liste genres): {e_prof_g}")


    if movies_to_display_df.empty: st.error("Impossible de charger des films pour le profilage.")
    else:
        with st.form(key="new_user_profiling_form"):
            st.subheader("1. Vos genres de films préférés (jusqu'à 3)")
            selected_fav_genres = st.multiselect(
                "Quels sont vos genres de prédilection ?",
                options=profiling_genre_options,
                default=st.session_state.get('new_user_fav_genres', []),
                max_selections=3,
                key="fav_genres_multiselect"
            )
            st.session_state.new_user_fav_genres = selected_fav_genres # Mettre à jour en session pour persistance si form non soumis

            st.subheader("2. Vos films préférés (jusqu'à 5)")
            movie_titles_options = movies_to_display_df[C.LABEL_COL].tolist()
            fav_titles = st.multiselect("Sélectionnez vos films préférés dans cette liste :", options=movie_titles_options, default=st.session_state.new_user_fav_movie_titles, max_selections=5, key="fav_titles_select_profiling")
            st.session_state.new_user_fav_movie_titles = fav_titles


            st.subheader("3. Films que vous n'aimez pas (jusqu'à 2, optionnel)")
            dis_options = [t for t in movie_titles_options if t not in fav_titles]
            disliked_titles = st.multiselect("Sélectionnez des films que vous n'aimez pas :", options=dis_options, default=st.session_state.new_user_disliked_movie_titles, max_selections=2, key="disliked_titles_select_profiling")
            st.session_state.new_user_disliked_movie_titles = disliked_titles


            st.subheader("4. Notez les films suivants")
            st.caption("Laissez sur 'Pas de note' si vous ne connaissez pas ou n'avez pas d'avis.")
            for _, row in movies_to_display_df.iterrows():
                movie_id, title = row[C.ITEM_ID_COL], row[C.LABEL_COL]
                if title in fav_titles or title in disliked_titles: continue 

                rating_key = f"rating_select_{movie_id}"
                current_rating = st.session_state.new_user_ratings.get(movie_id)
                col1, col2 = st.columns([3,2])
                with col1: st.write(f"**{title}**"); 
                if hasattr(C, 'GENRES_COL') and C.GENRES_COL in row and pd.notna(row[C.GENRES_COL]): st.caption(f"Genres: {row[C.GENRES_COL]}")
                with col2:
                    opts = [None, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
                    fmt_fn = lambda x: "Pas de note" if x is None else f"{x} ★"
                    idx = opts.index(current_rating) if current_rating in opts else 0
                    new_r = st.selectbox(f"Note:", opts, index=idx, format_func=fmt_fn, key=rating_key, label_visibility="collapsed") # label enlevé car titre est à gauche
                    if new_r is not None: st.session_state.new_user_ratings[movie_id] = new_r
                    elif movie_id in st.session_state.new_user_ratings: del st.session_state.new_user_ratings[movie_id]
            
            form_submitted = st.form_submit_button("✔️ Terminer et Obtenir mes Recommandations")

        # Logique après la soumission du formulaire
        final_prefs_for_model = st.session_state.new_user_ratings.copy()
        fav_ids = movies_to_display_df[movies_to_display_df[C.LABEL_COL].isin(st.session_state.new_user_fav_movie_titles)][C.ITEM_ID_COL].tolist()
        dis_ids = movies_to_display_df[movies_to_display_df[C.LABEL_COL].isin(st.session_state.new_user_disliked_movie_titles)][C.ITEM_ID_COL].tolist()
        
        for mid in dis_ids: final_prefs_for_model[mid] = 0.5 
        for mid in fav_ids: final_prefs_for_model[mid] = 5.0
        
        # Ajout de films basés sur les genres favoris
        num_genre_boost_movies = 3 # Nombre de films à ajouter par genre favori (ou au total)
        count_added_genre_movies = 0
        if st.session_state.new_user_fav_genres and not df_items_global.empty:
            # print(f"DEBUG: Genres favoris sélectionnés: {st.session_state.new_user_fav_genres}")
            for fav_genre in st.session_state.new_user_fav_genres:
                if count_added_genre_movies >= num_genre_boost_movies: break
                # Trouver des films populaires/bien notés de ce genre, non déjà dans final_prefs_for_model
                genre_specific_movies = df_items_global[
                    df_items_global[C.GENRES_COL].str.contains(re.escape(fav_genre), case=False, na=False) &
                    (~df_items_global[C.ITEM_ID_COL].isin(final_prefs_for_model.keys()))
                ]
                if not genre_specific_movies.empty:
                    # Prioriser par popularité ou note moyenne si disponible
                    movies_to_sample_from = genre_specific_movies
                    if hasattr(C, 'VOTE_AVERAGE_COL') and C.VOTE_AVERAGE_COL in genre_specific_movies.columns:
                         movies_to_sample_from = genre_specific_movies.sort_values(by=C.VOTE_AVERAGE_COL, ascending=False)
                    
                    sampled_movie = movies_to_sample_from.head(1) # Prendre le meilleur ou un échantillon
                    if not sampled_movie.empty:
                        movie_id_to_add = sampled_movie[C.ITEM_ID_COL].iloc[0]
                        if movie_id_to_add not in final_prefs_for_model: # Double vérification
                            final_prefs_for_model[movie_id_to_add] = 4.0 # Note implicite positive
                            # print(f"DEBUG: Ajout implicite du film {movie_id_to_add} (genre {fav_genre}) avec note 4.0")
                            count_added_genre_movies +=1
        
        num_total_prefs = len(final_prefs_for_model)
        st.info(f"Vous avez fourni {num_total_prefs} indication(s) de préférence (notes, favoris, détestés, et genres implicites). Au moins {min_prefs_needed} requises.")

        if form_submitted:
            if num_total_prefs < min_prefs_needed:
                st.warning(f"Veuillez fournir au moins {min_prefs_needed} indications.")
            else:
                ratings_list_for_model = []
                temp_uid = -1 # ID pour le modèle temporaire
                for mid, rval in final_prefs_for_model.items():
                    ratings_list_for_model.append({C.USER_ID_COL: temp_uid, C.ITEM_ID_COL: mid, C.RATING_COL: rval, C.TIMESTAMP_COL: int(pd.Timestamp.now().timestamp())})
                
                if ratings_list_for_model:
                    df_new_ratings = pd.DataFrame(ratings_list_for_model)
                    # print(f"DEBUG App.py: DataFrame pour l'entraînement du nouveau modèle:\n{df_new_ratings}")
                    try:
                        from surprise import Dataset, Reader
                        features = ["Genre_binary", "Tags_tfidf", "tmdb_vote_average", "Year_of_release", "title_tfidf", "average_ratings"]
                        actual_features = []
                        for feat in features: # Vérification basique de la disponibilité des données pour chaque feature
                            if feat == "Genre_binary" and not (hasattr(C, 'GENRES_COL') and C.GENRES_COL in df_items_global.columns): continue
                            if feat == "Tags_tfidf" and not (C.CONTENT_PATH / C.TAGS_FILENAME).is_file(): continue
                            if feat == "tmdb_vote_average" and not (hasattr(C, 'VOTE_AVERAGE_COL') and C.VOTE_AVERAGE_COL in df_items_global.columns): continue
                            if feat == "Year_of_release" and not (hasattr(C, 'RELEASE_YEAR_COL') and C.RELEASE_YEAR_COL in df_items_global.columns): continue
                            if feat == "title_tfidf" and not (hasattr(C, 'LABEL_COL') and C.LABEL_COL in df_items_global.columns): continue
                            if feat == "average_ratings" and (df_ratings_global.empty or getattr(C, 'RATING_COL', 'rating') not in df_ratings_global): continue
                            actual_features.append(feat)
                        
                        if not actual_features: raise ValueError("Aucune feature de contenu n'a pu être sélectionnée.")
                        # print(f"DEBUG App.py: Features utilisées pour cb_new_model: {actual_features}")

                        cb_model = ContentBased(features_methods=actual_features, regressor_method='linear')
                        reader = Reader(rating_scale=C.RATINGS_SCALE)
                        data = Dataset.load_from_df(df_new_ratings[[C.USER_ID_COL, C.ITEM_ID_COL, C.RATING_COL]], reader)
                        trainset = data.build_full_trainset()
                        # print(f"App.py: Entraînement nouveau profil. Trainset: {trainset.n_ratings} notes. Moyenne: {trainset.global_mean:.2f}")
                        
                        cb_model.fit(trainset)
                        st.session_state.new_user_model_instance = cb_model
                        
                        st.success("Profil créé ! Affichage de vos recommandations...")
                        st.session_state.active_page = "new_user_recs_ready"
                        st.session_state.current_user_id = "temp_newly_profiled"
                        st.rerun()
                    except Exception as e_train:
                        st.error(f"Erreur lors de la création de votre profil : {e_train}")
                        print(f"ERREUR App.py (new user train): {e_train}")
                else: st.error("Aucune préférence valide enregistrée.")


elif st.session_state.active_page == "new_user_recs_ready" and st.session_state.current_user_id == "temp_newly_profiled":
    st.header("Vos Recommandations Personnalisées (Nouveau Profil)")
    if 'new_user_model_instance' in st.session_state and st.session_state.new_user_model_instance is not None:
        model_instance = st.session_state.new_user_model_instance
        uid_for_pred = -1
        
        all_movies = content.get_all_movies_for_selection()
        if not all_movies.empty:
            # Exclure les films pour lesquels une préférence a été explicitement donnée
            # Reconstruire la liste des IDs avec préférence à partir de la session_state
            prefs_for_exclusion = {}
            for mid, r_val in st.session_state.get('new_user_ratings', {}).items(): prefs_for_exclusion[mid] = r_val
            if not df_items_global.empty: # Nécessaire pour mapper titres en IDs
                fav_titles = st.session_state.get('new_user_fav_movie_titles', [])
                dis_titles = st.session_state.get('new_user_disliked_movie_titles', [])
                for title in fav_titles:
                    movie_entry = df_items_global[df_items_global[C.LABEL_COL] == title]
                    if not movie_entry.empty: prefs_for_exclusion[movie_entry[C.ITEM_ID_COL].iloc[0]] = 5.0
                for title in dis_titles:
                    movie_entry = df_items_global[df_items_global[C.LABEL_COL] == title]
                    if not movie_entry.empty: prefs_for_exclusion[movie_entry[C.ITEM_ID_COL].iloc[0]] = 0.5
            # Pour les films ajoutés via genre favori, ils ont déjà été mis dans final_prefs_for_model.
            # Ce dict n'est pas en session. Il faudrait le reconstruire ou stocker les IDs ajoutés.
            # Pour l'instant, l'exclusion se base sur notes, favoris, détestés.
            # Les films ajoutés via genre pourraient être recommandés s'ils n'ont pas été notés. Ce n'est pas idéal.
            # Solution rapide: les films ajoutés par genre sont ceux qui ont eu 4.0
            # Mais `final_prefs_for_model` qui contenait cette info n'est pas en session.
            # On utilise `prefs_for_exclusion` qui contient notes explicites + fav/disliked.
            ids_with_pref = set(prefs_for_exclusion.keys())

            movies_to_predict = [mid for mid in all_movies[C.ITEM_ID_COL].tolist() if mid not in ids_with_pref]
            
            preds_list = []
            if movies_to_predict:
                for item_id in movies_to_predict:
                    try:
                        prediction = model_instance.predict(uid=uid_for_pred, iid=item_id)
                        preds_list.append({C.ITEM_ID_COL: prediction.iid, 'estimated_score': prediction.est})
                    except: continue

                if preds_list:
                    recs = pd.DataFrame(preds_list).sort_values(by='estimated_score', ascending=False).head(N_RECOS_PERSONNALISEES_TOTAL_FETCH)
                    details = content.get_movie_details_list(recs[C.ITEM_ID_COL].tolist())
                    final_recs = pd.DataFrame()
                    if not pd.DataFrame(details).empty and hasattr(C, 'ITEM_ID_COL') and C.ITEM_ID_COL in pd.DataFrame(details).columns:
                        final_recs = pd.merge(recs, pd.DataFrame(details), on=C.ITEM_ID_COL, how='left')
                    else: 
                        final_recs = recs.copy()
                        for col_attr in ['LABEL_COL', 'GENRES_COL', 'RELEASE_YEAR_COL', 'VOTE_AVERAGE_COL']:
                            col_name = getattr(C, col_attr, None)
                            if col_name and col_name not in final_recs.columns: final_recs[col_name] = pd.NA
                    
                    if not final_recs.empty:
                        display_movie_recommendations_section(final_recs, title="Voici quelques films basés sur vos notes !")
                    else: st.info("Aucune recommandation finale à afficher.")
                else: st.info("Impossible de générer des prédictions.")
            else: st.info("Aucun film à prédire (tous ont peut-être reçu une indication de préférence).")
        else: st.error("Liste globale de films non disponible.")
    else:
        st.warning("Le modèle pour votre nouveau profil n'a pas pu être chargé. Veuillez recréer votre profil.")
        if st.button("Recréer le profil"):
            st.session_state.active_page = "new_user_profiling"; st.session_state.current_user_id = "new_user_temp"
            st.session_state.new_user_ratings, st.session_state.new_user_fav_movie_titles, st.session_state.new_user_disliked_movie_titles, st.session_state.new_user_fav_genres = {}, [], [], []
            if 'new_user_model_instance' in st.session_state: del st.session_state.new_user_model_instance
            st.rerun()
else:
    if st.session_state.active_page not in ["general", "user_specific", "new_user_profiling", "new_user_recs_ready"]:
        pass # Évite les messages de debug pour les états non explicitement gérés

st.sidebar.markdown("---")
st.sidebar.info("Projet Recommender Systems MLSMM2156")