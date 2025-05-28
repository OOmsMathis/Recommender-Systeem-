# app.py
import streamlit as st
import pandas as pd
import os
import re
import random # Pour échantillonner les films de genre favori
import time # Ajouté pour le timestamp

import constants as C_module
C = C_module.Constant()
import content
import recommender
from loaders import load_items, load_ratings # Import direct pour s'assurer du chargement

# --- Constantes pour le nombre de recommandations ---
N_TOP_GENERAL = 30
N_RECOS_PERSONNALISEES_INITIAL_DISPLAY = 10
N_RECOS_PERSONNALISEES_PER_PAGE = 10
N_RECOS_PERSONNALISEES_TOTAL_FETCH = 30

# --- Chargement des données ---
# Essayer de charger explicitement au cas où les fichiers auraient changé.
# df_items_global et df_ratings_global sont utilisés globalement dans ce script.
try:
    df_items_global = load_items()
    df_ratings_global = load_ratings()
    if df_items_global.empty:
        print("app.py: ERREUR CRITIQUE - df_items_global est vide après chargement direct.")
        # st.error("Les données des films n'ont pas pu être chargées. Certaines fonctionnalités peuvent être indisponibles.")
    if df_ratings_global.empty:
        print("app.py: ERREUR CRITIQUE - df_ratings_global est vide après chargement direct.")
        # st.error("Les données des évaluations n'ont pas pu être chargées. Certaines fonctionnalités peuvent être indisponibles.")

except Exception as e_load:
    print(f"app.py: ERREUR FATALE lors du chargement initial des données: {e_load}")
    _cols_items = [getattr(C, col_attr, col_attr.lower()) for col_attr in ['ITEM_ID_COL', 'LABEL_COL', 'GENRES_COL', 'RELEASE_YEAR_COL', 'TMDB_ID_COL', 'VOTE_AVERAGE_COL', 'VOTE_COUNT_COL'] if hasattr(C, col_attr)]
    df_items_global = pd.DataFrame(columns=_cols_items)
    _cols_ratings = [getattr(C, col_attr, col_attr.lower()) for col_attr in ['USER_ID_COL', 'ITEM_ID_COL', 'RATING_COL'] if hasattr(C, col_attr)]
    df_ratings_global = pd.DataFrame(columns=_cols_ratings)
    st.error("Erreur critique lors du chargement des données initiales. L'application risque de ne pas fonctionner correctement.")


st.set_page_config(page_title="TMDB Recommender Assistance", layout="wide")
st.title("🎬 TMDB Recommender System Assistance")

# --- Fonctions de récupération de données ---
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
    else:
        items_to_consider[vote_count_col_to_use] = 100 

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
    if df_items_global.empty or not hasattr(C, 'GENRES_COL') or C.GENRES_COL not in df_items_global.columns or \
       not hasattr(C, 'VOTE_AVERAGE_COL') or C.VOTE_AVERAGE_COL not in df_items_global.columns:
        return pd.DataFrame()

    # S'assurer que la colonne GENRES_COL est de type string pour le .str.contains()
    # et que 'genre' est échappé pour éviter les problèmes avec les caractères spéciaux regex.
    genre_movies_df = df_items_global[
        df_items_global[C.GENRES_COL].astype(str).str.contains(re.escape(genre), case=False, na=False, regex=True)
    ]
    if genre_movies_df.empty: return pd.DataFrame()

    items_to_consider = genre_movies_df.copy()
    items_to_consider[C.VOTE_AVERAGE_COL] = pd.to_numeric(items_to_consider[C.VOTE_AVERAGE_COL], errors='coerce')

    vote_count_col_to_use = 'temp_vote_count_genre'
    if hasattr(C, 'VOTE_COUNT_COL') and C.VOTE_COUNT_COL in items_to_consider.columns:
        items_to_consider[C.VOTE_COUNT_COL] = pd.to_numeric(items_to_consider[C.VOTE_COUNT_COL], errors='coerce').fillna(0)
        vote_count_col_to_use = C.VOTE_COUNT_COL
    else:
        items_to_consider[vote_count_col_to_use] = 50

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
    item_id_col, label_col, genres_col, year_col, tmdb_id_col, vote_avg_col = \
        C.ITEM_ID_COL, C.LABEL_COL, C.GENRES_COL, C.RELEASE_YEAR_COL, C.TMDB_ID_COL, C.VOTE_AVERAGE_COL
    
    can_make_links = (hasattr(C, 'TMDB_ID_COL') and tmdb_id_col in df_items_global.columns and \
                      item_id_col in df_items_global.columns and not df_items_global[tmdb_id_col].isnull().all())
    
    for _, row in df_to_display.iterrows():
        with st.container(): # Utiliser st.container pour un meilleur groupement visuel
            title_display = str(row.get(label_col, "Titre Inconnu"))
            movie_id_current = row.get(item_id_col)

            if can_make_links and pd.notna(movie_id_current):
                tmdb_id_series = df_items_global.loc[df_items_global[item_id_col] == movie_id_current, tmdb_id_col]
                if not tmdb_id_series.empty:
                    tmdb_id_val = tmdb_id_series.iloc[0]
                    if pd.notna(tmdb_id_val):
                        try:
                            title_display = f"[{title_display}](https://themoviedb.org/movie/{int(tmdb_id_val)})"
                        except ValueError: 
                            pass 
            
            col_info, col_pred_score, col_global_score = st.columns([6, 2, 2]) # Ajuster les ratios si besoin
            with col_info:
                st.markdown(f"**{title_display}**", unsafe_allow_html=True)
                genres_val = str(row.get(genres_col, "N/A"))
                year_val = row.get(year_col)
                year_display = int(year_val) if pd.notna(year_val) and year_val != 0 else "N/A"
                st.caption(f"{genres_val} ({year_display})")
            with col_pred_score:
                if 'estimated_score' in row and pd.notna(row['estimated_score']):
                    st.markdown(f"<div style='font-size: small; text-align: center;'>Prédiction:<br><b>{row['estimated_score']:.1f}/10</b></div>", unsafe_allow_html=True)
            with col_global_score:
                tmdb_avg_val_display = row.get('tmdb_vote_average', row.get(vote_avg_col))
                if pd.notna(tmdb_avg_val_display):
                    try:
                        st.markdown(f"<div style='font-size: small; text-align: center;'>Note Globale:<br><b>{pd.to_numeric(tmdb_avg_val_display, errors='coerce'):.2f}/10</b></div>", unsafe_allow_html=True)
                    except: pass

def display_movie_recommendations_section(recs_df, title="Recommandations", page_size=N_RECOS_PERSONNALISEES_PER_PAGE):
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
    
    display_movie_cards(display_df_all.head(page_size)) # Affiche la première page
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
                display_movie_cards(chunk_to_display)
        idx = end_idx_chunk

# --- Session State & Sidebar ---
if 'active_page' not in st.session_state: st.session_state.active_page = "general"
if 'current_user_id' not in st.session_state: st.session_state.current_user_id = None
if 'new_user_ratings' not in st.session_state: st.session_state.new_user_ratings = {}
if 'new_user_name_input' not in st.session_state: st.session_state.new_user_name_input = ''
if 'last_selected_user_id' not in st.session_state: st.session_state.last_selected_user_id = None

# --- Sidebar ---
st.sidebar.header("Filtres et Options")

# Correction pour la liste des genres uniques
all_genres_list_sidebar = ["Tous les genres"]
if not df_items_global.empty and hasattr(C, 'GENRES_COL') and C.GENRES_COL in df_items_global.columns:
    try:
        # S'assurer que la colonne GENRES_COL est traitée comme une chaîne, gérer les NaN avant split
        genres_series = df_items_global[C.GENRES_COL].fillna('').astype(str)
        
        # Séparer les genres par '|', mettre chaque genre sur une nouvelle ligne (explode),
        # enlever les espaces superflus (strip), filtrer les chaînes vides ou les placeholders,
        # obtenir les genres uniques, puis les trier.
        s_genres = genres_series.str.split('|').explode()
        unique_sidebar_genres = sorted([
            g.strip() for g in s_genres.unique() if g.strip() and g.strip().lower() != '(no genres listed)'
        ])
        if unique_sidebar_genres:
            all_genres_list_sidebar.extend(unique_sidebar_genres)
    except Exception as e_g_sb:
        print(f"Erreur sidebar (liste genres): {e_g_sb}")
        st.sidebar.error("Erreur chargement des genres.") # Informer l'utilisateur dans l'UI

selected_genre = st.sidebar.selectbox("Filtrer par genre :", all_genres_list_sidebar, key="genre_filter_sb")

slider_min, slider_max, current_slider_val = 1900, pd.Timestamp.now().year, (1900, pd.Timestamp.now().year)
if not df_items_global.empty and hasattr(C, 'RELEASE_YEAR_COL') and C.RELEASE_YEAR_COL in df_items_global.columns:
    valid_years = pd.to_numeric(df_items_global[C.RELEASE_YEAR_COL], errors='coerce').dropna()
    if not valid_years.empty:
        calc_min, calc_max = int(valid_years.min()), int(valid_years.max())
        if calc_min <= calc_max and calc_min > 1800: # Ajout d'une vérification de sanité pour min_year
            slider_min = calc_min
        if calc_max <= pd.Timestamp.now().year + 5: # Ajout d'une vérification de sanité pour max_year
             slider_max = calc_max
        current_slider_val = (slider_min, slider_max) # Mettre à jour la valeur par défaut du slider
if slider_max < slider_min: slider_max = slider_min # Correction si max < min après calculs
selected_year_range = st.sidebar.slider("Filtrer par année :", min_value=slider_min, max_value=slider_max, value=current_slider_val, key="year_filter_sb")

st.sidebar.markdown("---")
st.sidebar.header("👤 Espace Utilisateur")

user_opts = ["Voir les Tops Généraux"]
can_sel_existing = not df_ratings_global.empty and hasattr(C, 'USER_ID_COL') and \
                   C.USER_ID_COL in df_ratings_global.columns and \
                   not df_ratings_global[C.USER_ID_COL].empty

if can_sel_existing: user_opts.append("Se connecter (ID existant)")
user_opts.append("Créer un nouveau profil")

idx_radio = 0
try:
    if st.session_state.active_page == "general": idx_radio = user_opts.index("Voir les Tops Généraux")
    elif st.session_state.active_page == "new_user_profiling": idx_radio = user_opts.index("Créer un nouveau profil")
    elif st.session_state.active_page == "user_specific" and "Se connecter (ID existant)" in user_opts:
        idx_radio = user_opts.index("Se connecter (ID existant)")
except ValueError: idx_radio = 0 # Fallback si l'option n'est pas trouvée

user_sel_opt = st.sidebar.radio("Choisissez une option :", user_opts, key="user_sel_main_radio", index=idx_radio)

orig_page, orig_uid = st.session_state.active_page, st.session_state.current_user_id
rerun_needed_after_sidebar_change = False # Renommé pour plus de clarté

if user_sel_opt == "Voir les Tops Généraux":
    st.session_state.active_page, st.session_state.current_user_id = "general", None
elif user_sel_opt == "Créer un nouveau profil":
    st.session_state.active_page, st.session_state.current_user_id = "new_user_profiling", "new_user_temp"
    if orig_page != "new_user_profiling": 
        st.session_state.new_user_ratings = {}
        st.session_state.new_user_name_input = ''
elif user_sel_opt == "Se connecter (ID existant)" and can_sel_existing:
    st.session_state.active_page = "user_specific"
    uids_list = sorted(df_ratings_global[C.USER_ID_COL].unique()) if not df_ratings_global.empty else []
    last_id_sel = st.session_state.last_selected_user_id
    if st.session_state.current_user_id is None or st.session_state.current_user_id == "new_user_temp":
        st.session_state.current_user_id = last_id_sel if last_id_sel in uids_list else (uids_list[0] if uids_list else None)

if st.session_state.active_page != orig_page or st.session_state.current_user_id != orig_uid:
    rerun_needed_after_sidebar_change = True
if rerun_needed_after_sidebar_change: st.rerun()

uid_for_reco = None
if st.session_state.active_page == "user_specific":
    if user_sel_opt == "Se connecter (ID existant)" and can_sel_existing:
        uids_avail = sorted(df_ratings_global[C.USER_ID_COL].unique()) if not df_ratings_global.empty else []
        
        user_profiles_map = {}
        user_profiles_path = C.DATA_PATH / getattr(C, 'USER_PROFILES_FILENAME', 'user_profiles.csv')
        if os.path.exists(user_profiles_path):
            try:
                df_profiles = pd.read_csv(user_profiles_path)
                if 'userId' in df_profiles.columns and 'userName' in df_profiles.columns:
                     user_profiles_map = pd.Series(df_profiles.userName.values, index=df_profiles.userId).to_dict()
            except Exception as e_pf:
                print(f"Erreur chargement user_profiles.csv: {e_pf}")

        if uids_avail:
            disp_opts_func = lambda uid_val: f"{user_profiles_map.get(uid_val, 'Utilisateur')} (ID: {uid_val})"
            
            current_selection_uid = st.session_state.current_user_id
            if current_selection_uid not in uids_avail:
                current_selection_uid = uids_avail[0] if uids_avail else None
                st.session_state.current_user_id = current_selection_uid

            idx_sel_box = uids_avail.index(current_selection_uid) if current_selection_uid in uids_avail else 0
            
            uid_sel_box_val = st.sidebar.selectbox(
                f"Profil ID:", 
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
            st.sidebar.warning("Aucun utilisateur existant dans les données chargées.")
    elif st.session_state.current_user_id not in [None, "new_user_temp"]:
        uid_for_reco = st.session_state.current_user_id


# --- Logique d'Affichage Principal ---
if st.session_state.active_page == "general":
    st.header("🏆 Tops des Films")
    yr_min, yr_max = selected_year_range[0], selected_year_range[1]
    # Si un genre est sélectionné (autre que "Tous les genres"), on appelle get_top_genre_movies_tmdb
    if selected_genre != "Tous les genres":
        df_to_show = get_top_genre_movies_tmdb(genre=selected_genre, n=N_TOP_GENERAL, year_min_filter=yr_min, year_max_filter=yr_max)
    else: # Sinon, on appelle get_top_overall_movies_tmdb
        df_to_show = get_top_overall_movies_tmdb(n=N_TOP_GENERAL, year_min_filter=yr_min, year_max_filter=yr_max)
    display_movie_cards(df_to_show)


elif st.session_state.active_page == "user_specific" and uid_for_reco is not None:
    user_display_name = f"Utilisateur {uid_for_reco}" 
    user_profiles_path_main = C.DATA_PATH / getattr(C, 'USER_PROFILES_FILENAME', 'user_profiles.csv')
    if os.path.exists(user_profiles_path_main):
        try:
            df_profiles_main = pd.read_csv(user_profiles_path_main)
            if 'userId' in df_profiles_main.columns and 'userName' in df_profiles_main.columns:
                # S'assurer que uid_for_reco est du même type que les userId dans le CSV
                profile_entry = df_profiles_main[df_profiles_main['userId'].astype(type(uid_for_reco)) == uid_for_reco]
                if not profile_entry.empty:
                    user_display_name = profile_entry['userName'].iloc[0]
        except Exception as e_pf_main:
            print(f"Erreur lecture user_profiles.csv pour nom affichage: {e_pf_main}")

    st.header(f"Recommandations Personnalisées pour {user_display_name} (ID: {uid_for_reco})")
    yr_min_p, yr_max_p = selected_year_range[0], selected_year_range[1]
    genre_f = selected_genre if selected_genre != "Tous les genres" else None
    
    models_p_dir = str(C.DATA_PATH / 'recs')
    avail_model_files = [f for f in os.listdir(models_p_dir) if f.endswith('.p') and not 'personalized' in f.lower()] if os.path.exists(models_p_dir) and os.path.isdir(models_p_dir) else []
    
    if not avail_model_files:
        st.error(f"Aucun modèle général pré-entraîné trouvé dans {models_p_dir}. Veuillez exécuter le script d'entraînement.")
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
            if not m_file:
                 m_file = next((mfile for mfile in avail_model_files if pfix in mfile.lower()), None)

            if m_file:
                recs_data = recommender.get_top_n_recommendations(
                    uid_for_reco, 
                    m_file, 
                    n=N_RECOS_PERSONNALISEES_TOTAL_FETCH, 
                    filter_genre=genre_f, 
                    filter_year_range=(yr_min_p, yr_max_p)
                )
                display_movie_recommendations_section(recs_data, title=title_str)
            else:
                st.warning(f"Aucun modèle général de type '{mt}' trouvé.")

elif st.session_state.active_page == "new_user_profiling":
    st.header("👤 Créez votre profil de goûts")
    st.write("Pour nous aider à comprendre vos préférences, veuillez indiquer votre nom et noter quelques films.")
    
    new_user_name = st.text_input("Quel est votre nom ?", st.session_state.get('new_user_name_input', ''))
    st.session_state.new_user_name_input = new_user_name # Sauvegarder pour pré-remplir

    movies_for_profiling_pool = df_items_global.copy() if not df_items_global.empty else pd.DataFrame()
    sample_size = 20
    min_prefs_needed = 5
    movies_to_display_df = pd.DataFrame()

    if not movies_for_profiling_pool.empty:
        if hasattr(C, 'POPULARITY_COL') and C.POPULARITY_COL in movies_for_profiling_pool.columns and \
           not movies_for_profiling_pool[C.POPULARITY_COL].isnull().all():
            movies_for_profiling_pool[C.POPULARITY_COL] = pd.to_numeric(movies_for_profiling_pool[C.POPULARITY_COL], errors='coerce').fillna(0)
            movies_to_display_df_temp = movies_for_profiling_pool.sort_values(by=C.POPULARITY_COL, ascending=False).head(150)
            if len(movies_to_display_df_temp) >= sample_size:
                 movies_to_display_df = movies_to_display_df_temp.sample(n=sample_size, random_state=42)
        
        if movies_to_display_df.empty: 
            if len(movies_for_profiling_pool) >= sample_size:
                movies_to_display_df = movies_for_profiling_pool.sample(n=sample_size, random_state=42)
            else: 
                movies_to_display_df = movies_for_profiling_pool.copy()
    
    if movies_to_display_df.empty:
        st.error("Impossible de charger des films pour le profilage. Veuillez vérifier les données sources.")
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
                    if hasattr(C, 'GENRES_COL') and C.GENRES_COL in row and pd.notna(row[C.GENRES_COL]):
                        st.caption(f"Genres: {row[C.GENRES_COL]}")
                with col2:
                    opts = [None, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
                    fmt_fn = lambda x: "Pas de note" if x is None else f"{x} ★"
                    idx = opts.index(current_rating) if current_rating in opts else 0
                    new_r = st.selectbox(f"Note pour {title}:", opts, index=idx, format_func=fmt_fn, key=rating_key, label_visibility="collapsed")
                    if new_r is not None:
                        st.session_state.new_user_ratings[movie_id] = new_r
                    elif movie_id in st.session_state.new_user_ratings: 
                        del st.session_state.new_user_ratings[movie_id]
            
            form_submitted = st.form_submit_button("✔️ Enregistrer mon profil")

        final_prefs = st.session_state.new_user_ratings.copy()
        num_total_prefs = len(final_prefs)
        st.info(f"Vous avez fourni {num_total_prefs} note(s). Au moins {min_prefs_needed} sont requises.")

        if form_submitted:
            if not new_user_name.strip():
                st.warning("Veuillez entrer votre nom.")
            elif num_total_prefs < min_prefs_needed:
                st.warning(f"Veuillez noter au moins {min_prefs_needed} films.")
            else:
                try:
                    current_ratings_df_for_id = df_ratings_global 
                    if current_ratings_df_for_id.empty:
                         print("app.py (new_user): df_ratings_global vide, tentative de rechargement pour ID.")
                         current_ratings_df_for_id = load_ratings()
                         if current_ratings_df_for_id.empty:
                              st.error("Impossible de charger les évaluations existantes pour déterminer le nouvel User ID. Sauvegarde annulée.")
                              raise Exception("Ratings non chargeables pour génération d'ID.")
                    
                    new_user_id_val = current_ratings_df_for_id[C.USER_ID_COL].max() + 1 if not current_ratings_df_for_id.empty else 1
                    
                    ratings_to_save_list = []
                    current_ts = int(time.time())
                    for movie_id_key, rating_val_key in final_prefs.items():
                        ratings_to_save_list.append({
                            C.USER_ID_COL: new_user_id_val,
                            C.ITEM_ID_COL: movie_id_key,
                            C.RATING_COL: rating_val_key,
                            C.TIMESTAMP_COL: current_ts
                        })
                    
                    if ratings_to_save_list:
                        df_new_ratings_to_save_out = pd.DataFrame(ratings_to_save_list)
                        
                        pending_ratings_filepath = C.EVIDENCE_PATH / getattr(C, 'NEW_RATINGS_PENDING_FILENAME', 'new_ratings_pending.csv')
                        file_exists_pending = os.path.exists(pending_ratings_filepath)
                        df_new_ratings_to_save_out.to_csv(pending_ratings_filepath, mode='a', header=not file_exists_pending, index=False)

                        user_profiles_filepath = C.DATA_PATH / getattr(C, 'USER_PROFILES_FILENAME', 'user_profiles.csv')
                        user_profile_data_out = pd.DataFrame([{'userId': new_user_id_val, 'userName': new_user_name.strip()}])
                        file_exists_profiles = os.path.exists(user_profiles_filepath)
                        user_profile_data_out.to_csv(user_profiles_filepath, mode='a', header=not file_exists_profiles, index=False)
                        
                        st.success(f"Profil pour {new_user_name.strip()} (ID: {new_user_id_val}) et {len(ratings_to_save_list)} évaluations sauvegardés !")
                        st.info("Vos données ont été enregistrées. Les recommandations personnalisées seront disponibles après la prochaine mise à jour des modèles par l'administrateur du système.")
                        
                        st.session_state.new_user_ratings = {}
                        st.session_state.new_user_name_input = ''
                    else:
                        st.error("Aucune évaluation valide à sauvegarder.")
                except Exception as e_save_profile:
                    st.error(f"Erreur lors de la sauvegarde de votre profil : {e_save_profile}")
                    print(f"ERREUR App.py (new user save): {e_save_profile}")
else:
    if st.session_state.active_page not in ["general", "user_specific", "new_user_profiling"]:
        pass

st.sidebar.markdown("---")
st.sidebar.info("Projet Recommender Systems MLSMM2156")