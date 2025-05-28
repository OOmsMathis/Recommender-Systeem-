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

# --- Fonction d'aide pour la sérialisation JSON ---
def convert_to_native_python_types(data):
    if isinstance(data, dict): return {k: convert_to_native_python_types(v) for k, v in data.items()}
    if isinstance(data, list): return [convert_to_native_python_types(i) for i in data]
    if isinstance(data, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)): return int(data)
    if isinstance(data, (np.float_, np.float16, np.float32, np.float64)): return float(data)
    if isinstance(data, (np.bool_)): return bool(data)
    if isinstance(data, pd.Timestamp): return data.isoformat()
    return data

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

st.sidebar.markdown("---")
st.sidebar.info("Projet Recommender Systems MLSMM2156")