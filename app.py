# app.py
import streamlit as st
import pandas as pd
import os
import re

import constants as C_module
C = C_module.Constant()
import content 
import recommender 
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
        st.error("Erreur critique lors du chargement des données initiales. L'application pourrait ne pas fonctionner correctement.")


st.set_page_config(page_title="TMDB Recommender Assistance", layout="wide")
st.title("🎬 TMDB Recommender System Assistance")

# --- Constantes pour le nombre de recommandations (DÉPLACÉES ICI HAUT) ---
N_TOP_GENERAL = 30
N_RECOS_PERSONNALISEES_INITIAL_DISPLAY = 10 # Afficher 10 films initialement
N_RECOS_PERSONNALISEES_TOTAL_FETCH = 30    # Récupérer 30 recos au total

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
    if C.VOTE_AVERAGE_COL in top_movies_df.columns: cols_out.append(C.VOTE_AVERAGE_COL)
    if vote_count_col_to_use == C.VOTE_COUNT_COL and C.VOTE_COUNT_COL in top_movies_df.columns:
        cols_out.append(C.VOTE_COUNT_COL)
    return top_movies_df[[col for col in cols_out if col in top_movies_df.columns]]

@st.cache_data
def get_top_genre_movies_tmdb(genre, n=N_TOP_GENERAL, year_min_filter=None, year_max_filter=None):
    if df_items_global.empty or C.GENRES_COL not in df_items_global.columns or not hasattr(C, 'VOTE_AVERAGE_COL') or C.VOTE_AVERAGE_COL not in df_items_global.columns:
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
    if C.VOTE_AVERAGE_COL in top_genre_df.columns: cols_out.append(C.VOTE_AVERAGE_COL)
    if vote_count_col_to_use == C.VOTE_COUNT_COL and C.VOTE_COUNT_COL in top_genre_df.columns:
        cols_out.append(C.VOTE_COUNT_COL)
    return top_genre_df[[col for col in cols_out if col in top_genre_df.columns]]

def display_movie_cards(df_to_display):
    if df_to_display.empty: return
    can_make_links = (hasattr(C, 'TMDB_ID_COL') and C.TMDB_ID_COL in df_items_global.columns and 
                      C.ITEM_ID_COL in df_items_global.columns)
    for _, row in df_to_display.iterrows():
        with st.container():
            title_display = str(row.get(C.LABEL_COL, "Titre Inconnu"))
            movie_id_current = row.get(C.ITEM_ID_COL)
            if can_make_links and pd.notna(movie_id_current):
                tmdb_id_series = df_items_global.loc[df_items_global[C.ITEM_ID_COL] == movie_id_current, C.TMDB_ID_COL]
                if not tmdb_id_series.empty:
                    tmdb_id_val = tmdb_id_series.iloc[0]
                    if pd.notna(tmdb_id_val):
                        try: title_display = f"[{title_display}](https://themoviedb.org/movie/{int(tmdb_id_val)})"
                        except: pass
            col_title_details, col_score_pred, col_score_tmdb_display = st.columns([5, 2, 2]) 
            with col_title_details:
                st.markdown(f"**{title_display}**", unsafe_allow_html=True)
                genres_val = str(row.get(C.GENRES_COL, ""))
                year_val = row.get(C.RELEASE_YEAR_COL)
                year_display = int(year_val) if pd.notna(year_val) and year_val != 0 else "N/A"
                st.caption(f"{genres_val} ({year_display})")
            with col_score_pred:
                if 'estimated_score' in row and pd.notna(row['estimated_score']): 
                    st.markdown(f"<div style='font-size: small; text-align: center;'>Prédictions:<br><b>{row['estimated_score']:.1f}/10</b></div>", unsafe_allow_html=True)
            with col_score_tmdb_display:
                tmdb_avg_val_display = row.get('tmdb_vote_average', row.get(C.VOTE_AVERAGE_COL)) # Utilise la colonne TMDB
                if pd.notna(tmdb_avg_val_display):
                    st.markdown(f"<div style='font-size: small; text-align: center;'>Note Globale:<br><b>{pd.to_numeric(tmdb_avg_val_display, errors='coerce'):.2f}/10</b></div>", unsafe_allow_html=True)

def display_movie_recommendations_section(recs_df, title="Recommandations", 
                                          initial_display_count=N_RECOS_PERSONNALISEES_INITIAL_DISPLAY, 
                                          page_size=N_RECOS_PERSONNALISEES_INITIAL_DISPLAY): # page_size est maintenant 10
    st.subheader(title)
    if recs_df.empty:
        st.info("Aucun film à afficher pour cette sélection.")
        return

    display_df_all = recs_df.copy()
    if 'estimated_score' in display_df_all.columns:
        display_df_all['estimated_score'] = pd.to_numeric(display_df_all['estimated_score'], errors='coerce') * 2
        display_df_all['estimated_score'] = display_df_all['estimated_score'].round(1)
    
    tmdb_avg_col_name_display = 'tmdb_vote_average' # Nom cohérent pour l'affichage
    if hasattr(C, 'VOTE_AVERAGE_COL') and C.VOTE_AVERAGE_COL in display_df_all.columns and tmdb_avg_col_name_display not in display_df_all.columns:
        display_df_all = display_df_all.rename(columns={C.VOTE_AVERAGE_COL: tmdb_avg_col_name_display})
    if tmdb_avg_col_name_display in display_df_all.columns:
         display_df_all[tmdb_avg_col_name_display] = pd.to_numeric(display_df_all[tmdb_avg_col_name_display], errors='coerce').round(2)

    # Afficher les premiers `page_size` (10)
    display_movie_cards(display_df_all.head(page_size))

    remaining_recs = display_df_all.iloc[page_size:]
    num_remaining_total = len(remaining_recs)
    
    idx = 0
    expander_count = 1
    while idx < num_remaining_total:
        end_idx = min(idx + page_size, num_remaining_total)
        chunk_to_display = remaining_recs.iloc[idx:end_idx]
        if not chunk_to_display.empty:
            with st.expander(f"Voir plus de recommandations ({expander_count * page_size + 1} - {expander_count * page_size + len(chunk_to_display)})..."):
                display_movie_cards(chunk_to_display)
        idx = end_idx
        expander_count +=1


if 'active_page' not in st.session_state: st.session_state.active_page = "general"
if 'current_user_id' not in st.session_state: st.session_state.current_user_id = None
if 'new_user_ratings' not in st.session_state: st.session_state.new_user_ratings = {}
if hasattr(C, 'MAXIME_USER_ID'): st.session_state.MAXIME_USER_ID = C.MAXIME_USER_ID
else: st.session_state.MAXIME_USER_ID = 0 

st.sidebar.header("Filtres et Options")
genre_list_for_select = ["Tous les genres"]
if not df_items_global.empty and C.GENRES_COL in df_items_global.columns:
    try:
        all_genres_set = set()
        df_items_global[C.GENRES_COL].dropna().astype(str).apply(
            lambda x: all_genres_set.update([genre.strip() for genre in str(x).replace(", ", "|").replace(",", "|").split('|') if genre.strip()])
        )
        unique_genres = sorted([g for g in list(all_genres_set) if g and g.lower() != '(no genres listed)'])
        if unique_genres: genre_list_for_select.extend(unique_genres)
    except Exception as e_genre: print(f"Erreur préparation filtre genre: {e_genre}")
selected_genre = st.sidebar.selectbox("Filtrer par genre :", genre_list_for_select)

min_year_default, max_year_default = 1900, pd.Timestamp.now().year 
if not df_items_global.empty and hasattr(C, 'RELEASE_YEAR_COL') and C.RELEASE_YEAR_COL in df_items_global.columns:
    valid_years = pd.to_numeric(df_items_global[C.RELEASE_YEAR_COL], errors='coerce').dropna()
    if not valid_years.empty and valid_years.min() <= valid_years.max():
        min_slider, max_slider = int(valid_years.min()), int(valid_years.max())
        if min_slider < max_slider : min_year_default, max_year_default = min_slider, max_slider
        elif min_slider == max_slider: min_year_default, max_year_default = min_slider -1 if min_slider > 1900 else min_slider, max_slider 
selected_year_range = st.sidebar.slider("Filtrer par année :", min_year_default, max_year_default, (min_year_default, max_year_default))

st.sidebar.markdown("---")
st.sidebar.header("👤 Espace Utilisateur")
user_options = ["Voir les Tops Généraux"]
known_users_map_default = {st.session_state.MAXIME_USER_ID: "Mon Profil (Maxime)"} 
known_users_map = getattr(C, 'KNOWN_USERS_MAP', known_users_map_default) 
can_select_existing_user = not df_ratings_global.empty and C.USER_ID_COL in df_ratings_global.columns and not df_ratings_global[C.USER_ID_COL].empty
if can_select_existing_user:
    user_options.append("Se connecter (ID existant)")
    maxime_is_selectable = st.session_state.MAXIME_USER_ID in known_users_map or \
                           (not df_ratings_global.empty and st.session_state.MAXIME_USER_ID in df_ratings_global[C.USER_ID_COL].unique())
    if maxime_is_selectable and "Mon Profil (Maxime)" not in user_options : user_options.insert(len(user_options)-1, "Mon Profil (Maxime)")
user_options.append("Créer un nouveau profil")
user_selection_option = st.sidebar.radio("Choisissez une option :", user_options, key="user_selection_main_radio")
current_user_id_for_reco = None

if user_selection_option == "Voir les Tops Généraux":
    st.session_state.active_page = "general"; st.session_state.current_user_id = None
elif user_selection_option == "Se connecter (ID existant)":
    st.session_state.active_page = "user_specific"
    user_ids_available = sorted(df_ratings_global[C.USER_ID_COL].unique())
    user_display_options = {uid: f"{known_users_map.get(uid, '')} (ID: {uid})" if uid in known_users_map else uid for uid in user_ids_available}
    default_user_val = user_ids_available[0] if user_ids_available else None
    if 'last_selected_user_id' in st.session_state and st.session_state.last_selected_user_id in user_ids_available: default_user_val = st.session_state.last_selected_user_id
    if default_user_val:
        user_id_selected = st.sidebar.selectbox(f"Profil ou {C.USER_ID_COL}:", options=user_ids_available, format_func=lambda x: user_display_options.get(x,x), index=user_ids_available.index(default_user_val))
        st.session_state.current_user_id = user_id_selected; st.session_state.last_selected_user_id = user_id_selected; current_user_id_for_reco = user_id_selected
    else: st.sidebar.warning("Aucun utilisateur existant.")
elif user_selection_option == "Mon Profil (Maxime)":
    st.session_state.active_page = "user_specific"; st.session_state.current_user_id = st.session_state.MAXIME_USER_ID; current_user_id_for_reco = st.session_state.MAXIME_USER_ID
    st.sidebar.success(f"Connecté: {known_users_map.get(st.session_state.MAXIME_USER_ID, 'Maxime')} (ID: {st.session_state.MAXIME_USER_ID})")
elif user_selection_option == "Créer un nouveau profil":
    st.session_state.active_page = "new_user_profiling"; st.session_state.current_user_id = "new_user_temp" 

if st.session_state.active_page == "general":
    st.header("🏆 Tops des Films")
    year_min, year_max = selected_year_range
    if selected_genre == "Tous les genres":
        top_movies_df = get_top_overall_movies_tmdb(n=N_TOP_GENERAL, year_min_filter=year_min, year_max_filter=year_max)
        display_movie_cards(top_movies_df) # Utilise display_movie_cards pour les tops aussi
    else:
        top_genre_movies_df = get_top_genre_movies_tmdb(genre=selected_genre, n=N_TOP_GENERAL, year_min_filter=year_min, year_max_filter=year_max)
        display_movie_cards(top_genre_movies_df)

elif st.session_state.active_page == "user_specific" and current_user_id_for_reco is not None:
    user_display_name = known_users_map.get(current_user_id_for_reco, f"Utilisateur {current_user_id_for_reco}")
    st.header(f"Recommandations Personnalisées pour {user_display_name}")
    year_min_perso, year_max_perso = selected_year_range
    genre_filter_to_apply = selected_genre if selected_genre != "Tous les genres" else None
    models_dir = str(C.DATA_PATH / 'recs')
    available_models_files = [f for f in os.listdir(models_dir) if f.endswith('.p')] if os.path.exists(models_dir) and os.path.isdir(models_dir) else []
    if not available_models_files: st.error(f"Aucun modèle entraîné trouvé dans {models_dir}.")
    else:
        model_types_to_display = st.multiselect( "Afficher les recommandations de :",
            ["Utilisateurs Similaires", "Contenu Similaire", "Modèle Général (SVD)"],
            default=["Utilisateurs Similaires", "Contenu Similaire"] )

        if "Utilisateurs Similaires" in model_types_to_display:
            user_based_model_file = next((m for m in available_models_files if 'user_based' in m.lower() or 'userbased' in m.lower()), None)
            if user_based_model_file:
                recs_user_based = recommender.get_top_n_recommendations(current_user_id_for_reco, user_based_model_file, n=N_RECOS_PERSONNALISEES_TOTAL_FETCH, 
                                                                      filter_genre=genre_filter_to_apply, filter_year_range=(year_min_perso, year_max_perso))
                display_movie_recommendations_section(recs_user_based, title="❤️ Pourrait vous plaire (Utilisateurs similaires)", initial_display_count=N_RECOS_PERSONNALISEES_INITIAL_DISPLAY, page_size=N_RECOS_PERSONNALISEES_INITIAL_DISPLAY)
            else: st.warning("Aucun modèle User-Based trouvé.")
        
        if "Contenu Similaire" in model_types_to_display:
            content_based_model_file = next((m for m in available_models_files if 'content_based' in m.lower() or 'contentbased' in m.lower()), None)
            if content_based_model_file:
                recs_content_based = recommender.get_top_n_recommendations(current_user_id_for_reco, content_based_model_file, n=N_RECOS_PERSONNALISEES_TOTAL_FETCH,
                                                                         filter_genre=genre_filter_to_apply, filter_year_range=(year_min_perso, year_max_perso))
                display_movie_recommendations_section(recs_content_based, title="👍 Basé sur vos goûts (Contenu similaire)", initial_display_count=N_RECOS_PERSONNALISEES_INITIAL_DISPLAY, page_size=N_RECOS_PERSONNALISEES_INITIAL_DISPLAY)
            else: st.warning("Aucun modèle Content-Based trouvé.")

        if "Modèle Général (SVD)" in model_types_to_display:
            svd_model_file = next((m for m in available_models_files if 'svd' in m.lower() and 'personalized' not in m.lower() and 'maxime' not in m.lower()), None)
            if svd_model_file:
                recs_svd = recommender.get_top_n_recommendations(current_user_id_for_reco, svd_model_file, n=N_RECOS_PERSONNALISEES_TOTAL_FETCH,
                                                                     filter_genre=genre_filter_to_apply, filter_year_range=(year_min_perso, year_max_perso))
                display_movie_recommendations_section(recs_svd, title="✨ Recommandations Générales (SVD)", initial_display_count=N_RECOS_PERSONNALISEES_INITIAL_DISPLAY, page_size=N_RECOS_PERSONNALISEES_INITIAL_DISPLAY)
            else: st.warning("Aucun modèle SVD général trouvé.")
        
        if current_user_id_for_reco == st.session_state.MAXIME_USER_ID:
            maxime_model_file = next((m for m in available_models_files if 'personalized' in m.lower() or str(st.session_state.MAXIME_USER_ID) in m or 'maxime' in m.lower()), None)
            if maxime_model_file:
                st.markdown("---")
                recs_maxime = recommender.get_top_n_recommendations(current_user_id_for_reco, maxime_model_file, n=N_RECOS_PERSONNALISEES_TOTAL_FETCH,
                                                                     filter_genre=genre_filter_to_apply, filter_year_range=(year_min_perso, year_max_perso))
                display_movie_recommendations_section(recs_maxime, title=f"🌟 Spécialement pour {known_users_map.get(st.session_state.MAXIME_USER_ID, 'Maxime')} (Modèle Personnalisé)", initial_display_count=N_RECOS_PERSONNALISEES_INITIAL_DISPLAY, page_size=N_RECOS_PERSONNALISEES_INITIAL_DISPLAY)

elif st.session_state.active_page == "new_user_profiling":
    st.header("👤 Création de votre Profil")
    st.write("Veuillez noter au moins 5 films pour que nous puissions vous connaître :")
    movies_for_rating_query = df_items_global ; sample_size_profiling = 15; min_ratings_needed_profiling = 5
    if hasattr(C, 'POPULARITY_COL') and C.POPULARITY_COL in df_items_global.columns and not df_items_global[C.POPULARITY_COL].isnull().all():
        movies_for_rating_query[C.POPULARITY_COL] = pd.to_numeric(movies_for_rating_query[C.POPULARITY_COL], errors='coerce').fillna(0)
        movies_to_rate_df = movies_for_rating_query.sort_values(by=C.POPULARITY_COL, ascending=False).head(30)
        if len(movies_to_rate_df) >= sample_size_profiling: movies_to_rate_df = movies_to_rate_df.sample(n=sample_size_profiling, random_state=42)
        elif not movies_to_rate_df.empty: movies_to_rate_df = movies_to_rate_df.sample(n=min(sample_size_profiling, len(movies_to_rate_df)), random_state=42)
        else: movies_to_rate_df = df_items_global.sample(n=min(sample_size_profiling, len(df_items_global)), random_state=42) if not df_items_global.empty else pd.DataFrame()
    elif not df_items_global.empty: movies_to_rate_df = df_items_global.sample(n=min(sample_size_profiling, len(df_items_global)), random_state=42)
    else: movies_to_rate_df = pd.DataFrame()
    if movies_to_rate_df.empty: st.error("Impossible de charger des films à noter.")
    else:
        with st.form(key="new_user_rating_form"):
            st.markdown("**Instructions:** Donnez une note (0.5-5) ou laissez sur 'Pas de note'.")
            for _, row in movies_to_rate_df.iterrows():
                movie_id, title = row[C.ITEM_ID_COL], row[C.LABEL_COL]
                rating_key = f"rating_select_{movie_id}" 
                current_rating_value = st.session_state.new_user_ratings.get(movie_id)
                col1, col2 = st.columns([3,2]);
                with col1: st.write(f"**{title}**"); 
                if C.GENRES_COL in row and pd.notna(row[C.GENRES_COL]): st.caption(f"Genres: {row[C.GENRES_COL]}")
                with col2:
                    note_options = [None, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
                    format_func = lambda x: "Pas de note" if x is None else f"{x} ★"
                    try: current_idx = note_options.index(current_rating_value)
                    except ValueError: current_idx = 0 
                    new_rating = st.selectbox(f"Note:", options=note_options, format_func=format_func, key=rating_key, index=current_idx)
                if new_rating != current_rating_value: st.session_state.new_user_ratings[movie_id] = new_rating
            submitted = st.form_submit_button(f"✔️ Terminer et Obtenir mes Recommandations")
        num_rated = len([r for r in st.session_state.new_user_ratings.values() if r is not None])
        st.info(f"Vous avez noté {num_rated} film(s). Au moins {min_ratings_needed_profiling} requis.")
        if submitted:
            if num_rated < min_ratings_needed_profiling: st.warning(f"Veuillez noter au moins {min_ratings_needed_profiling} films.")
            else:
                st.session_state.active_page = "user_specific"; st.session_state.current_user_id = "new_user_profiled"
                new_user_ratings_list = []
                temp_new_user_id = -1 
                for mid, r_val in st.session_state.new_user_ratings.items():
                    if r_val is not None: new_user_ratings_list.append({C.USER_ID_COL: temp_new_user_id, C.ITEM_ID_COL: mid, C.RATING_COL: r_val, C.TIMESTAMP_COL: int(pd.Timestamp.now().timestamp())})
                if new_user_ratings_list:
                    df_new_user_ratings = pd.DataFrame(new_user_ratings_list)
                    from surprise import Dataset, Reader                     
                    cb_features_new = ["title_length", "Year_of_release", "Genre_binary"]
                    if hasattr(C, 'VOTE_AVERAGE_COL') and C.VOTE_AVERAGE_COL in df_items_global.columns: cb_features_new.append("tmdb_vote_average")
                    cb_new_model = ContentBased(features_methods=cb_features_new, regressor_method='ridge')
                    reader = Reader(rating_scale=C.RATINGS_SCALE)
                    new_data = Dataset.load_from_df(df_new_user_ratings, reader)
                    new_trainset = new_data.build_full_trainset()
                    cb_new_model.fit(new_trainset)
                    st.session_state.new_user_model_instance = cb_new_model
                    st.experimental_rerun()
                else: st.error("Aucune note valide enregistrée.")

if st.session_state.active_page == "user_specific" and st.session_state.current_user_id == "new_user_profiled":
    st.header("Vos Recommandations Personnalisées (Nouveau Profil)")
    if 'new_user_model_instance' in st.session_state and st.session_state.new_user_model_instance is not None:
        new_model_instance = st.session_state.new_user_model_instance
        temp_new_user_id = -1 
        all_movies_df_select = content.get_all_movies_for_selection()
        if not all_movies_df_select.empty:
            all_movie_ids = all_movies_df_select[C.ITEM_ID_COL].tolist()
            new_user_rated_ids = set(st.session_state.new_user_ratings.keys())
            movies_to_predict_for = [mid for mid in all_movie_ids if mid not in new_user_rated_ids]
            predictions_new = []
            if movies_to_predict_for:
                for movie_id_np in movies_to_predict_for:
                    pred_np = new_model_instance.predict(uid=temp_new_user_id, iid=movie_id_np) 
                    predictions_new.append({C.ITEM_ID_COL: pred_np.iid, 'estimated_score': pred_np.est})
                if predictions_new:
                    recs_new_df = pd.DataFrame(predictions_new).sort_values(by='estimated_score', ascending=False).head(N_RECOS_PERSONNALISEES_TOTAL_FETCH)
                    details_new = content.get_movie_details_list(recs_new_df[C.ITEM_ID_COL].tolist())
                    final_recs_new_df = pd.DataFrame()
                    if not pd.DataFrame(details_new).empty and C.ITEM_ID_COL in pd.DataFrame(details_new).columns:
                        final_recs_new_df = pd.merge(recs_new_df, pd.DataFrame(details_new), on=C.ITEM_ID_COL, how='left')
                    else:
                        final_recs_new_df = recs_new_df
                        for col_attr_name in ['LABEL_COL', 'GENRES_COL', 'RELEASE_YEAR_COL', 'VOTE_AVERAGE_COL', 'VOTE_COUNT_COL']:
                            col_name_add = getattr(C, col_attr_name, None)
                            if col_name_add and col_name_add not in final_recs_new_df.columns: final_recs_new_df[col_name_add] = pd.NA
                    if hasattr(C, 'VOTE_AVERAGE_COL') and C.VOTE_AVERAGE_COL in final_recs_new_df.columns:
                        final_recs_new_df = final_recs_new_df.rename(columns={C.VOTE_AVERAGE_COL: 'tmdb_vote_average'})
                    display_movie_recommendations_section(final_recs_new_df, title="Voici quelques films basés sur vos notes !", initial_display_count=N_RECOS_PERSONNALISEES_INITIAL_DISPLAY, page_size=N_RECOS_PERSONNALISEES_INITIAL_DISPLAY)
                else: st.info("Impossible de générer des recommandations.")
        else: st.error("Liste de films globale non disponible.")
    else: st.info("Le modèle pour votre nouveau profil n'est pas encore prêt.")

st.sidebar.markdown("---")
st.sidebar.info("Projet Recommender Systems MLSMM2156")