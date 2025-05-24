import streamlit as st
import pandas as pd
from pathlib import Path

# Importer vos modules personnalisés
import constants as C
import content
import recommender

# --- Configuration de la Page Streamlit ---
st.set_page_config(
    page_title="Système de Recommandation de Films",
    page_icon="🎬",
    layout="wide"
)

# --- Chargement des Données Globales (mis en cache par Streamlit via content.py) ---
@st.cache_data # Cache le résultat de cette fonction pour éviter de recharger inutilement
def load_global_data():
    """Charge les données globales nécessaires au fonctionnement de l'application."""
    all_movie_features_df = content.get_all_movie_features()
    if all_movie_features_df.empty:
        st.error("Erreur critique: Impossible de charger les données des films depuis content.py.")
        return None, []
    all_movie_ids_list = all_movie_features_df.index.tolist()
    
    # Charger les ratings pour les recommandations populaires une seule fois
    ratings_df_for_popular = None
    try:
        ratings_file_path = C.Constant.EVIDENCE_PATH / C.Constant.RATINGS_FILENAME
        if ratings_file_path.exists():
            ratings_df_for_popular = pd.read_csv(ratings_file_path)
        else:
            st.warning(f"Fichier de ratings {ratings_file_path} non trouvé. Les recommandations populaires pourraient ne pas fonctionner.")
    except Exception as e:
        st.warning(f"Erreur lors du chargement des ratings pour les recommandations populaires: {e}")
        
    return all_movie_features_df, all_movie_ids_list, ratings_df_for_popular

ALL_MOVIE_FEATURES, ALL_MOVIE_IDS, RATINGS_DF_POPULAR = load_global_data()

# --- Définition des Profils Utilisateurs Personnalisés ---
# Ce dictionnaire doit être mis à jour manuellement si vous ajoutez/modifiez des profils
# avec recommender_building.py
# La clé est le nom affiché, la valeur contient l'ID numérique et le suffixe du nom du modèle.
PROFIL_UTILISATEURS_PERSONNALISES = {
    "Alice (Profil Personnalisé)": {
        "user_id": -1, # L'ID numérique utilisé dans recommender_building.py
        "model_config_name_suffix": "svd_implicit" # Le suffixe du modèle
    },
    "Bob (Profil Personnalisé)": { # Exemple d'un autre profil
        "user_id": -2,
        "model_config_name_suffix": "nmf_implicit" # Assurez-vous que ce modèle existe
    }
    # Ajoutez d'autres profils ici si vous en avez créé
}

# --- Interface Utilisateur ---
st.title("🎬 Système de Recommandation de Films")
st.markdown("Découvrez votre prochain film coup de cœur !")

# Options pour la sélection de l'utilisateur
user_type_options = ["Recommandations Générales (Populaires)"] + \
                    list(PROFIL_UTILISATEURS_PERSONNALISES.keys()) + \
                    ["Utilisateur MovieLens (par ID)"]

# Barre latérale pour les contrôles
with st.sidebar:
    st.header("⚙️ Vos Préférences")
    
    selected_user_type = st.selectbox(
        "Choisissez votre type de profil :",
        user_type_options,
        index=0 # Par défaut sur "Recommandations Générales"
    )

    target_user_id_input = None
    user_name_for_profile_input = None
    model_config_name_suffix_input = None

    if selected_user_type == "Utilisateur MovieLens (par ID)":
        # Max userId pour MovieLens small est 610
        target_user_id_input = st.number_input(
            "Entrez votre UserID MovieLens (1-610) :", 
            min_value=1, 
            max_value=610, # Ajustez si vous utilisez un autre dataset
            value=1, 
            step=1
        )
    elif selected_user_type in PROFIL_UTILISATEURS_PERSONNALISES:
        profile_details = PROFIL_UTILISATEURS_PERSONNALISES[selected_user_type]
        target_user_id_input = profile_details["user_id"]
        user_name_for_profile_input = selected_user_type.split(" (")[0] # Extrait "Alice" de "Alice (Profil...)"
        model_config_name_suffix_input = profile_details["model_config_name_suffix"]
        st.caption(f"Profil sélectionné : {user_name_for_profile_input} (ID interne: {target_user_id_input})")

    num_recommendations = st.slider(
        "Nombre de recommandations souhaitées :", 
        min_value=5, 
        max_value=20, 
        value=C.Constant.DEFAULT_N_RECOMMENDATIONS if hasattr(C.Constant, 'DEFAULT_N_RECOMMENDATIONS') else 10
    )

    recommend_button = st.button("Obtenir des Recommandations", type="primary", use_container_width=True)

# --- Logique de Recommandation et Affichage ---
if recommend_button and ALL_MOVIE_FEATURES is not None:
    st.subheader("✨ Vos Recommandations Personnalisées ✨")
    
    recommendations_ids = []
    with st.spinner("Recherche de films pour vous... Merci de patienter !"):
        if selected_user_type == "Recommandations Générales (Populaires)":
            # La fonction get_popular_movies_recommendations retourne une liste de (id, titre)
            # ou juste une liste d'IDs si all_movie_features_df n'est pas passé.
            # Ici, on veut juste les IDs pour être cohérent, les détails seront récupérés ensuite.
            raw_popular_recs = recommender.get_popular_movies_recommendations(
                n=num_recommendations, 
                ratings_df=RATINGS_DF_POPULAR # Utilise le DataFrame pré-chargé
            )
            if raw_popular_recs and isinstance(raw_popular_recs[0], tuple): # Si (id, titre)
                recommendations_ids = [rec[0] for rec in raw_popular_recs]
            else: # Si juste une liste d'IDs
                recommendations_ids = raw_popular_recs

        elif target_user_id_input is not None: # Cas profil personnalisé ou MovieLens ID
            recommendations_ids = recommender.generate_recommendations_for_user(
                user_id=target_user_id_input,
                n=num_recommendations,
                all_movie_ids=ALL_MOVIE_IDS,
                user_name_for_profile=user_name_for_profile_input, # Sera None si MovieLens ID
                model_config_name_suffix=model_config_name_suffix_input, # Sera None si MovieLens ID
                ratings_df_path=C.Constant.EVIDENCE_PATH / C.Constant.RATINGS_FILENAME # Pour le trainset général si besoin
            )
        else:
            st.warning("Veuillez sélectionner un type d'utilisateur ou un profil valide.")

    if not recommendations_ids:
        st.info("Nous n'avons pas pu générer de recommandations pour cette sélection. Essayez d'autres options !")
    else:
        st.success(f"Voici {len(recommendations_ids)} films qui pourraient vous plaire :")
        
        # Affichage en colonnes
        num_cols = 3 # Nombre de films par ligne
        cols = st.columns(num_cols)
        for i, movie_id_rec in enumerate(recommendations_ids):
            movie_details = content.get_movie_details(movie_id_rec, ALL_MOVIE_FEATURES)
            if movie_details is not None:
                col = cols[i % num_cols] # Répartit dans les colonnes
                with col:
                    st.markdown(f"##### {content.get_movie_title(movie_id_rec, ALL_MOVIE_FEATURES)}")
                    
                    poster_url = content.get_movie_poster_url(movie_id_rec, ALL_MOVIE_FEATURES)
                    if poster_url:
                        st.image(poster_url, use_column_width=True)
                    else:
                        # Placeholder si pas d'affiche
                        st.image(f"https://placehold.co/300x450/222/fff?text={content.get_movie_title(movie_id_rec, ALL_MOVIE_FEATURES)}", use_column_width=True)

                    genres = movie_details.get(C.Constant.GENRES_COL, "N/A")
                    st.caption(f"Genres: {genres}")
                    
                    # Afficher d'autres détails si souhaité
                    # overview = movie_details.get('overview', None) # Si 'overview' est dans vos features TMDB
                    # if pd.notna(overview):
                    #     with st.expander("Synopsis"):
                    #         st.markdown(f"<small>{overview[:200]}...</small>", unsafe_allow_html=True)
                    
                    # Lien IMDB si disponible (tmdbId est plus direct pour TMDB)
                    imdb_id = movie_details.get('imdbId', None)
                    if pd.notna(imdb_id):
                        st.markdown(f"<small>[Voir sur IMDB](https://www.imdb.com/title/tt{str(int(imdb_id)).zfill(7)}/)</small>", unsafe_allow_html=True)
                    st.markdown("---") # Séparateur
            else:
                st.warning(f"Détails non trouvés pour le film ID: {movie_id_rec}")
elif recommend_button and ALL_MOVIE_FEATURES is None:
    st.error("Les données des films n'ont pas pu être chargées. Impossible de générer des recommandations.")

st.sidebar.markdown("---")
st.sidebar.markdown("Projet MLSMM2156 - Système de Recommandation")
