# app.py
import streamlit as st
import pandas as pd
from pathlib import Path
import time # Pour simuler des délais ou gérer l'attente

# Importer vos modules personnalisés
import constants as C
import content
import recommender
# Supposons que vous ayez une fonction pour la création de profil dans recommender_building
# ou que vous importiez directement les fonctions nécessaires.
# Pour l'instant, nous allons esquisser l'idée.
# import recommender_building # Ou des fonctions spécifiques

# --- Configuration de la Page Streamlit ---
st.set_page_config(
    page_title="CinéRecommends",
    page_icon="🍿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Chargement des Données Globales ---
@st.cache_data
def load_global_data():
    all_movie_features_df = content.get_all_movie_features()
    if all_movie_features_df.empty:
        st.error("Erreur critique: Impossible de charger les données des films.")
        return None, [], None
    all_movie_ids_list = all_movie_features_df.index.tolist()
    
    ratings_df_for_popular = None
    try:
        ratings_file_path = C.Constant.EVIDENCE_PATH / C.Constant.RATINGS_FILENAME
        if ratings_file_path.exists():
            ratings_df_for_popular = pd.read_csv(ratings_file_path)
    except Exception as e:
        st.warning(f"Erreur chargement ratings pour populaires: {e}")
        
    return all_movie_features_df, all_movie_ids_list, ratings_df_for_popular

ALL_MOVIE_FEATURES, ALL_MOVIE_IDS, RATINGS_DF_POPULAR = load_global_data()

# --- Définition des Profils Utilisateurs Personnalisés (Exemples) ---
# À remplir avec vos profils réels créés par recommender_building.py
PROFIL_UTILISATEURS_PERSONNALISES = {
    "Alice (Profil SVD)": {
        "user_id": -1, 
        "user_name_for_profile": "alice", # Le nom utilisé pour les fichiers
        "model_config_name_suffix": "svd_implicit" 
    },
    "TestUser (Profil SVD)": { # Ajouté basé sur vos tests précédents
        "user_id": -1, # Important: doit être le même ID que celui utilisé pour générer les fichiers pour "testuser"
        "user_name_for_profile": "testuser",
        "model_config_name_suffix": "svd_implicit" 
    }
    # Ajoutez d'autres profils ici
}

# --- Fonctions Utilitaires pour l'Affichage ---
def display_recommendations(recommendation_ids, num_cols=4):
    if not recommendation_ids:
        st.info("Désolé, nous n'avons pas pu trouver de recommandations pour votre sélection.")
        return

    st.success(f"Voici {len(recommendation_ids)} films qui pourraient vous plaire :")
    
    cols = st.columns(num_cols)
    for i, movie_id_rec in enumerate(recommendation_ids):
        movie_details = content.get_movie_details(movie_id_rec, ALL_MOVIE_FEATURES)
        if movie_details is not None:
            col = cols[i % num_cols]
            with col:
                title = content.get_movie_title(movie_id_rec, ALL_MOVIE_FEATURES)
                st.markdown(f"<h6>{title}</h6>", unsafe_allow_html=True)
                
                poster_url = content.get_movie_poster_url(movie_id_rec, ALL_MOVIE_FEATURES)
                if poster_url:
                    st.image(poster_url, use_column_width="always")
                else:
                    st.image(f"http{":"}//placehold.co/300x450/222/fff?text={title.replace(' ','+')}", use_column_width="always")

                genres = movie_details.get(C.Constant.GENRES_COL, "N/A")
                st.caption(f"Genres: {genres}")
                
                avg_rating_tmdb = movie_details.get('vote_average_tmdb', None) # Si cette colonne existe de tmdb_full_features
                if pd.notna(avg_rating_tmdb) and avg_rating_tmdb > 0:
                     st.caption(f"Note TMDB: {avg_rating_tmdb:.1f}/10")

                imdb_id = movie_details.get('imdbId', None)
                if pd.notna(imdb_id):
                    st.markdown(f"<small>[IMDB](https://www.imdb.com/title/tt{str(int(imdb_id)).zfill(7)}/)</small>", unsafe_allow_html=True)
                st.markdown("---")
        else:
            st.warning(f"Détails non trouvés pour le film ID: {movie_id_rec}")


# --- Structure du Menu Principal (Navigation en Barre Latérale) ---
st.sidebar.title("CinéRecommends 🎬")
menu_choice = st.sidebar.radio(
    "Navigation Principale",
    ["Accueil (Populaires)", "Par Catégorie (Content-Based)", "Pour Moi (User-Based)", "Créer Mon Profil"],
    captions=["Les incontournables", "Selon vos goûts de genre", "Adapté à votre historique", "Recommandations sur mesure"]
)
st.sidebar.markdown("---")
num_recommendations_sidebar = st.sidebar.slider(
    "Nombre de films à recommander :", 
    min_value=3, 
    max_value=21, 
    value=6, # Un multiple de 3 pour un bel affichage en 3 colonnes
    step=3
)

# =========================== SECTION ACCUEIL (FILMS POPULAIRES) ===========================
if menu_choice == "Accueil (Populaires)":
    st.header("🌟 Les Films du Moment et les Mieux Notés")
    st.markdown("Une sélection des films les plus appréciés par notre communauté.")
    
    if ALL_MOVIE_FEATURES is not None and RATINGS_DF_POPULAR is not None:
        with st.spinner("Chargement des pépites..."):
            popular_recs_data = recommender.get_popular_movies_recommendations(
                n=num_recommendations_sidebar, 
                all_movie_features_df=ALL_MOVIE_FEATURES, # Pour obtenir les titres directement
                ratings_df=RATINGS_DF_POPULAR
            )
            if popular_recs_data and isinstance(popular_recs_data[0], tuple): # Si (id, titre)
                recommendations_ids = [rec[0] for rec in popular_recs_data]
            else: 
                recommendations_ids = popular_recs_data
            display_recommendations(recommendations_ids)
    else:
        st.error("Données nécessaires non disponibles pour les recommandations populaires.")

# =========================== SECTION PAR CATÉGORIE (CONTENT-BASED) ===========================
elif menu_choice == "Par Catégorie (Content-Based)":
    st.header("🎭 Recommandations par Genre (Content-Based)")
    st.markdown("Choisissez vos genres préférés et découvrez des films similaires.")

    if ALL_MOVIE_FEATURES is not None:
        # Extraire tous les genres uniques pour le multiselect
        # La colonne GENRES_COL contient des chaînes comme "Action|Adventure|Sci-Fi"
        all_genres_list = list(set([genre for sublist in ALL_MOVIE_FEATURES[C.Constant.GENRES_COL].str.split('|').dropna() for genre in sublist]))
        all_genres_list.sort()
        
        selected_genres = st.multiselect(
            "Sélectionnez un ou plusieurs genres :",
            options=all_genres_list,
            # default=["Action", "Adventure"] # Optionnel: genres par défaut
        )

        if st.button("Trouver des films par genre", use_container_width=True) and selected_genres:
            with st.spinner(f"Recherche de films dans les genres : {', '.join(selected_genres)}..."):
                # ----- ICI VOTRE LOGIQUE CONTENT-BASED -----
                # Exemple simplifié : filtrer les films qui ont TOUS les genres sélectionnés
                # Une vraie approche content-based utiliserait des similarités TF-IDF, etc.
                
                # Pour l'instant, un filtre simple :
                filtered_movies = ALL_MOVIE_FEATURES.copy()
                for genre_sel in selected_genres:
                    filtered_movies = filtered_movies[filtered_movies[C.Constant.GENRES_COL].str.contains(genre_sel, case=False, na=False)]
                
                recommendations_ids = filtered_movies.head(num_recommendations_sidebar).index.tolist()
                
                # Vous devriez remplacer ceci par l'appel à votre fonction content-based
                # ex: recommendations_ids = recommender.get_content_based_recommendations(
                #                                 selected_genres=selected_genres,
                #                                 all_movie_features=ALL_MOVIE_FEATURES,
                #                                 n=num_recommendations_sidebar
                #                             )
                # -------------------------------------------
                display_recommendations(recommendations_ids)
        elif not selected_genres and st.session_state.get('find_by_genre_clicked', False):
             st.warning("Veuillez sélectionner au moins un genre.")
        # Pour gérer l'état du bouton (optionnel)
        if "find_by_genre_clicked" not in st.session_state:
            st.session_state.find_by_genre_clicked = False
        if st.button("Trouver des films par genre (cliqué)", key="hidden_btn_genre", on_click=lambda: st.session_state.update(find_by_genre_clicked=True), type="primary", help="Cliquez pour activer la recherche par genre après sélection"):
             pass # Ce bouton caché ne fait rien visuellement mais met à jour l'état


# =========================== SECTION POUR MOI (USER-BASED / COLLABORATIVE) ===========================
elif menu_choice == "Pour Moi (User-Based)":
    st.header("👤 Recommandations Personnalisées (User-Based)")
    st.markdown("Basées sur votre profil ou celui d'un utilisateur MovieLens.")

    user_profile_options = ["Sélectionnez un profil..."] + list(PROFIL_UTILISATEURS_PERSONNALISES.keys())
    selected_profile_name = st.selectbox(
        "Quel profil souhaitez-vous utiliser ?",
        user_profile_options,
        index=0
    )
    
    movielens_id_input = None
    if selected_profile_name == "Sélectionnez un profil...":
        st.info("Veuillez choisir un profil dans la liste ci-dessus, ou entrez un ID MovieLens ci-dessous.")

    movielens_id_input = st.number_input(
        "Ou entrez un UserID MovieLens (1-610) :", 
        min_value=0, # 0 pour indiquer "pas d'ID MovieLens"
        max_value=610, # Ajustez selon dataset
        value=0, 
        step=1,
        help="Laissez à 0 si vous avez sélectionné un profil personnalisé."
    )

    if st.button("Obtenir mes recommandations User-Based", use_container_width=True):
        recommendations_ids = []
        user_id_to_use = None
        user_name_for_model_files = None
        model_suffix_for_model_files = None

        if selected_profile_name != "Sélectionnez un profil...":
            profile_data = PROFIL_UTILISATEURS_PERSONNALISES[selected_profile_name]
            user_id_to_use = profile_data["user_id"]
            user_name_for_model_files = profile_data["user_name_for_profile"]
            model_suffix_for_model_files = profile_data["model_config_name_suffix"]
            st.write(f"Utilisation du profil personnalisé : {user_name_for_model_files}")
        elif movielens_id_input is not None and movielens_id_input > 0:
            user_id_to_use = movielens_id_input
            st.write(f"Utilisation de l'ID MovieLens : {user_id_to_use} (avec le modèle général)")
        else:
            st.warning("Veuillez sélectionner un profil personnalisé ou entrer un ID MovieLens valide.")
            st.stop()

        if user_id_to_use is not None and ALL_MOVIE_FEATURES is not None:
            with st.spinner(f"Calcul des recommandations pour l'utilisateur {user_id_to_use}..."):
                recommendations_ids = recommender.generate_recommendations_for_user(
                    user_id=user_id_to_use,
                    n=num_recommendations_sidebar,
                    all_movie_ids=ALL_MOVIE_IDS,
                    user_name_for_profile=user_name_for_model_files,
                    model_config_name_suffix=model_suffix_for_model_files,
                    ratings_df_path_for_general_trainset=C.Constant.EVIDENCE_PATH / C.Constant.RATINGS_FILENAME
                )
            display_recommendations(recommendations_ids)

# =========================== SECTION CRÉER MON PROFIL ===========================
elif menu_choice == "Créer Mon Profil":
    st.header("✍️ Créez Votre Profil Personnalisé")
    st.markdown("Notez quelques films pour obtenir des recommandations sur mesure. "
                "Cette fonctionnalité est en cours de développement.")

    # Étape 1: Sélectionner des films à noter
    # Présenter N films (populaires ou aléatoires) que l'utilisateur n'a pas encore vus (si possible)
    num_movies_to_rate = 10 # Consigne du projet
    
    # On prend un échantillon de films pour la notation
    # S'assurer que les films n'ont pas déjà été notés si on a un user_id temporaire.
    # Pour un nouveau user, on peut prendre les plus populaires ou un échantillon aléatoire.
    if ALL_MOVIE_FEATURES is not None and len(ALL_MOVIE_FEATURES) >= num_movies_to_rate:
        sample_movies_to_rate_df = ALL_MOVIE_FEATURES.sample(n=min(num_movies_to_rate * 2, len(ALL_MOVIE_FEATURES)), random_state=42) # *2 pour avoir du choix
        
        st.subheader(f"Veuillez noter au moins {num_movies_to_rate} films suivants :")
        
        # Utiliser st.session_state pour stocker les notes temporaires
        if 'new_user_ratings' not in st.session_state:
            st.session_state.new_user_ratings = {}
        if 'movies_to_rate_ids' not in st.session_state:
             # On stocke les ID pour ne pas les changer à chaque re-render
            st.session_state.movies_to_rate_ids = sample_movies_to_rate_df.head(num_movies_to_rate).index.tolist()

        current_ratings = st.session_state.new_user_ratings
        
        # Afficher les films à noter
        # On pourrait utiliser un formulaire st.form
        with st.form("new_user_rating_form"):
            for movie_id in st.session_state.movies_to_rate_ids:
                movie_title = content.get_movie_title(movie_id, ALL_MOVIE_FEATURES)
                # Note de 0 (pas vu/pas d'avis) à 5, avec des pas de 0.5
                current_ratings[movie_id] = st.slider(
                    f"Votre note pour : **{movie_title}**", 
                    min_value=0.0, max_value=5.0, 
                    value=float(current_ratings.get(movie_id, 0.0)), # Valeur par défaut
                    step=0.5,
                    key=f"rating_{movie_id}"
                )
            
            submitted_ratings_form = st.form_submit_button("J'ai fini de noter !")

        if submitted_ratings_form:
            # Filtrer les films qui ont reçu une note > 0
            final_ratings_list = []
            new_user_implicit_data_list = [] # Pour la fonction de recommender_building
            
            num_rated_movies = 0
            for movie_id, rating_value in current_ratings.items():
                if movie_id in st.session_state.movies_to_rate_ids and rating_value > 0:
                    num_rated_movies += 1
                    final_ratings_list.append({'movieId': movie_id, 'rating': rating_value})
                    # Pour simuler les données pour `calculate_implicit_ratings_from_library`
                    # C'est une simplification, vous devrez adapter la logique d'entrée
                    # de `calculate_implicit_ratings_from_library`.
                    new_user_implicit_data_list.append({
                        C.Constant.MOVIE_ID_COL: movie_id,
                        'watched_count': 1 if rating_value > 0 else 0, # Simplification
                        'last_watched_months_ago': 1, # Simplification
                        'is_favorite': 1 if rating_value >= 4.0 else 0, # Simplification
                        'on_wishlist': 0 # Simplification
                    })

            if num_rated_movies < min(num_movies_to_rate, len(st.session_state.movies_to_rate_ids)): # Au moins N films ou tous si moins de N proposés
                st.warning(f"Veuillez noter au moins {min(num_movies_to_rate, len(st.session_state.movies_to_rate_ids))} films pour continuer.")
            else:
                st.success(f"Merci d'avoir noté {num_rated_movies} films !")
                st.write("Nous allons maintenant générer votre profil de recommandation...")

                # ----- ICI LA LOGIQUE POUR CRÉER LE PROFIL UTILISATEUR "À LA VOLÉE" -----
                # 1. Créer un DataFrame à partir de new_user_implicit_data_list
                # 2. Attribuer un nouvel ID utilisateur (ex: max_id + 1, ou un ID négatif unique)
                # 3. Appeler les fonctions de `recommender_building.py` pour :
                #    a. Calculer les ratings implicites (ou utiliser directement les notes explicites ici)
                #    b. Augmenter le dataset MovieLens
                #    c. Entraîner un modèle pour ce nouvel utilisateur
                #    d. Sauvegarder ce modèle
                # 4. Gérer le délai d'entraînement (très important !) - cf. options discutées.
                #    Pour l'instant, simulation d'un délai et affichage des films notés.

                with st.spinner("Création de votre profil et entraînement du modèle... Cela peut prendre un moment."):
                    # --- DÉBUT DE LA PARTIE COMPLEXE : APPEL À recommender_building ---
                    # Ceci est une esquisse. L'intégration réelle est plus complexe.
                    
                    # Exemple: Créer un nom et un ID pour ce nouveau profil
                    # Vous auriez besoin d'une meilleure gestion des ID et des noms
                    timestamp_id = int(time.time()) # Pour un ID unique
                    new_profile_name = f"UserDynamic_{timestamp_id}"
                    new_user_numeric_id = -timestamp_id # Un ID numérique négatif

                    # Convertir les notes en DataFrame pour la fonction de `recommender_building`
                    new_user_library_df = pd.DataFrame(new_user_implicit_data_list)

                    st.markdown("#### Simulation de la création du profil (étape non implémentée en 'live')")
                    st.markdown(f"Nom du profil (simulé): `{new_profile_name}` (ID: `{new_user_numeric_id}`)")
                    st.markdown("Les films que vous avez notés (rating > 0) :")
                    for item in final_ratings_list:
                        st.write(f"- {content.get_movie_title(item['movieId'], ALL_MOVIE_FEATURES)}: {item['rating']}/5.0")
                    
                    st.warning("La création de profil 'live' avec ré-entraînement du modèle n'est pas entièrement implémentée "
                               "dans cet exemple en raison de sa complexité et des délais potentiels. "
                               "Normalement, ici, on appellerait `recommender_building.main_recommender_building_for_user(...)` "
                               "et on gérerait l'attente ou un processus en tâche de fond.")
                    
                    # Si vous aviez une version "rapide" ou une autre logique:
                    # recommendations_ids_new_user = recommender.generate_recommendations_for_new_user_profile(
                    # new_user_id=new_user_numeric_id,
                    # user_library_df=new_user_library_df,
                    # ...
                    # )
                    # display_recommendations(recommendations_ids_new_user)
                    # 
                    # Ajouter ce nouveau profil à PROFIL_UTILISATEURS_PERSONNALISES (pour la session)
                    # if 'session_profiles' not in st.session_state:
                    #    st.session_state.session_profiles = PROFIL_UTILISATEURS_PERSONNALISES.copy()
                    # st.session_state.session_profiles[new_profile_name] = {
                    # "user_id": new_user_numeric_id,
                    # "user_name_for_profile": new_profile_name, # ou une partie
                    # "model_config_name_suffix": "svd_implicit" # ou le type de modèle entraîné
                    # }
                    # st.success(f"Profil {new_profile_name} créé ! Vous pouvez maintenant le sélectionner dans 'Pour Moi (User-Based)'.")
                    # --- FIN DE LA PARTIE COMPLEXE ---
                    
    else:
        st.info("Les données des films ne sont pas disponibles pour la notation.")

# --- Pied de Page ---
st.sidebar.markdown("---")
st.sidebar.info("© 2025 - Votre Nom/Groupe - Projet MLSMM2156")