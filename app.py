import streamlit as st
import pandas as pd

# Vos modules
import recommender # Le backend logique
import content     # Pour les détails des films
import constants as C # Pour PERSONAL_USER_ID notamment

def display_recommendations_in_app(recommendations_data):
    """Affiche les recommandations formatées dans Streamlit."""
    st.subheader("🏆 Vos Films Recommandés")
    if not recommendations_data:
        st.info("Aucune recommandation n'a pu être générée pour le moment. " \
                "Assurez-vous d'avoir exécuté `recommender_building.py` et que les modèles sont disponibles.")
        return

    for i, (movie_id, score, explanation) in enumerate(recommendations_data):
        movie_details = content.get_movie_details(movie_id) # Utilise content.py
        if movie_details:
            st.markdown(f"---") # Séparateur
            cols = st.columns([1, 4]) # Colonne pour image/placeholder, colonne pour texte
            
            with cols[0]:
                # Vous pourriez ajouter une image ici si vous avez les URLs des posters
                st.markdown(f"### #{i+1}")

            with cols[1]:
                st.markdown(f"**{movie_details['title']}**")
                st.caption(f"Genres: {movie_details['genres'].replace('|', ', ')}")
                st.markdown(f"*Score estimé : {score:.2f}*")
            
            with st.expander("💡 Pourquoi ce film ?"):
                st.markdown(explanation)
        else:
            st.warning(f"Détails non trouvés pour le film ID: {movie_id}")


def main():
    st.set_page_config(page_title="Recommandations de Films Perso", layout="wide", initial_sidebar_state="expanded")
    st.title("🎬 Recommandations de Films Personnalisées")
    st.markdown("Bienvenue ! Ce système utilise vos préférences implicites pour vous suggérer des films.")

    # --- Sélection de l'utilisateur et du modèle dans la sidebar ---
    st.sidebar.header("👤 Votre Profil")
    # Pour ce projet, on se concentre sur l'utilisateur personnalisé
    # La fonction get_available_user_ids_for_app() devrait retourner [str(C.Constant.PERSONAL_USER_ID)]
    available_users = recommender.get_available_user_ids_for_app()
    
    if not available_users:
        st.sidebar.error("Aucun profil utilisateur personnalisé n'a été trouvé. " \
                         "Veuillez exécuter `recommender_building.py`.")
        st.stop()

    # Par défaut, on sélectionne le premier (et unique) utilisateur personnalisé
    selected_user_id_str = st.sidebar.selectbox(
        "Choisissez votre profil utilisateur:",
        available_users,
        index=0,
        help="Ce profil a été enrichi avec votre bibliothèque de films personnels."
    )
    
    # Convertir l'ID en entier si PERSONAL_USER_ID est un entier. Soyez cohérent.
    try:
        selected_user_id_for_model = int(selected_user_id_str)
    except ValueError:
        st.error("L'ID utilisateur sélectionné n'est pas valide.")
        st.stop()


    st.sidebar.header("⚙️ Type de Recommandation")
    model_type_display_names = {
        "SVD": "Analyse Détaillée (SVD)",
        "UserBased": "Selon Goûts Similaires (User-Based)",
        "ContentBased": "Basé sur le Contenu (Content-Based)"
    }
    # Clés techniques pour charger les modèles
    model_technical_names = list(model_type_display_names.keys()) 
    
    selected_model_technical_name = st.sidebar.radio(
        "Méthode de recommandation :",
        options=model_technical_names,
        format_func=lambda tech_name: model_type_display_names[tech_name], # Noms affichés à l'utilisateur
        index=0
    )

    num_recs = st.sidebar.slider("Nombre de films à recommander :", min_value=3, max_value=20, value=10, step=1)

    # --- Bouton pour obtenir les recommandations ---
    if st.sidebar.button("🚀 Me Recommander des Films !", type="primary", use_container_width=True):
        st.markdown("---")
        with st.spinner(f"Recherche des pépites pour vous avec le modèle '{model_type_display_names[selected_model_technical_name]}' 🤔..."):
            # Charger le modèle et le trainset associés à l'utilisateur et au type de modèle
            loaded_model, loaded_trainset = recommender.load_model_and_trainset(
                selected_user_id_for_model, 
                selected_model_technical_name # Utiliser le nom technique
            )

            if loaded_model and loaded_trainset:
                recommendations = recommender.get_top_n_recommendations(
                    loaded_model,
                    loaded_trainset,
                    selected_user_id_for_model, # L'ID brut pour la prédiction (doit correspondre au type dans le trainset)
                    selected_model_technical_name, # Le nom technique du modèle
                    n=num_recs
                )
                display_recommendations_in_app(recommendations)
            else:
                st.error("Impossible de charger le modèle sélectionné. " \
                         "Assurez-vous que `recommender_building.py` a été exécuté correctement.")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"Profil actuel : **Utilisateur {selected_user_id_str}**")
    st.sidebar.info("Ce projet utilise le dataset MovieLens et vos évaluations implicites.")

if __name__ == '__main__':
    # Étapes avant de lancer app.py :
    # 1. Vérifiez/complétez constants.py avec les chemins nécessaires.
    # 2. Créez votre fichier library_votrenom.csv.
    # 3. Exécutez `python recommender_building.py` pour entraîner et sauvegarder vos modèles personnalisés.
    # 4. Ensuite, lancez cette application avec `streamlit run app.py`.
    main()