import pandas as pd
import streamlit as st

# Vos modules existants
import loaders
import constants as C # S'assurer que ITEM_ID_COL, LABEL_COL, GENRES_COL sont définis

@st.cache_data # Mise en cache des données des films
def load_movie_data():
    """Charge et retourne le DataFrame des items (films)."""
    try:
        # loaders.load_items() devrait retourner un DataFrame avec ITEM_ID_COL comme index
        df_items = loaders.load_items() 
        if not isinstance(df_items.index.name, str) or df_items.index.name.lower() != C.Constant.ITEM_ID_COL.lower():
            # Si l'index n'est pas ITEM_ID_COL, ou si ce n'est pas le nom de l'index
            # (loaders.load_items() le met en index)
            # Ceci est une vérification, votre loaders.load_items() doit déjà bien le faire
            if C.Constant.ITEM_ID_COL in df_items.columns:
                 df_items = df_items.set_index(C.Constant.ITEM_ID_COL)
            else:
                st.error(f"La colonne ID des items '{C.Constant.ITEM_ID_COL}' est introuvable pour l'indexation.")
                return pd.DataFrame() # Retourner un DF vide en cas d'erreur
        return df_items
    except Exception as e:
        st.error(f"Erreur lors du chargement des données des films: {e}")
        return pd.DataFrame()

DF_MOVIES = load_movie_data()

def get_movie_details(movie_id):
    """
    Retourne un dictionnaire avec les détails d'un film (titre, genres).
    Retourne None si le film n'est pas trouvé.
    """
    if DF_MOVIES.empty:
        return None
    try:
        # movie_id est attendu comme étant l'index de DF_MOVIES
        details = DF_MOVIES.loc[movie_id]
        return {
            'title': details[C.Constant.LABEL_COL],
            'genres': details[C.Constant.GENRES_COL]
            # Ajoutez d'autres champs si nécessaire
        }
    except KeyError:
        return None
    except Exception as e:
        # Log l'erreur si nécessaire pour le débogage
        print(f"Erreur dans get_movie_details pour movie_id {movie_id}: {e}")
        return None


def get_all_movie_titles_ids():
    """
    Retourne un dictionnaire de {movie_id: title} pour tous les films.
    """
    if DF_MOVIES.empty:
        return {}
    return DF_MOVIES[C.Constant.LABEL_COL].to_dict()

if __name__ == '__main__':
    # Petit test
    print("Chargement des données films...")
    if not DF_MOVIES.empty:
        print(f"{len(DF_MOVIES)} films chargés.")
        test_movie_id = DF_MOVIES.index[0] # Prendre le premier movieId pour tester
        details = get_movie_details(test_movie_id)
        if details:
            print(f"\nDétails pour le film ID {test_movie_id}: {details['title']} - {details['genres']}")
        
        all_titles = get_all_movie_titles_ids()
        if all_titles:
            print(f"\nNombre total de titres récupérés: {len(all_titles)}")
    else:
        print("Aucune donnée de film n'a pu être chargée.")