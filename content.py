import pandas as pd
from pathlib import Path
import streamlit as st # Pour @st.cache_data

# Votre module constants
import constants as C

# --- Fonctions de Chargement des Données Individuelles ---
# Ces fonctions pourraient aussi résider dans loaders.py et être importées ici.

@st.cache_data # Optimisation pour Streamlit: ne recharge pas si les données n'ont pas changé
def load_movies_df():
    """Charge le fichier movies.csv."""
    movies_file_path = C.Constant.CONTENT_PATH / C.Constant.ITEMS_FILENAME
    try:
        df = pd.read_csv(movies_file_path)
        # S'assurer que MOVIE_ID_COL est du bon type (int)
        df[C.Constant.MOVIE_ID_COL] = df[C.Constant.MOVIE_ID_COL].astype(int)
        return df
    except FileNotFoundError:
        st.error(f"Fichier movies non trouvé à l'emplacement : {movies_file_path}")
        return pd.DataFrame(columns=[C.Constant.MOVIE_ID_COL, C.Constant.LABEL_COL, C.Constant.GENRES_COL])
    except Exception as e:
        st.error(f"Erreur lors du chargement de {movies_file_path}: {e}")
        return pd.DataFrame(columns=[C.Constant.MOVIE_ID_COL, C.Constant.LABEL_COL, C.Constant.GENRES_COL])

@st.cache_data
def load_links_df():
    """Charge le fichier links.csv."""
    # Supposition: links.csv est dans CONTENT_PATH et s'appelle 'links.csv'
    # Vous devriez ajouter LINK_FILENAME = 'links.csv' à constants.py
    links_filename = "links.csv" # À ajouter à constants.py si ce n'est pas déjà fait
    links_file_path = C.Constant.CONTENT_PATH / links_filename
    try:
        df = pd.read_csv(links_file_path)
        # S'assurer que MOVIE_ID_COL est du bon type (int)
        df[C.Constant.MOVIE_ID_COL] = df[C.Constant.MOVIE_ID_COL].astype(int)
        # tmdbId peut avoir des NaN, on les garde pour l'instant
        return df
    except FileNotFoundError:
        st.error(f"Fichier links non trouvé à l'emplacement : {links_file_path}")
        return pd.DataFrame(columns=[C.Constant.MOVIE_ID_COL, 'imdbId', 'tmdbId'])
    except Exception as e:
        st.error(f"Erreur lors du chargement de {links_file_path}: {e}")
        return pd.DataFrame(columns=[C.Constant.MOVIE_ID_COL, 'imdbId', 'tmdbId'])

@st.cache_data
def load_tags_df():
    """Charge et agrège le fichier tags.csv."""
    # Supposition: tags.csv est dans CONTENT_PATH et s'appelle 'tags.csv'
    # Vous devriez ajouter TAGS_FILENAME = 'tags.csv' à constants.py
    tags_filename = "tags.csv" # À ajouter à constants.py
    tags_file_path = C.Constant.CONTENT_PATH / tags_filename
    try:
        df = pd.read_csv(tags_file_path)
        # S'assurer que MOVIE_ID_COL est du bon type (int)
        df[C.Constant.MOVIE_ID_COL] = df[C.Constant.MOVIE_ID_COL].astype(int)
        # Agréger les tags par movieId
        # On enlève userId et timestamp qui ne sont pas utiles pour les features du film
        # On groupe les tags dans une liste ou une chaîne de caractères par film
        tags_agg = df.groupby(C.Constant.MOVIE_ID_COL)['tag'].apply(lambda x: '|'.join(x.astype(str))).reset_index()
        tags_agg.rename(columns={'tag': 'aggregated_tags'}, inplace=True)
        return tags_agg
    except FileNotFoundError:
        st.error(f"Fichier tags non trouvé à l'emplacement : {tags_file_path}")
        return pd.DataFrame(columns=[C.Constant.MOVIE_ID_COL, 'aggregated_tags'])
    except Exception as e:
        st.error(f"Erreur lors du chargement de {tags_file_path}: {e}")
        return pd.DataFrame(columns=[C.Constant.MOVIE_ID_COL, 'aggregated_tags'])

@st.cache_data
def load_tmdb_features_df():
    """Charge le fichier tmdb_full_features.csv."""
    # Supposition: tmdb_full_features.csv est dans CONTENT_PATH et s'appelle 'tmdb_full_features.csv'
    # Vous devriez ajouter TMDB_FEATURES_FILENAME = 'tmdb_full_features.csv' à constants.py
    tmdb_features_filename = "tmdb_full_features.csv" # À ajouter à constants.py
    tmdb_file_path = C.Constant.CONTENT_PATH / tmdb_features_filename
    try:
        df = pd.read_csv(tmdb_file_path)
        # S'assurer que MOVIE_ID_COL est présent et du bon type (int)
        if C.Constant.MOVIE_ID_COL not in df.columns:
            st.warning(f"La colonne '{C.Constant.MOVIE_ID_COL}' n'est pas présente dans {tmdb_features_filename}. "
                       f"Les features TMDB ne pourront pas être directement jointes par movieId.")
            # Si 'movieId' n'est pas là, mais 'id' (tmdbId) l'est, il faudra joindre via links.csv
            # Pour l'instant, on retourne le df tel quel ou un df vide si movieId est crucial ici.
            # return pd.DataFrame() ou gérer une jointure alternative plus tard.
        else:
            df[C.Constant.MOVIE_ID_COL] = df[C.Constant.MOVIE_ID_COL].astype(int)
        return df
    except FileNotFoundError:
        st.error(f"Fichier TMDB features non trouvé à l'emplacement : {tmdb_file_path}")
        return pd.DataFrame() # Retourne un DataFrame vide en cas d'erreur
    except Exception as e:
        st.error(f"Erreur lors du chargement de {tmdb_file_path}: {e}")
        return pd.DataFrame()


# --- Fonction Principale de Fusion des Features ---

@st.cache_data
def get_all_movie_features():
    """
    Charge toutes les données des films et les fusionne en un seul DataFrame.
    """
    movies_df = load_movies_df()
    links_df = load_links_df()
    tags_df = load_tags_df()
    tmdb_df = load_tmdb_features_df()

    if movies_df.empty:
        st.warning("Le DataFrame movies est vide, impossible de fusionner les features.")
        return pd.DataFrame()

    # 1. Fusionner movies_df avec links_df
    # Utiliser C.Constant.MOVIE_ID_COL comme clé
    all_features_df = pd.merge(movies_df, links_df, on=C.Constant.MOVIE_ID_COL, how='left')

    # 2. Fusionner avec tags_df (agrégés)
    if not tags_df.empty:
        all_features_df = pd.merge(all_features_df, tags_df, on=C.Constant.MOVIE_ID_COL, how='left')
    else:
        all_features_df['aggregated_tags'] = None # Ajouter colonne vide si pas de tags

    # 3. Fusionner avec tmdb_df
    # tmdb_full_features.csv a déjà une colonne 'movieId'.
    if not tmdb_df.empty and C.Constant.MOVIE_ID_COL in tmdb_df.columns:
        # S'il y a des colonnes en commun (autres que movieId), elles auront des suffixes _x, _y
        # On peut vouloir ne garder que les colonnes pertinentes de tmdb_df ou les renommer avant
        # Par exemple, tmdb_df peut avoir sa propre colonne 'title', 'genres', etc.
        # Pour éviter les conflits, on peut suffixer les colonnes de tmdb_df (sauf movieId)
        tmdb_cols_to_merge = tmdb_df.columns.difference([C.Constant.MOVIE_ID_COL])
        all_features_df = pd.merge(
            all_features_df,
            tmdb_df[[C.Constant.MOVIE_ID_COL] + tmdb_cols_to_merge.tolist()],
            on=C.Constant.MOVIE_ID_COL,
            how='left',
            suffixes=('', '_tmdb') # Ajoute _tmdb aux colonnes dupliquées venant de tmdb_df
        )
    elif not tmdb_df.empty and 'id' in tmdb_df.columns and 'tmdbId' in all_features_df.columns:
        # Cas où tmdb_df doit être joint sur tmdbId (si movieId n'y était pas)
        st.info("Tentative de jointure des features TMDB via tmdbId.")
        tmdb_df_renamed = tmdb_df.rename(columns={'id': 'tmdbId'}) # 'id' dans tmdb_df est le tmdbId
        tmdb_cols_to_merge = tmdb_df_renamed.columns.difference(['tmdbId'])
        all_features_df = pd.merge(
            all_features_df,
            tmdb_df_renamed[['tmdbId'] + tmdb_cols_to_merge.tolist()],
            on='tmdbId',
            how='left',
            suffixes=('', '_tmdb')
        )

    # Mettre movieId comme index pour un accès plus facile avec .loc
    if C.Constant.MOVIE_ID_COL in all_features_df.columns:
         all_features_df = all_features_df.set_index(C.Constant.MOVIE_ID_COL)
    
    return all_features_df

# --- Fonctions d'Accès aux Données ---

def get_movie_details(movie_id, all_features_df):
    """
    Retourne les détails d'un film spécifique à partir du DataFrame fusionné.
    """
    if all_features_df.empty or movie_id not in all_features_df.index:
        return None
    return all_features_df.loc[movie_id]

def get_movie_title(movie_id, all_features_df):
    """Retourne le titre d'un film."""
    details = get_movie_details(movie_id, all_features_df)
    if details is not None and C.Constant.LABEL_COL in details: # LABEL_COL est 'title'
        return details[C.Constant.LABEL_COL]
    return "Titre inconnu"

def get_movie_poster_url(movie_id, all_features_df, tmdb_api_key="VOTRE_CLE_API_TMDB_ICI"):
    """
    Construit l'URL de l'affiche d'un film en utilisant son tmdbId.
    Nécessite une clé API TMDB.
    """
    if tmdb_api_key == "VOTRE_CLE_API_TMDB_ICI":
        # st.warning("Clé API TMDB non configurée. Impossible de récupérer les affiches.")
        return None # Ou une URL d'affiche par défaut

    details = get_movie_details(movie_id, all_features_df)
    if details is not None:
        tmdb_id = details.get('tmdbId', None) # tmdbId vient de links_df
        poster_path_tmdb = details.get('poster_path', None) # poster_path vient de tmdb_full_features

        if pd.notna(poster_path_tmdb): # Utiliser directement poster_path si disponible
            return f"https://image.tmdb.org/t/p/w500{poster_path_tmdb}"
        elif pd.notna(tmdb_id) and tmdb_api_key != "VOTRE_CLE_API_TMDB_ICI":
            # Si poster_path n'est pas dans les features directes mais qu'on a tmdbId,
            # on pourrait faire un appel API ici (plus complexe et non idéal dans content.py)
            # Pour l'instant, on suppose que 'poster_path' est dans tmdb_full_features.csv
            # Si vous devez faire un appel API dynamique:
            # import requests
            # try:
            #     response = requests.get(f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={tmdb_api_key}")
            #     response.raise_for_status()
            #     movie_api_details = response.json()
            #     if movie_api_details.get('poster_path'):
            #         return f"https://image.tmdb.org/t/p/w500{movie_api_details['poster_path']}"
            # except requests.RequestException as e:
            #     st.error(f"Erreur API TMDB pour tmdbId {tmdb_id}: {e}")
            #     return None
            # except ValueError: # Si tmdb_id n'est pas un int valide
            #     st.error(f"tmdbId invalide pour l'appel API : {tmdb_id}")
            #     return None
            pass # Laissez vide si tmdb_full_features.csv contient déjà poster_path

    return None # URL d'une affiche par défaut si rien n'est trouvé

# --- Exemple d'utilisation (peut être testé en exécutant ce script directement) ---
if __name__ == '__main__':
    st.set_page_config(layout="wide")
    st.title("Test du Module Content")

    # Charger toutes les features
    # Pour les tests en dehors de Streamlit, @st.cache_data n'a pas d'effet bloquant
    # mais les messages st.error/warning s'afficheront dans la console si Streamlit n'est pas l'exécuteur.
    
    # Pour que les chemins relatifs dans constants.py fonctionnent correctement lors de l'exécution
    # directe de ce script, assurez-vous que votre répertoire de travail est la racine du projet.
    # (Par exemple, si constants.py a Path('data/small'), 'data' doit être au même niveau que là où vous exécutez).

    # Si vous voulez tester sans Streamlit, commentez les @st.cache_data et les appels st.error/st.info
    # ou encapsulez les appels st. dans des conditions `if st._is_running_with_streamlit:`

    st.header("Chargement de toutes les features des films...")
    all_features = get_all_movie_features()

    if not all_features.empty:
        st.write(f"Nombre total de films avec features fusionnées : {len(all_features)}")
        st.write("Premières lignes du DataFrame fusionné :")
        st.dataframe(all_features.head())

        st.subheader("Détails d'un film spécifique (exemple movieId = 1)")
        movie_id_test = 1
        if movie_id_test in all_features.index:
            details = get_movie_details(movie_id_test, all_features)
            st.write(details)

            title = get_movie_title(movie_id_test, all_features)
            st.write(f"Titre récupéré : {title}")

            poster_url = get_movie_poster_url(movie_id_test, all_features)
            if poster_url:
                st.image(poster_url, caption=f"Affiche de {title}")
            else:
                st.write("URL de l'affiche non disponible (vérifiez la clé API TMDB ou la présence de 'poster_path').")
            
            st.subheader("Exemple movieId = 2 (si existe)")
            movie_id_test_2 = 2
            if movie_id_test_2 in all_features.index:
                details_2 = get_movie_details(movie_id_test_2, all_features)
                st.write(details_2)
                title_2 = get_movie_title(movie_id_test_2, all_features)
                poster_url_2 = get_movie_poster_url(movie_id_test_2, all_features)
                if poster_url_2:
                    st.image(poster_url_2, caption=f"Affiche de {title_2}")

        else:
            st.warning(f"Le film avec movieId = {movie_id_test} n'a pas été trouvé dans les features fusionnées.")
        
        st.subheader("Colonnes disponibles dans le DataFrame fusionné:")
        st.write(all_features.columns.tolist())
    else:
        st.error("Aucune feature n'a pu être chargée ou fusionnée.")