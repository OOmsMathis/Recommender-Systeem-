# loaders.py

import pandas as pd
import re
from pathlib import Path
import constants as C_module
C = C_module.Constant() # Instancier la classe Constant
import ast 
import requests
import os
import zipfile
import streamlit as st


def download_csvs_from_drive(file_urls, base_local_dir):
    """
    Télécharge une série de fichiers CSV depuis des URLs (Google Drive ou autre)
    et les stocke dans un répertoire local en respectant la structure de base_local_dir.
    file_urls: dict {relative_path (str): url (str)}
    base_local_dir: Path ou str, racine locale (ex: Path('mlsmm2156/data'))
    """
    base_local_dir = Path(base_local_dir)
    for rel_path, url in file_urls.items():
        local_path = base_local_dir / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(local_path, 'wb') as f:
                f.write(response.content)
            print(f"Téléchargé: {rel_path}")
        except Exception as e:
            print(f"Erreur lors du téléchargement de {url} -> {rel_path}: {e}")



# --- Fonctions Helpers ---
def parse_literal_eval_column(series):
    """Helper function to safely parse stringified lists/dicts in a column."""
    def safe_eval(x):
        if isinstance(x, str) and x.startswith(('[', '{')):
            try:
                return ast.literal_eval(x)
            except (ValueError, SyntaxError):
                return [] 
        elif pd.isna(x):
            return [] 
        return x
    return series.apply(safe_eval)

def extract_names(data_list, key='name', max_items=5):
    """Helper function to extract 'name' from a list of dicts."""
    if not isinstance(data_list, list):
        return '' 
    names = [item[key] for item in data_list if isinstance(item, dict) and key in item]
    return ', '.join(names[:max_items])

def clean_genre_string(genre_str):
    """
    Nettoie une chaîne de genres.
    Ex: " Action ,Adventure | |  Sci-Fi " -> "Action|Adventure|Sci-Fi"
    """
    if pd.isna(genre_str) or not isinstance(genre_str, str):
        return '' 
    cleaned_genres = re.sub(r'\s*[,;/]\s*', '|', genre_str)
    genres_list = [g.strip() for g in cleaned_genres.split('|') if g.strip()]
    return '|'.join(genres_list)

def reformat_movie_title_core(title_str_no_year):
    """
    Reformate les titres de films (sans année) pour déplacer les articles de la fin vers le début.
    Ex: "Lion King, The" -> "The Lion King"
    Ex: "Léon: The Professional" -> "Léon: The Professional" (ne change pas)
    Ex: "Beautiful Mind, A" -> "A Beautiful Mind"
    Ex: "Legend of 1900, The (a.k.a. The Legend of the...)" -> "The Legend of 1900 (a.k.a. The Legend of the...)"
    """
    if pd.isna(title_str_no_year) or not isinstance(title_str_no_year, str):
        return title_str_no_year

    title_stripped = title_str_no_year.strip()
    
    # Regex modifiée pour "Titre, L'Article SuffixeOptionnel"
    # Groupe 1: (.*?) - Le titre principal (non-gourmand)
    # Groupe 2: (The|A|An|L'|Le|La|Les) - L'article
    # Groupe 3: (.*) - Tout ce qui suit l'article (y compris un espace initial s'il existe, ou une parenthèse)
    match = re.match(r'^(.*?),\s*(The|A|An|L\'|Le|La|Les)(.*)$', title_stripped)
    
    if match:
        main_title_part = match.group(1).strip()  # Titre avant la virgule et l'article
        article = match.group(2).strip()          # L'article lui-même
        suffix = match.group(3)                   # Le reste de la chaîne après l'article
                                                  # suffix peut être vide, ou commencer par " (quelque chose)"
                                                  # ou " quelque chose d'autre"

        # Reconstruit le titre avec l'article au début, suivi du titre principal, puis du suffixe.
        # Le .strip() final nettoie les espaces superflus au début ou à la fin.
        return f"{article} {main_title_part}{suffix}".strip()
        
    return title_stripped 



# --- Fonctions de Chargement ---
@st.cache_data
def load_ratings():
    ratings_filepath = C.EVIDENCE_PATH / C.RATINGS_FILENAME
    try:
        df_ratings = pd.read_csv(ratings_filepath)
        expected_cols = [C.USER_ID_COL, C.ITEM_ID_COL, C.RATING_COL]
        if not all(col in df_ratings.columns for col in expected_cols):
            print(f"AVERTISSEMENT (load_ratings): Colonnes attendues {expected_cols} non toutes trouvées dans {ratings_filepath}. Colonnes présentes: {df_ratings.columns.tolist()}")
        return df_ratings
    except FileNotFoundError:
        print(f"ERREUR (load_ratings): Fichier ratings non trouvé à '{ratings_filepath}'.")
        raise
    except Exception as e:
        print(f"ERREUR (load_ratings): Erreur inattendue lors du chargement de {ratings_filepath}: {e}")
        raise

@st.cache_data
def load_items():
    movies_filepath = C.CONTENT_PATH / C.ITEMS_FILENAME
    tmdb_filepath = C.CONTENT_PATH / C.TMDB_FILENAME
    links_filepath = C.CONTENT_PATH / C.LINKS_FILENAME

    print("load_items: Début du chargement des items...")
    try:
        df_movies = pd.read_csv(movies_filepath)

        if C.ITEM_ID_COL not in df_movies.columns:
            raise KeyError(f"Colonne '{C.ITEM_ID_COL}' non trouvée dans {movies_filepath}.")
        
        if C.LABEL_COL in df_movies.columns:
            print(f"Traitement des titres dans la colonne '{C.LABEL_COL}'...")
            # 1. Extraire l'année et la stocker
            df_movies[C.RELEASE_YEAR_COL] = df_movies[C.LABEL_COL].str.extract(r'\((\d{4})\)\s*$', expand=False)
            df_movies[C.RELEASE_YEAR_COL] = pd.to_numeric(df_movies[C.RELEASE_YEAR_COL], errors='coerce').astype('Int64')
            # print("  Année extraite.")

            # 2. Supprimer l'année de la chaîne de titre originale
            df_movies[C.LABEL_COL] = df_movies[C.LABEL_COL].str.replace(r'\s*\(\d{4}\)\s*$', '', regex=True).str.strip()
            # print("  Année supprimée de la chaîne de titre.")

            # 3. Reformater le titre (déplacer l'article) sur le titre sans année
            df_movies[C.LABEL_COL] = df_movies[C.LABEL_COL].apply(reformat_movie_title_core)
            # print(f"  Articles déplacés au début des titres.")
        else:
            print(f"AVERTISSEMENT (load_items): Colonne titre '{C.LABEL_COL}' non trouvée. Année et titres non traités.")
            df_movies[C.RELEASE_YEAR_COL] = pd.NA 

        df_items_rich = df_movies.copy()

        if C.GENRES_COL in df_items_rich.columns:
            df_items_rich[C.GENRES_COL] = df_items_rich[C.GENRES_COL].apply(clean_genre_string)

        try:
            df_tmdb = pd.read_csv(tmdb_filepath, low_memory=False)
            if 'id' in df_tmdb.columns and C.ITEM_ID_COL not in df_tmdb.columns:
                df_tmdb = df_tmdb.rename(columns={'id': C.ITEM_ID_COL})
            
            if C.ITEM_ID_COL not in df_tmdb.columns:
                print(f"AVERTISSEMENT (load_items): Colonne '{C.ITEM_ID_COL}' non trouvée dans {tmdb_filepath}. TMDB non fusionné.")
            else: 
                tmdb_cols_to_select = [C.ITEM_ID_COL]
                defined_tmdb_constants_attributes = [
                    'GENRES_COL', 'RUNTIME_COL', 'CAST_COL', 'DIRECTORS_COL',
                    'VOTE_COUNT_COL', 'VOTE_AVERAGE_COL', 
                    'POPULARITY_COL', 'BUDGET_COL', 'REVENUE_COL', 'ORIGINAL_LANGUAGE_COL',
                ]
                
                tmdb_title_col_name = 'title' # Nom commun pour le titre dans les datasets TMDB
                tmdb_release_date_col_name = 'release_date' # Nom commun pour la date de sortie TMDB

                if tmdb_title_col_name in df_tmdb.columns and tmdb_title_col_name not in tmdb_cols_to_select:
                    tmdb_cols_to_select.append(tmdb_title_col_name)
                if tmdb_release_date_col_name in df_tmdb.columns and tmdb_release_date_col_name not in tmdb_cols_to_select:
                     tmdb_cols_to_select.append(tmdb_release_date_col_name)


                for attr_name in defined_tmdb_constants_attributes:
                    if hasattr(C, attr_name): 
                        col_name = getattr(C, attr_name) 
                        if col_name in df_tmdb.columns and col_name not in tmdb_cols_to_select:
                            tmdb_cols_to_select.append(col_name)
                
                df_tmdb_subset = df_tmdb[tmdb_cols_to_select].copy()

                # Traitement des titres TMDB (supprimer année, reformater article)
                processed_tmdb_titles = pd.Series(index=df_tmdb_subset.index, dtype=str)
                if tmdb_title_col_name in df_tmdb_subset.columns:
                    # Supprimer l'année si elle est entre parenthèses à la fin du titre TMDB
                    processed_tmdb_titles = df_tmdb_subset[tmdb_title_col_name].str.replace(r'\s*\(\d{4}\)\s*$', '', regex=True).str.strip()
                    processed_tmdb_titles = processed_tmdb_titles.apply(reformat_movie_title_core)
                    df_tmdb_subset['processed_tmdb_title'] = processed_tmdb_titles # Nouvelle colonne pour le titre TMDB traité

                # Extraire l'année de la date de sortie TMDB si disponible et si RELEASE_YEAR_COL n'est pas déjà prioritaire
                if tmdb_release_date_col_name in df_tmdb_subset.columns:
                    df_tmdb_subset['tmdb_year'] = pd.to_datetime(df_tmdb_subset[tmdb_release_date_col_name], errors='coerce').dt.year.astype('Int64')


                if C.GENRES_COL in df_tmdb_subset.columns:
                    def parse_tmdb_genres(genre_json_str):
                        if pd.isna(genre_json_str) or not isinstance(genre_json_str, str): return ''
                        try:
                            genre_list = ast.literal_eval(genre_json_str)
                            if isinstance(genre_list, list):
                                names = [item['name'] for item in genre_list if isinstance(item, dict) and 'name' in item]
                                return '|'.join(sorted(list(set(names))))
                            return ''
                        except: return '' 
                    df_tmdb_subset[C.GENRES_COL] = df_tmdb_subset[C.GENRES_COL].apply(parse_tmdb_genres)

                if hasattr(C, 'CAST_COL') and C.CAST_COL in df_tmdb_subset.columns:
                    df_tmdb_subset[C.CAST_COL] = parse_literal_eval_column(df_tmdb_subset[C.CAST_COL])
                    if hasattr(C, 'TMDB_CAST_NAMES_COL'):
                         df_tmdb_subset[C.TMDB_CAST_NAMES_COL] = df_tmdb_subset[C.CAST_COL].apply(lambda x: extract_names(x, key='name', max_items=5))

                if hasattr(C, 'DIRECTORS_COL') and C.DIRECTORS_COL in df_tmdb_subset.columns:
                    df_tmdb_subset[C.DIRECTORS_COL] = parse_literal_eval_column(df_tmdb_subset[C.DIRECTORS_COL])
                
                # Fusion
                df_items_rich = pd.merge(
                    df_items_rich, 
                    df_tmdb_subset.drop(columns=[tmdb_title_col_name, tmdb_release_date_col_name], errors='ignore'), # Drop originaux TMDB si traités
                    on=C.ITEM_ID_COL, 
                    how='left',
                    suffixes=('_ml', '_tmdb') # Suffixes pour les colonnes restantes du merge
                )

                # Prioriser le titre TMDB traité s'il existe et est non vide
                if 'processed_tmdb_title' in df_items_rich.columns:
                    mask_tmdb_title_valid = df_items_rich['processed_tmdb_title'].notna() & (df_items_rich['processed_tmdb_title'].str.strip() != '')
                    df_items_rich.loc[mask_tmdb_title_valid, C.LABEL_COL] = df_items_rich.loc[mask_tmdb_title_valid, 'processed_tmdb_title']
                    df_items_rich.drop(columns=['processed_tmdb_title'], inplace=True, errors='ignore')

                # Prioriser l'année TMDB si la colonne RELEASE_YEAR_COL est vide et tmdb_year est disponible
                if 'tmdb_year' in df_items_rich.columns and C.RELEASE_YEAR_COL in df_items_rich.columns:
                    mask_ml_year_missing = df_items_rich[C.RELEASE_YEAR_COL].isna()
                    mask_tmdb_year_present = df_items_rich['tmdb_year'].notna()
                    df_items_rich.loc[mask_ml_year_missing & mask_tmdb_year_present, C.RELEASE_YEAR_COL] = df_items_rich.loc[mask_ml_year_missing & mask_tmdb_year_present, 'tmdb_year']
                    df_items_rich.drop(columns=['tmdb_year'], inplace=True, errors='ignore')


                # Prioriser les genres de TMDB s'ils existent et ne sont pas vides (logique de suffixes)
                genres_col_tmdb = C.GENRES_COL + '_tmdb'
                genres_col_ml = C.GENRES_COL + '_ml' # Si C.GENRES_COL était dans df_items_rich avant merge
                
                if genres_col_tmdb in df_items_rich.columns : # Si la colonne genre de TMDB existe après merge
                    if genres_col_ml in df_items_rich.columns: # Si la colonne genre de MovieLens existe aussi
                        # Utiliser les genres TMDB si présents et non vides, sinon ceux de MovieLens
                        df_items_rich[C.GENRES_COL] = df_items_rich[genres_col_tmdb].fillna('')
                        mask_tmdb_g_empty = df_items_rich[C.GENRES_COL] == ''
                        df_items_rich.loc[mask_tmdb_g_empty, C.GENRES_COL] = df_items_rich.loc[mask_tmdb_g_empty, genres_col_ml].fillna('')
                    else: # Si seulement la colonne TMDB existe (ou a été nommée C.GENRES_COL par le merge)
                        df_items_rich[C.GENRES_COL] = df_items_rich[genres_col_tmdb].fillna('')
                elif genres_col_ml in df_items_rich.columns: # Si seulement la colonne MovieLens existe
                     df_items_rich[C.GENRES_COL] = df_items_rich[genres_col_ml].fillna('')
                # else: C.GENRES_COL a été géré correctement par le merge ou était déjà bon

                cols_to_drop_after_merge = [col for col in df_items_rich.columns if col.endswith('_ml') or col.endswith('_tmdb')]
                df_items_rich.drop(columns=cols_to_drop_after_merge, inplace=True, errors='ignore')
                
        except FileNotFoundError:
            print(f"AVERTISSEMENT (load_items): Fichier TMDB '{tmdb_filepath}' non trouvé.")
        except Exception as e:
            print(f"AVERTISSEMENT (load_items): Erreur lors du chargement/fusion de TMDB: {e}.")
        
        try:
            df_links = pd.read_csv(links_filepath)
            if C.ITEM_ID_COL in df_links.columns and C.TMDB_ID_COL in df_links.columns:
                df_links_subset = df_links[[C.ITEM_ID_COL, C.TMDB_ID_COL]].copy()
                df_links_subset[C.TMDB_ID_COL] = pd.to_numeric(df_links_subset[C.TMDB_ID_COL], errors='coerce').astype('Int64')
                if C.TMDB_ID_COL in df_items_rich.columns:
                    df_items_rich = df_items_rich.drop(columns=[C.TMDB_ID_COL], errors='ignore')
                df_items_rich = pd.merge(df_items_rich, df_links_subset, on=C.ITEM_ID_COL, how='left')
        except FileNotFoundError:
            print(f"AVERTISSEMENT (load_items): Fichier links '{links_filepath}' non trouvé.")
        except Exception as e:
            print(f"AVERTISSEMENT (load_items): Erreur lors du chargement/fusion de links: {e}.")
                
        if C.RELEASE_YEAR_COL in df_items_rich.columns:
            df_items_rich[C.RELEASE_YEAR_COL] = pd.to_numeric(df_items_rich[C.RELEASE_YEAR_COL], errors='coerce').astype('Int64')

        df_items_rich = df_items_rich.loc[:,~df_items_rich.columns.duplicated()]

        print(f"load_items: Chargement des items terminé. DataFrame final: {df_items_rich.shape} lignes.")
        return df_items_rich

    except FileNotFoundError: 
        print(f"ERREUR Critique (load_items): Fichier movies.csv '{movies_filepath}' non trouvé.")
        raise
    except KeyError as e: 
        print(f"ERREUR Critique de clé (load_items) sur {movies_filepath}: {e}.")
        raise
    except Exception as e:
        print(f"Une erreur inattendue est survenue dans load_items: {e}")
        raise

def export_evaluation_report(df, model_name, accuracy=None, precision=None):
    """
    Export the evaluation report to the specified evaluation directory

    """

    today_str = datetime.now().strftime("%Y_%m_%d")
    base_filename = f"evaluation_report_{today_str}"
    i = 1
    while True:
        filename = f"{base_filename}_{i}.csv"
        path = C.EVALUATION_PATH / filename
        if not path.exists():
            break
        i += 1
    df = df.copy()
    # Insert model names as first column to keep track the model considered
    df.insert(0, "name", model_name)

    # Add accuracy and precision 
    if accuracy is not None:
        df["accuracy"] = accuracy
    if precision is not None:
        df["precision"] = precision
    df.to_csv(path, index=False)
    print(f"Evaluation report successfully exported to: {path}")
    
@st.cache_data
def load_posters_dict(posters_dir=None):
        """
        Charge les chemins des posters JPG dans un dictionnaire {movieId: chemin_image}.
        Le nom du fichier doit commencer par l'id MovieLens (movieId), ex: '1234_some_title.jpg'.
        Le répertoire par défaut est 'mlsmm2156/data/small/content/posters'.
        """
        if posters_dir is None:
            posters_dir =  C.CONTENT_PATH / C.POSTERS_LOCAL_DIR
        else:
            posters_dir = Path(posters_dir)
        posters_dict = {}
        if not posters_dir.exists():
            print(f"Le dossier des posters '{posters_dir}' n'existe pas.")
            return posters_dict
        for img_path in posters_dir.glob("*.jpg"):
            match = re.match(r"^(\d+)", img_path.stem)
            if match:
                movie_id = int(match.group(1))
                posters_dict[movie_id] = str(img_path.resolve())
        return posters_dict



if __name__ == '__main__':
    print("Test de chargement des items...")
    try:
        def download_and_extract_zip(url, extract_to="mlsmm2156"):
            zip_path = "data_temp.zip"

            # Télécharger le ZIP
            print("📥 Téléchargement du fichier ZIP...")
            response = requests.get(url)
            if response.status_code != 200:
                raise Exception(f"Échec du téléchargement: {response.status_code}")

            with open(zip_path, "wb") as f:
                f.write(response.content)

            # Décompresser le ZIP
            print("📦 Décompression...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)

            # Supprimer le fichier ZIP après extraction
            os.remove(zip_path)
            print(f"✅ Données extraites dans: {extract_to}")

        # Appel prioritaire au lancement du main
        if not os.path.exists("mlsmm2156/data/small/content"):  # éviter de re-télécharger si déjà là
            download_and_extract_zip("https://www.dropbox.com/scl/fi/zr5184em7ajn0d3naqnof/mlsmm2156.zip?rlkey=eny294bo4s2dmih6msfnausbq&st=8fm7tmhg&dl=1")

        df_i = load_items()
        print("\nExtrait de df_items_global (titres sans année pour affichage):")
        cols_to_show = [C.ITEM_ID_COL, C.LABEL_COL]
        if C.RELEASE_YEAR_COL in df_i.columns: cols_to_show.append(C.RELEASE_YEAR_COL)
        if C.GENRES_COL in df_i.columns: cols_to_show.append(C.GENRES_COL)
        print(df_i[cols_to_show].head(10))

        print("\nQuelques titres spécifiques pour vérifier le format (devraient être sans année) :")
        titles_to_check = ["The Lion King", "A Beautiful Mind", "The Usual Suspects", "The Pianist"]
        for title_check in titles_to_check:
            # Chercher une correspondance partielle car les titres peuvent légèrement varier (ex: accents)
            found_movies = df_i[df_i[C.LABEL_COL].str.contains(title_check.replace("The ", "").replace("A ", ""), case=False, na=False)]
            if not found_movies.empty:
                print(f"Trouvé pour '{title_check}':")
                print(found_movies[[C.LABEL_COL, C.RELEASE_YEAR_COL]].head())
            else:
                print(f"Pas de correspondance trouvée pour '{title_check}' dans les titres traités.")
        print("\nChargement des posters...")
        posters_dict = load_posters_dict()
        print("Début du dictionnaire des posters (10 premiers éléments) :")
        posters_items = list(posters_dict.items())
        # Exemple d'utilisation de la méthode get sur le dictionnaire posters_dict
        # On tente de récupérer le chemin du poster pour un movie_id donné (par exemple 1)
        movie_id_test = 1
        poster_path = posters_dict.get(movie_id_test)
        if poster_path is not None:
            print(f"Poster trouvé pour movie_id {movie_id_test}: {poster_path}")
        else:
            print(f"Aucun poster trouvé pour movie_id {movie_id_test}")
        
    except Exception as e_test:
        print(f"Erreur pendant le test de load_items: {e_test}")
