import pandas as pd
import requests
import time
import os
import constants as c

# Clé API TMDb
TMDB_API_KEY = "8585fb6e648921d5d32373bb5dd4d9c1"

# Lecture du fichier links.csv (movieId, imdbId, tmdbId)
df_links = pd.read_csv(c.Constant.CONTENT_PATH / c.Constant.LINKS_FILENAME).dropna(subset=["tmdbId"])
df_links["tmdbId"] = df_links["tmdbId"].astype(int)

# Fichier de sauvegarde progressive
SAVE_PATH = "data/small/content/tmdb_full_features.csv"

def get_tmdb_full_info(tmdb_id):
    base_url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
    params = {"api_key": TMDB_API_KEY, "language": "en-US"}

    info_resp = requests.get(base_url, params=params)
    if info_resp.status_code == 429:
        print("Trop de requêtes - Pause de 10 secondes...")
        time.sleep(10)
        return get_tmdb_full_info(tmdb_id)
    if info_resp.status_code != 200:
        print(f"Erreur info pour {tmdb_id} : {info_resp.status_code}")
        return None
    info = info_resp.json()

    credits_url = f"{base_url}/credits"
    credits_resp = requests.get(credits_url, params=params)
    if credits_resp.status_code == 429:
        print("Trop de requêtes (credits) - Pause de 10 secondes...")
        time.sleep(10)
        return get_tmdb_full_info(tmdb_id)
    if credits_resp.status_code == 200:
        credits = credits_resp.json()
        cast = ", ".join([member["name"] for member in credits["cast"][:3]])
        director = next((m["name"] for m in credits["crew"] if m["job"] == "Director"), None)
    else:
        cast, director = None, None

    genres = ", ".join([g["name"] for g in info.get("genres", [])])

    return {
        "runtime": info.get("runtime"),
        "genres": genres,
        "cast": cast,
        "director": director,
        "vote_count": info.get("vote_count"),
        "popularity": info.get("popularity"),
        "budget": info.get("budget"),
        "revenue": info.get("revenue"),
        "original_language": info.get("original_language"),
        "vote_average": info.get("vote_average")
    }

if os.path.exists(SAVE_PATH):
    df_saved = pd.read_csv(SAVE_PATH, index_col="movieId")
    already_scraped_ids = set(df_saved.index)
else:
    df_saved = pd.DataFrame()
    already_scraped_ids = set()

for _, row in df_links.iterrows():
    movie_id = row["movieId"]
    tmdb_id = row["tmdbId"]

    if movie_id in already_scraped_ids:
        continue

    try:
        features = get_tmdb_full_info(tmdb_id)
        if features:
            features["movieId"] = movie_id
            df_row = pd.DataFrame([features]).set_index("movieId")
            df_saved = pd.concat([df_saved, df_row])
            df_saved.to_csv(SAVE_PATH)
            print(f"Scraping réussi - movieId {movie_id} (tmdbId {tmdb_id})")
        else:
            print(f"Scraping échoué - movieId {movie_id} (tmdbId {tmdb_id})")
    except Exception as e:
        print(f"Erreur pour movieId {movie_id} : {e}")
        print("Pause de sécurité de 10 secondes")
        time.sleep(10)

    time.sleep(0.1)

print("Scraping terminé. Données enregistrées dans :", SAVE_PATH)