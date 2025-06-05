import requests

def download_file(url, save_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        print("Téléchargement terminé.")
    else:
        print("Erreur lors du téléchargement du fichier.")

# Exemple d'utilisation
file_url = "https://1drv.ms/u/c/3814d4299f55d577/ETMtHrSqeiJHgKrchDqw_g8BkmUEuMxfcabL3OD1T1ld9Q"
save_path = "télécharé_fichier"  # Assurez-vous de spécifier le chemin complet et le nom du fichier
download_file(file_url, save_path)

import webbrowser

# Lien vers le fichier OneDrive
url = "https://1drv.ms/u/c/3814d4299f55d577/ETMtHrSqeiJHgKrchDqw_g8BkmUEuMxfcabL3OD1T1ld9Q"

# Ouvrir le lien dans le navigateur par défaut
webbrowser.open(url)