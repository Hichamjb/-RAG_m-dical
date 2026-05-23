import os
from dotenv import load_dotenv

# Charge les variables cachées depuis le fichier .env
load_dotenv()
# Vérification de la clé API
if "XAI_API_KEY" not in os.environ:
    raise ValueError("Veuillez configurer la variable d'environnement 'XAI_API_KEY'")

# Configuration des dossiers
UPLOAD_FOLDER = "./uploaded_files"
DB_DIR = "./db"

# Création des dossiers s'ils n'existent pas
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)