# 1. Image de base Python légère
FROM python:3.12-slim

# 2. Répertoire de travail dans le conteneur
WORKDIR /app

# 3. Installation des dépendances système requises (nécessaires pour ChromaDB et la compilation)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Copie et installation des exigences Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copie de tout le reste du projet (en respectant le .dockerignore)
COPY . .

# 6. Exposition du port Flask
EXPOSE 5000

# 7. Commande de démarrage du serveur
CMD ["python", "app.py"]