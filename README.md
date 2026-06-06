# 🏥 Automated Medical RAG API System (AI-Agent)

Un système RAG (Retrieval-Augmented Generation) médical de niveau entreprise. Ce projet est une **API backend robuste construite avec Flask**, intégrant **LangGraph** (pour le workflow de l'agent IA) et **ChromaDB** utilisant les **Embeddings HuggingFace** et **Grok (XAI)**. L'ensemble de l'application est entièrement conteneurisé avec **Docker** pour un déploiement cloud immédiat et sans frictions.

---

## 📌 1. Comment ça marche ? (L'idée et la logique)

Ce système repose sur l'architecture **RAG** couplée à la puissance des **AI Agents** :

1. **Traitement des documents (Document Processing) :** L'utilisateur (ou le médecin) upload un dossier médical au format PDF pour un patient spécifique (`patient_id`). Le système découpe ce document en petits morceaux (Chunks) tout en conservant le contexte médical.
2. **Base de données vectorielle (Vector Database) :** Ces textes sont transformés en vecteurs mathématiques (Embeddings) via un modèle HuggingFace performant, puis stockés localement dans **ChromaDB**. Chaque document est strictement lié à l'identifiant du patient.
3. **L'Agent Intelligent (LangGraph) :** Lorsqu'une question est posée sur un patient, la requête ne va pas directement au modèle LLM. Un agent intelligent (construit avec `StateGraph`) interroge d'abord ChromaDB pour récupérer uniquement les informations pertinentes du patient concerné, puis il envoie ces informations exactes à **Grok (XAI)** pour formuler une réponse médicale précise et sourcée à 100%.

---

## 📂 2. Structure du Projet

```text
RAG_médical/
├── db/                       # Dossier généré pour la base de données vectorielle ChromaDB
├── uploaded_files/           # Dossier de stockage temporaire des PDF médicaux
├── app.py                    # Point d'entrée principal de l'API Flask (Endpoints)
├── config.py                 # Gestionnaire de configuration et variables d'environnement
├── database.py               # Initialisation de ChromaDB et des modèles d'Embeddings
├── document_processor.py     # Logique d'extraction et de découpage des fichiers PDF
├── rag_agent.py              # Logique de l'Agent IA et orchestration via LangGraph
├── requirements.txt          # Dépendances et bibliothèques Python
├── Dockerfile                # Configuration pour la création de l'image Docker
├── .dockerignore             # Fichiers à exclure pour alléger l'image Docker (ex: env/, db/)
└── .env                      # Fichier contenant les clés API (non versionné sur Git)
