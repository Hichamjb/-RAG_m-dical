from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from config import DB_DIR

print("⏳ Initialisation de la base de données ChromaDB...")

# Initialisation des embeddings
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Initialisation de la base vectorielle persistante
vector_store = Chroma(
    collection_name="medical_records",
    embedding_function=embeddings,
    persist_directory=DB_DIR
)

print(f"✅ Base de données ChromaDB prête dans : {DB_DIR}")