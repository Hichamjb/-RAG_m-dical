from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from database import vector_store

def process_and_add_file(file_path: str, patient_id: str, file_id: str) -> int:
    """Découpe le PDF en morceaux et l'ajoute à ChromaDB avec les métadonnées."""
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    
    timestamp = datetime.now().isoformat()
    for chunk in chunks:
        chunk.metadata.update({
            "patient_id": patient_id,
            "file_id": file_id,
            "timestamp": timestamp
        })
    
    vector_store.add_documents(chunks)
    return len(chunks)