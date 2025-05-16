# --- AGENT_VF/etl/indexation.py ---
"""
Wrapper pour le script d'indexation des données structurées dans une base vectorielle locale.
"""
import logging
from typing import List, Any, Dict, Optional
from AGENT_VF.core.models import LogEntry, LogChunk
import uuid
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", 'sentence-transformers/all-MiniLM-L6-v2')
VECTOR_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db") # Lu depuis l'env ou défaut
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "log_entries_v2") # Lu depuis l'env ou défaut

embedding_model_instance = None
vector_db_collection = None

def get_embedding_model():
    """Charge ou retourne l'instance du modèle d'embedding."""
    global embedding_model_instance
    if embedding_model_instance is None:
        try:
            logging.info(f"Chargement du modèle d'embedding local: {EMBEDDING_MODEL_NAME}")
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': False}
            embedding_model_instance = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            logging.info("Modèle d'embedding chargé avec succès.")
        except Exception as e:
            logging.error(f"Erreur lors du chargement du modèle d'embedding: {e}", exc_info=True)
            raise SystemExit("Échec critique: Impossible de charger le modèle d'embedding.")
    return embedding_model_instance

def get_vector_db_collection() -> Chroma: # Signature sans arguments, retourne un objet Chroma
    """
    Initialise ou retourne la collection ChromaDB.
    Utilise VECTOR_DB_PATH et COLLECTION_NAME définis globalement (via env ou défauts).
    Crée le répertoire VECTOR_DB_PATH si nécessaire.
    """
    global vector_db_collection
    if vector_db_collection is None:
        try:
            # Création du répertoire si non existant
            # Cet appel est crucial pour que Chroma(persist_directory=...) fonctionne si le chemin n'existe pas.
            os.makedirs(VECTOR_DB_PATH, exist_ok=True) 
            logging.info(f"Initialisation de ChromaDB: path='{VECTOR_DB_PATH}', collection='{COLLECTION_NAME}'")
            
            vector_db_client = Chroma(
                persist_directory=VECTOR_DB_PATH,
                embedding_function=get_embedding_model(),
                collection_name=COLLECTION_NAME
            )
            vector_db_collection = vector_db_client
            logging.info("Collection ChromaDB initialisée/récupérée.")
        except Exception as e:
            logging.error(f"Erreur lors de l'initialisation de ChromaDB: {e}", exc_info=True)
            raise SystemExit("Échec critique: Impossible d'initialiser la base vectorielle.")
    return vector_db_collection

def chunk_log_entry(log_entry: LogEntry) -> List[LogChunk]:
    """Divise le texte nettoyé d'une LogEntry en LogChunks."""
    chunks = []
    if not log_entry.cleaned_text:
        return chunks
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
        )
        text_chunks = text_splitter.split_text(log_entry.cleaned_text)

        for i, chunk_text in enumerate(text_chunks):
            chunk_metadata = {
                "source_filename": log_entry.document_metadata.source_filename,
                "entry_date": log_entry.entry_date.isoformat() if log_entry.entry_date else "unknown",
                "parent_entry_id": str(log_entry.entry_id)
            }
            log_chunk = LogChunk(
                parent_entry_id=log_entry.entry_id,
                chunk_text=chunk_text,
                chunk_order=i,
                metadata=chunk_metadata
            )
            chunks.append(log_chunk)
        logging.info(f"Entrée {log_entry.entry_id} divisée en {len(chunks)} chunks.")
        return chunks
    except Exception as e:
         logging.error(f"Erreur lors du chunking pour {log_entry.entry_id}: {e}", exc_info=True)
         return []

def index_chunks(log_chunks: List[LogChunk]) -> bool:
    """Indexe les chunks dans la base de données vectorielle ChromaDB."""
    if not log_chunks:
        logging.warning("Aucun chunk à indexer.")
        return True
    collection = get_vector_db_collection()
    try:
        chunk_texts = [chunk.chunk_text for chunk in log_chunks]
        chunk_ids = [str(chunk.chunk_id) for chunk in log_chunks]
        chunk_metadatas = [chunk.metadata for chunk in log_chunks]
        collection.add_texts(ids=chunk_ids, texts=chunk_texts, metadatas=chunk_metadatas)
        logging.info(f"{len(log_chunks)} chunks indexés avec succès pour l'entrée {log_chunks[0].parent_entry_id}.")
        return True
    except Exception as e:
        parent_id = log_chunks[0].parent_entry_id if log_chunks else "N/A"
        logging.error(f"Erreur lors de l'indexation des chunks pour l'entrée {parent_id}: {e}", exc_info=True)
        return False

def run_indexation(log_entry: LogEntry) -> bool: 
    """Orchestre le chunking et l'indexation pour une LogEntry complète."""
    logging.info(f"Lancement de l'indexation pour l'entrée: {log_entry.entry_id} ({log_entry.document_metadata.source_filename})")
    if not log_entry or not log_entry.cleaned_text:
        logging.warning(f"LogEntry invalide ou texte nettoyé vide pour {log_entry.entry_id}. Skip indexation.")
        return False
    collection = get_vector_db_collection()
    try:
        existing_results = collection.get(where={"parent_entry_id": str(log_entry.entry_id)}, limit=1) 
        if existing_results and existing_results.get('ids') and len(existing_results['ids']) > 0:
            logging.info(f"Entrée {log_entry.entry_id} semble déjà indexée ({len(existing_results['ids'])} chunk(s) trouvé(s)). Skip.")
            return True
    except Exception as e:
        logging.warning(f"Erreur lors de la vérification d'existence pour {log_entry.entry_id}: {e}. Tentative d'indexation.")

    log_chunks = chunk_log_entry(log_entry)
    if not log_chunks:
        logging.warning(f"Aucun chunk généré pour l'entrée {log_entry.entry_id}. Rien à indexer.")
        return True
    success = index_chunks(log_chunks)
    return success