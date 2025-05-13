# --- AGENT_VF/etl/indexation.py ---
"""
Wrapper pour le script d'indexation des données structurées dans une base vectorielle locale.
"""
import logging
from typing import List, Any, Dict, Optional
from core.models import LogEntry, LogChunk # Importer depuis core.models
import uuid

# Importer les dépendances nécessaires pour le chunking et l'embedding local
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Choisir un modèle d'embedding local (ex: Sentence Transformers via LangChain)
# Assurez-vous que le modèle est téléchargé localement ou accessible.
from langchain_community.embeddings import HuggingFaceEmbeddings # Ou OllamaEmbeddings si préféré

# Importer le client de la base vectorielle choisie (ChromaDB ici)
from langchain_community.vectorstores import Chroma

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
CHUNK_SIZE = 512 # Taille des chunks en caractères/tokens
CHUNK_OVERLAP = 64 # Chevauchement
# Utiliser un modèle d'embedding léger et performant, exécutable localement.
# 'all-MiniLM-L6-v2' est un bon point de départ.
# Alternative: utiliser OllamaEmbeddings si un serveur Ollama gère les embeddings.
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
VECTOR_DB_PATH = "./chroma_db" # Chemin pour la persistance locale de ChromaDB
COLLECTION_NAME = "log_entries_v2"

# --- Initialisation (Gestion simplifiée pour le stub) ---
# Dans une application réelle, l'initialisation serait gérée de manière centralisée
# et les instances passées en argument ou via un système d'injection de dépendances.
embedding_model_instance = None
vector_db_collection = None

def get_embedding_model():
    """Charge ou retourne l'instance du modèle d'embedding."""
    global embedding_model_instance
    if embedding_model_instance is None:
        try:
            logging.info(f"Chargement du modèle d'embedding local: {EMBEDDING_MODEL_NAME}")
            # Spécifier device='cpu' si pas de GPU ou pour forcer CPU
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': False} # Ou True selon le modèle/besoin
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

def get_vector_db_collection():
    """Initialise ou retourne la collection ChromaDB."""
    global vector_db_collection
    if vector_db_collection is None:
        try:
            logging.info(f"Initialisation de ChromaDB: path='{VECTOR_DB_PATH}', collection='{COLLECTION_NAME}'")
            # Initialise le client ChromaDB avec persistance locale
            vector_db_client = Chroma(
                persist_directory=VECTOR_DB_PATH,
                embedding_function=get_embedding_model(), # Utilise le modèle chargé
                collection_name=COLLECTION_NAME
            )
            # Note: Chroma crée la collection si elle n'existe pas lors de l'ajout.
            # Pour forcer la création ou vérifier l'existence, des appels client directs seraient nécessaires.
            # Pour ce stub, on suppose que l'instance gère cela.
            vector_db_collection = vector_db_client
            logging.info("Collection ChromaDB initialisée/récupérée.")
        except Exception as e:
            logging.error(f"Erreur lors de l'initialisation de ChromaDB: {e}", exc_info=True)
            raise SystemExit("Échec critique: Impossible d'initialiser la base vectorielle.")
    return vector_db_collection

# --- Fonctions de Chunking et d'Indexation ---

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
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""] # Séparateurs courants
        )
        text_chunks = text_splitter.split_text(log_entry.cleaned_text)

        for i, chunk_text in enumerate(text_chunks):
            chunk_metadata = {
                "source_filename": log_entry.document_metadata.source_filename,
                "entry_date": log_entry.entry_date.isoformat() if log_entry.entry_date else "unknown",
                "parent_entry_id": str(log_entry.entry_id) # Convertir UUID en str pour Chroma
                # Ajouter d'autres métadonnées filtrables si besoin
            }
            # Filtrer les métadonnées None n'est plus nécessaire si on gère les types correctement
            # chunk_metadata = {k: v for k, v in chunk_metadata.items() if v is not None}

            log_chunk = LogChunk(
                parent_entry_id=log_entry.entry_id,
                chunk_text=chunk_text,
                chunk_order=i,
                metadata=chunk_metadata
                # chunk_id est généré par Pydantic
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
        return True # Pas d'erreur, juste rien à faire

    collection = get_vector_db_collection()
    embedding_func = collection.embedding_function # Récupérer la fonction d'embedding configurée

    try:
        chunk_texts = [chunk.chunk_text for chunk in log_chunks]
        chunk_ids = [str(chunk.chunk_id) for chunk in log_chunks] # Utiliser l'ID Pydantic comme ID Chroma
        chunk_metadatas = [chunk.metadata for chunk in log_chunks]

        # Chroma gère l'embedding via l'embedding_function fournie à l'initialisation
        collection.add_texts(
            ids=chunk_ids,
            texts=chunk_texts,
            metadatas=chunk_metadatas
        )
        # Forcer la persistance si nécessaire (selon la config Chroma)
        # collection._client.persist() # Peut être nécessaire selon la version/config

        logging.info(f"{len(log_chunks)} chunks indexés avec succès pour l'entrée {log_chunks[0].parent_entry_id}.")
        return True

    except Exception as e:
        parent_id = log_chunks[0].parent_entry_id if log_chunks else "N/A"
        logging.error(f"Erreur lors de l'indexation des chunks pour l'entrée {parent_id}: {e}", exc_info=True)
        return False

# --- Fonction Principale d'Indexation pour une Entrée ---
def run_indexation(log_entry: LogEntry) -> bool:
    """Orchestre le chunking et l'indexation pour une LogEntry complète."""
    logging.info(f"Lancement de l'indexation pour l'entrée: {log_entry.entry_id} ({log_entry.document_metadata.source_filename})")
    if not log_entry or not log_entry.cleaned_text:
        logging.warning(f"LogEntry invalide ou texte nettoyé vide pour {log_entry.entry_id}. Skip indexation.")
        return False

    # Idempotence: Vérifier si des chunks existent déjà pour cette entrée
    collection = get_vector_db_collection()
    try:
        existing = collection.get(where={"parent_entry_id": str(log_entry.entry_id)}, limit=1)
        if existing and existing.get('ids'):
            logging.info(f"Entrée {log_entry.entry_id} semble déjà indexée ({len(existing['ids'])} chunk(s) trouvé(s)). Skip.")
            # Option: Supprimer les anciens chunks avant de réindexer si une mise à jour est souhaitée
            # collection.delete(where={"parent_entry_id": str(log_entry.entry_id)})
            # logging.info(f"Anciens chunks supprimés pour {log_entry.entry_id}.")
            return True # Considérer comme succès si déjà traité
    except Exception as e:
        logging.warning(f"Erreur lors de la vérification d'existence pour {log_entry.entry_id}: {e}. Tentative d'indexation.")


    log_chunks = chunk_log_entry(log_entry)
    if not log_chunks:
        logging.warning(f"Aucun chunk généré pour l'entrée {log_entry.entry_id}. Rien à indexer.")
        return True # Pas d'échec, juste rien à faire

    success = index_chunks(log_chunks)

    return success

# End of file