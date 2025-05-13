# --- AGENT_VF/rag/retriever.py ---
"""
Module pour récupérer des informations pertinentes depuis la base vectorielle locale (ChromaDB).
"""
import logging
from typing import List, Dict, Any

# Importer le client ChromaDB et le modèle d'embedding
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings # Doit correspondre à celui de l'indexation

# Importer les configurations depuis indexation.py ou un fichier config centralisé
from etl.indexation import VECTOR_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL_NAME, get_embedding_model, get_vector_db_collection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Initialisation (Gestion simplifiée pour le stub) ---
# Les fonctions get_embedding_model et get_vector_db_collection gèrent l'initialisation lazy.

def retrieve(query: str, k: int = 5, filter_criteria: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Récupère les k chunks les plus pertinents pour une requête, avec filtre optionnel.

    Args:
        query (str): La requête de l'utilisateur.
        k (int): Le nombre de chunks à récupérer.
        filter_criteria (Optional[Dict[str, Any]]): Dictionnaire pour filtrer les métadonnées
                                                    (ex: {"source_filename": "doc.pdf"}).

    Returns:
        List[Dict[str, Any]]: Une liste de dictionnaires, chaque dict contenant
                              'chunk_text' et 'metadata'.
    """
    logging.info(f"Récupération de {k} chunks pour la requête: '{query[:50]}...' (Filtre: {filter_criteria})")
    if not query:
        logging.warning("Requête vide fournie au retriever.")
        return []

    try:
        # Assurer l'initialisation du modèle et de la collection
        collection = get_vector_db_collection()

        # 1. Interroger la base de données vectorielle
        # Chroma utilise la fonction d'embedding configurée lors de l'init
        # Le filtre 'where' utilise la syntaxe de filtrage de ChromaDB:
        # https://docs.trychroma.com/usage-guide#using-where-filters
        results = collection.similarity_search_with_score(
            query=query,
            k=k,
            where=filter_criteria # Appliquer le filtre s'il est fourni
        )

        # 2. Formater les résultats
        relevant_docs = []
        for doc, score in results:
            relevant_docs.append({
                "chunk_text": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score) # Convertir le score en float standard
            })

        logging.info(f"{len(relevant_docs)} chunks pertinents trouvés pour '{query[:50]}...'.")
        return relevant_docs

    except Exception as e:
        logging.error(f"Erreur lors de la récupération pour la requête '{query[:50]}...': {e}", exc_info=True)
        return []

# --- Justification du choix ChromaDB vs FAISS ---
# ChromaDB a été choisi pour cette architecture locale car :
# 1. Simplicité d'installation et d'utilisation : Fonctionne comme une bibliothèque Python
#    avec une option de persistance simple basée sur des fichiers locaux (ou peut tourner en client/serveur).
#    FAISS nécessite souvent une gestion plus manuelle des index et de la persistance.
# 2. API Python-native : L'intégration avec LangChain et l'écosystème Python est très directe.
# 3. Fonctionnalités de filtrage : ChromaDB offre des capacités de filtrage sur les métadonnées
#    relativement faciles à utiliser, ce qui est utile pour affiner la recherche RAG.
# 4. Développement actif : ChromaDB est un projet open-source activement développé.
# FAISS reste une excellente option, surtout pour des besoins de performance extrêmes sur de très
# grands datasets, mais sa mise en place locale peut être légèrement plus complexe.
# Pour une application démarrant localement, ChromaDB offre un meilleur compromis facilité/performance.

# End of file