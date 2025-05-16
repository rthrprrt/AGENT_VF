# --- AGENT_VF/rag/retriever.py ---
"""
Module pour récupérer des informations pertinentes depuis la base vectorielle locale (ChromaDB).
"""
import logging
from typing import List, Dict, Any, Optional

from langchain_community.vectorstores import Chroma
from AGENT_VF.etl.indexation import get_vector_db_collection 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def retrieve(query: str, k: int = 5, filter_criteria: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Récupère les k chunks les plus pertinents pour une requête, avec filtre optionnel.
    """
    logging.info(f"Récupération de {k} chunks pour la requête: '{query[:50]}...' (Filtre: {filter_criteria})")
    if not query:
        logging.warning("Requête vide fournie au retriever.")
        return []

    try:
        collection = get_vector_db_collection()
        if collection is None:
            logging.error("Impossible de récupérer la collection de la base vectorielle.")
            return []
        
        effective_filter = filter_criteria if filter_criteria else None

        results_with_scores = collection.similarity_search_with_score(
            query=query,
            k=k,
            # La méthode `where` de LangChain pour ChromaDB attend un dictionnaire simple
            # pour le filtrage des métadonnées.
            # Exemple: where={"source_filename": "mon_fichier.pdf"}
            # Si des filtres plus complexes sont nécessaires (ex: $in, $gt),
            # il faudrait construire le dictionnaire `where` selon la syntaxe de ChromaDB
            # et s'assurer que le wrapper LangChain le transmet correctement.
            # Pour l'instant, on suppose un filtre simple.
            filter=effective_filter # LangChain Chroma utilise 'filter' pour le 'where' de ChromaDB
        )
        
        relevant_docs = []
        for doc, score in results_with_scores:
            metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
            relevant_docs.append({
                "chunk_text": doc.page_content,
                "metadata": metadata,
                "score": float(score) 
            })

        logging.info(f"{len(relevant_docs)} chunks pertinents trouvés pour '{query[:50]}...'.")
        return relevant_docs

    except Exception as e:
        logging.error(f"Erreur lors de la récupération pour la requête '{query[:50]}...': {e}", exc_info=True)
        return []