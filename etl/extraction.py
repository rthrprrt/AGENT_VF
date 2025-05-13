# --- AGENT_VF/etl/extraction.py ---
"""
Wrapper pour le script d'extraction de métadonnées (date, entités).
"""
import logging
import datetime
from typing import Optional, Tuple, Dict, Any

# Supposons que vos scripts originaux sont accessibles
try:
    from etl_scripts import extraction as original_extraction
except ImportError:
    logging.warning("Impossible d'importer 'etl_scripts.extraction'.")
    # Fonction factice
    class original_extraction:
        @staticmethod
        def extract_metadata(text: str, filename: str) -> Tuple[Optional[datetime.date], Dict[str, Any]]:
            logging.warning("Fonction factice 'extract_metadata' appelée.")
            return datetime.date.today(), {"persons": ["Dummy Person"], "organizations": ["Dummy Org"]}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_extraction(text: str, filename: str) -> Tuple[Optional[datetime.date], Dict[str, Any]]:
    """
    Exécute le script d'extraction de métadonnées sur le texte donné.

    Args:
        text (str): Le texte à analyser.
        filename (str): Le nom du fichier source pour contexte (ex: extraction de date).

    Returns:
        Tuple[Optional[datetime.date], Dict[str, Any]]: La date extraite et un dictionnaire d'entités.
    """
    logging.info(f"Lancement de l'extraction pour le fichier source: {filename}")
    try:
        # Appel de la fonction principale de votre script original
        entry_date, extracted_entities = original_extraction.extract_metadata(text, filename)
        logging.info(f"Extraction terminée pour {filename}. Date: {entry_date}, Entités: {list(extracted_entities.keys())}")
        return entry_date, extracted_entities
    except Exception as e:
        logging.error(f"Erreur inattendue lors de l'extraction pour {filename}: {e}", exc_info=True)
        return None, {}

# End of file