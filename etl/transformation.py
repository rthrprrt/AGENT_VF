# --- AGENT_VF/etl/transformation.py ---
"""
Wrapper pour le script de transformation des données extraites en un format structuré.
Utilise les modèles Pydantic définis dans core.models.
"""
import logging
import datetime
from typing import Optional, Dict, Any

# Importer les modèles Pydantic depuis core
from core.models import DocumentMetadata, ExtractedEntities, LogEntry
# Importer les wrappers de nettoyage et d'extraction
from .cleaning import run_cleaning
from .extraction import run_extraction

# Supposer que le script original n'est plus directement appelé ici,
# mais que sa logique est intégrée ou remplacée par les wrappers.

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def transform_to_log_entry(
    raw_text: str,
    source_filename: str,
    source_file_path: Optional[str] = None,
    converter_used: Optional[str] = None
    ) -> Optional[LogEntry]:
    """
    Orchestre le nettoyage, l'extraction et la création d'un objet LogEntry.

    Args:
        raw_text (str): Le texte brut original.
        source_filename (str): Nom du fichier source.
        source_file_path (Optional[str]): Chemin complet du fichier source.
        converter_used (Optional[str]): Nom du convertisseur utilisé.

    Returns:
        Optional[LogEntry]: L'objet LogEntry structuré ou None en cas d'échec.
    """
    logging.info(f"Lancement de la transformation pour: {source_filename}")
    if not raw_text:
        logging.warning(f"Texte brut vide fourni pour la transformation de {source_filename}. Skip.")
        return None

    try:
        # 1. Nettoyer le texte brut
        cleaned_text = run_cleaning(raw_text)
        if not cleaned_text:
            logging.warning(f"Texte nettoyé vide pour {source_filename}. Skip transformation.")
            return None

        # 2. Extraire les métadonnées (date, entités)
        # Utiliser cleaned_text pour l'extraction peut être plus fiable
        entry_date, extracted_entities_dict = run_extraction(cleaned_text, source_filename)

        # 3. Créer l'objet DocumentMetadata
        doc_meta = DocumentMetadata(
            source_filename=source_filename,
            source_file_path=source_file_path,
            converter_used=converter_used
        )

        # 4. Créer l'objet ExtractedEntities
        entities = ExtractedEntities(**(extracted_entities_dict or {}))

        # 5. Créer l'objet LogEntry final
        log_entry = LogEntry(
            document_metadata=doc_meta,
            entry_date=entry_date,
            raw_text=raw_text, # Garder le texte brut si utile
            cleaned_text=cleaned_text,
            extracted_entities=entities
        )

        logging.info(f"Transformation réussie pour {source_filename}. ID: {log_entry.entry_id}")
        return log_entry

    except Exception as e:
        logging.error(f"Erreur lors de la transformation de {source_filename}: {e}", exc_info=True)
        return None

# End of file