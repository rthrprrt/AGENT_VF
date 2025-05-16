# --- AGENT_VF/etl/transformation.py ---
"""
Wrapper pour le script de transformation des données extraites en un format structuré.
Utilise les modèles Pydantic définis dans core.models.
"""
import logging
import datetime
from typing import Optional, Dict, Any, Tuple 

from AGENT_VF.core.models import DocumentMetadata, ExtractedEntities, LogEntry
from AGENT_VF.etl.cleaning import run_cleaning
from AGENT_VF.etl.extraction import run_extraction # Assurez-vous que cet import est correct

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def transform_to_log_entry(
    raw_text: str,
    source_filename: str,
    source_file_path: Optional[str] = None,
    converter_used: Optional[str] = None
    # entry_id n'est plus un argument
    ) -> Optional[LogEntry]:
    """
    Orchestre le nettoyage, l'extraction et la création d'un objet LogEntry.
    L'entry_id est auto-généré par Pydantic.
    """
    logging.info(f"Lancement de la transformation pour: {source_filename}")
    if not raw_text:
        logging.warning(f"Texte brut vide fourni pour la transformation de {source_filename}. Skip.")
        return None

    try:
        cleaned_text = run_cleaning(raw_text)
        if not cleaned_text:
            logging.warning(f"Texte nettoyé vide pour {source_filename}. Skip transformation.")
            return None

        # S'assurer que run_extraction retourne bien un tuple de deux éléments
        extraction_result: Tuple[Optional[datetime.date], Dict[str, Any]] = run_extraction(cleaned_text, source_filename)
        entry_date, extracted_entities_dict = extraction_result
        
        entities_payload = extracted_entities_dict if isinstance(extracted_entities_dict, dict) else {}


        doc_meta = DocumentMetadata(
            source_filename=source_filename,
            source_file_path=source_file_path,
            converter_used=converter_used
        )

        entities = ExtractedEntities(**entities_payload)

        log_entry = LogEntry(
            # entry_id est auto-généré
            document_metadata=doc_meta,
            entry_date=entry_date,
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            extracted_entities=entities
        )

        logging.info(f"Transformation réussie pour {source_filename}. ID: {log_entry.entry_id}")
        return log_entry

    except ValueError as ve:
        if "not enough values to unpack" in str(ve) or "too many values to unpack" in str(ve):
            logging.error(f"Erreur de déballage des valeurs de run_extraction pour {source_filename}: {ve}. "
                          f"Vérifiez que AGENT_VF.etl.extraction.run_extraction (et par extension etl_scripts.extraction.extract_metadata) retourne bien un tuple (date, dict_entites).", exc_info=True)
        else:
            logging.error(f"Erreur de valeur lors de la transformation de {source_filename}: {ve}", exc_info=True)
        return None
    except Exception as e:
        logging.error(f"Erreur inattendue lors de la transformation de {source_filename}: {e}", exc_info=True)
        return None