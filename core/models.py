# --- AGENT_VF/core/models.py ---
"""
Modèles Pydantic partagés pour l'application.
"""
import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, UUID4, field_validator
import uuid
import logging

logger = logging.getLogger(__name__)

class DocumentMetadata(BaseModel):
    """Métadonnées associées à un document source."""
    source_filename: str
    source_file_path: Optional[str] = None
    converter_used: Optional[str] = None
    processed_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)

class ExtractedEntities(BaseModel):
    """Entités extraites d'un texte."""
    persons: List[str] = Field(default_factory=list)
    organizations: List[str] = Field(default_factory=list)
    locations: List[str] = Field(default_factory=list)
    misc: List[str] = Field(default_factory=list)

class LogEntry(BaseModel):
    """Représentation structurée d'une entrée de journal après ETL."""
    entry_id: UUID4 = Field(default_factory=uuid.uuid4)
    document_metadata: DocumentMetadata
    entry_date: Optional[datetime.date] = None
    raw_text: Optional[str] = None
    cleaned_text: str
    extracted_entities: ExtractedEntities

    @field_validator('entry_date', mode='before')
    @classmethod
    def parse_date(cls, value: Any) -> Optional[datetime.date]:
        """Tente de parser une date si elle est fournie comme string."""
        if isinstance(value, str):
            try:
                return datetime.datetime.fromisoformat(value.replace('Z', '+00:00')).date()
            except ValueError:
                 try:
                     return datetime.datetime.strptime(value, '%Y-%m-%d').date()
                 except ValueError as e:
                    logger.warning(f"Impossible de parser la date '{value}': {e}. Utilisation de None.")
                    return None
        elif isinstance(value, datetime.datetime):
            return value.date()
        elif isinstance(value, datetime.date):
            return value
        elif value is None:
            return None
        else:
            logger.warning(f"Type de date inattendu '{type(value)}' pour la valeur '{value}'. Utilisation de None.")
            return None

class LogChunk(BaseModel):
    """Représentation d'un chunk de texte pour l'indexation vectorielle."""
    chunk_id: UUID4 = Field(default_factory=uuid.uuid4)
    parent_entry_id: UUID4
    chunk_text: str
    chunk_order: int # Standardisé sur chunk_order
    metadata: Dict[str, Any] = Field(default_factory=dict)