# --- AGENT_VF/core/models.py ---
"""
Modèles Pydantic partagés pour l'application.
"""
import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, UUID4, field_validator
import uuid

class DocumentMetadata(BaseModel):
    """Métadonnées associées à un document source."""
    source_filename: str
    source_file_path: Optional[str] = None
    converter_used: Optional[str] = None
    processed_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)

class ExtractedEntities(BaseModel):
    """Entités extraites d'un texte."""
    persons: List[str] = []
    organizations: List[str] = []
    locations: List[str] = []
    misc: List[str] = [] # Pour les autres types d'entités (ex: MISC, DATE non structurée)

class LogEntry(BaseModel):
    """Représentation structurée d'une entrée de journal après ETL."""
    entry_id: UUID4 = Field(default_factory=uuid.uuid4)
    document_metadata: DocumentMetadata
    entry_date: Optional[datetime.date] = None
    raw_text: Optional[str] = None # Optionnel: garder le texte brut pour référence
    cleaned_text: str
    extracted_entities: ExtractedEntities

    @field_validator('entry_date', mode='before')
    @classmethod
    def parse_date(cls, value):
        """Tente de parser une date si elle est fournie comme string."""
        if isinstance(value, str):
            try:
                # Essayer plusieurs formats si nécessaire
                return datetime.datetime.fromisoformat(value.replace('Z', '+00:00')).date()
            except ValueError:
                 try:
                     # Ajouter d'autres formats si besoin, ex: '%d/%m/%Y'
                     return datetime.datetime.strptime(value, '%Y-%m-%d').date()
                 except ValueError:
                    return None
        return value

class LogChunk(BaseModel):
    """Représentation d'un chunk de texte pour l'indexation vectorielle."""
    chunk_id: UUID4 = Field(default_factory=uuid.uuid4)
    parent_entry_id: UUID4
    chunk_text: str
    chunk_order: int
    metadata: Dict[str, Any] = {} # Métadonnées spécifiques au chunk

# End of file