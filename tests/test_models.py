# --- AGENT_VF/api/main.py ---
"""
API FastAPI pour interagir avec l'agent IA (Version LangGraph).
"""
import logging
from fastapi import FastAPI, HTTPException, Path, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
import asyncio
import os
import json

# Importer la fonction d'orchestration LangGraph
from AGENT_VF.orchestrator.workflow import run_chapter_generation
# Importer les tâches ETL
from AGENT_VF.etl import conversion, extraction, cleaning, transformation, indexation
# Importer les modèles Pydantic (si utilisés directement ici, sinon via les modules ETL/Orchestrator)
# from AGENT_VF.core.models import LogEntry # Pas directement utilisé ici, mais bon exemple d'import

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(
    title="Agent IA de Rédaction de Mémoire (LangGraph)",
    description="API pour gérer l'ingestion de documents et la génération de chapitres via LangGraph.",
    version="0.2.0",
)

# --- Modèles Pydantic ---

class IngestRequest(BaseModel):
    directory_path: str

class IngestResponse(BaseModel):
    message: str
    processed_files: int
    failed_files: List[str]

class GenerateRequest(BaseModel):
    max_retries: int = Field(default=3, ge=0, le=5, description="Nombre maximum de tentatives de réécriture.")

class GenerateResponse(BaseModel):
    message: str
    us_id: str
    result: Optional[Dict[str, Any]] = None

# --- Endpoints ---

@app.post("/ingest", response_model=IngestResponse, status_code=200)
async def ingest_documents(request: IngestRequest):
    """
    Lance l'ingestion des documents d'un répertoire spécifié.
    """
    logging.info(f"Requête d'ingestion reçue pour le répertoire: {request.directory_path}")
    processed_count = 0
    failed_files = []

    try:
        if not os.path.isdir(request.directory_path):
             raise HTTPException(status_code=404, detail=f"Répertoire non trouvé: {request.directory_path}")

        # Utilisation de process_directory pour une gestion plus centralisée si souhaité à l'avenir
        # Pour l'instant, on itère sur les fichiers et appelle run_conversion
        # comme dans les versions précédentes pour minimiser les changements de logique ici.
        files_to_process = [
            os.path.join(request.directory_path, f)
            for f in os.listdir(request.directory_path)
            if os.path.isfile(os.path.join(request.directory_path, f))
        ]
        
        # Exemple d'utilisation de process_directory si on voulait l'utiliser ici :
        # conversion_results = conversion.process_directory(
        #     directory_path=request.directory_path,
        #     # use_unstructured_processing=True, # Exemple
        #     # unstructured_base_output_dir="./unstructured_output" # Exemple
        # )
        # for file_path, (raw_text, converter) in conversion_results.items():
        #    filename = os.path.basename(file_path)
        #    ... (suite du traitement)

        for file_path in files_to_process:
            filename = os.path.basename(file_path)
            logging.info(f"Traitement du fichier: {filename}")
            
            raw_text, converter = conversion.run_conversion(file_path)

            if raw_text is None:
                logging.error(f"Échec conversion: {filename}")
                failed_files.append(filename)
                continue

            log_entry = transformation.transform_to_log_entry(
                raw_text=raw_text,
                source_filename=filename,
                source_file_path=file_path,
                converter_used=converter
            )
            if log_entry is None:
                logging.error(f"Échec transformation: {filename}")
                failed_files.append(filename)
                continue

            success = indexation.run_indexation(log_entry)
            if not success:
                logging.error(f"Échec indexation: {filename}")
                failed_files.append(filename)
                continue
            processed_count += 1
        
        msg = f"Ingestion terminée. {processed_count} fichiers traités."
        if failed_files:
            msg += f" Échecs pour: {', '.join(failed_files)}"
        logging.info(msg)
        return IngestResponse(message=msg, processed_files=processed_count, failed_files=failed_files)

    except Exception as e:
        logging.error(f"Erreur majeure lors de l'ingestion depuis {request.directory_path}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur serveur lors de l'ingestion: {str(e)}")


@app.post("/generate/{us_id}", response_model=GenerateResponse, status_code=200)
async def generate_report_chapter(
    us_id: str = Path(..., title="ID de la User Story", description="L'ID de la US (ex: US001)"),
    request: GenerateRequest = Depends()
    ):
    """
    Lance la génération d'un chapitre spécifique via le workflow LangGraph.
    """
    logging.info(f"Requête de génération reçue pour US ID: {us_id} avec max_retries={request.max_retries}")

    workflow_result = run_chapter_generation(us_id=us_id, max_retries=request.max_retries)

    if workflow_result and workflow_result.get("status") == "success":
        logging.info(f"Génération réussie pour {us_id}.")
        return GenerateResponse(
            message=f"Chapitre pour {us_id} généré avec succès.",
            us_id=us_id,
            result=workflow_result
        )
    elif workflow_result:
        error_detail = workflow_result.get('error', 'Raison inconnue')
        logging.error(f"Échec de la génération pour {us_id}: {error_detail}")
        return GenerateResponse(
            message=f"Échec de la génération du chapitre pour {us_id}.",
            us_id=us_id,
            result=workflow_result
        )
    else:
         logging.error(f"Erreur inattendue lors de la génération pour {us_id}. Le workflow n'a pas retourné de résultat.")
         raise HTTPException(status_code=500, detail=f"Erreur serveur inattendue lors de la génération pour {us_id}.")


@app.get("/report/{us_id}", response_model=Dict[str, Any])
async def get_report_content(us_id: str):
    """Récupère le dernier résultat de génération pour une US ID (simulation)."""
    reports_dir = os.getenv("GENERATED_REPORTS_DIR", "./generated_reports")
    if not os.path.exists(reports_dir):
        try:
            os.makedirs(reports_dir, exist_ok=True)
        except OSError as e:
            logging.error(f"Impossible de créer le répertoire des rapports {reports_dir}: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur serveur: impossible de créer le répertoire des rapports.")

    result_file = os.path.join(reports_dir, f"{us_id}_result.json")
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Aucun rapport trouvé pour US ID: {us_id}")
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Erreur lors de la lecture du rapport: {str(e)}")