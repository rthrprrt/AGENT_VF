# --- AGENT_VF/api/main.py ---
"""
API FastAPI pour interagir avec l'agent IA (Version LangGraph).
"""
import logging
from fastapi import FastAPI, HTTPException, Path, BackgroundTasks, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uuid
import asyncio

# Importer la fonction d'orchestration LangGraph
from orchestrator.workflow import run_chapter_generation
# Importer les tâches ETL (si elles sont déclenchées via API)
from etl import conversion, extraction, cleaning, transformation, indexation
import os # Pour lister les fichiers dans l'ingestion

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
    # Ajouter d'autres paramètres si nécessaire (ex: contexte additionnel)

class GenerateResponse(BaseModel):
    message: str
    us_id: str
    result: Optional[Dict[str, Any]] = None # Contient le résultat final ou les détails de l'échec

# --- Endpoints ---

@app.post("/ingest", response_model=IngestResponse, status_code=200)
async def ingest_documents(request: IngestRequest):
    """
    Lance l'ingestion des documents d'un répertoire spécifié (traitement synchrone simple pour l'API).
    NOTE: Pour de gros volumes, une approche asynchrone (ex: BackgroundTasks + suivi) serait préférable.
    """
    logging.info(f"Requête d'ingestion reçue pour le répertoire: {request.directory_path}")
    processed_count = 0
    failed_files = []

    try:
        if not os.path.isdir(request.directory_path):
             raise HTTPException(status_code=404, detail=f"Répertoire non trouvé: {request.directory_path}")

        for filename in os.listdir(request.directory_path):
            file_path = os.path.join(request.directory_path, filename)
            if os.path.isfile(file_path):
                logging.info(f"Traitement du fichier: {filename}")
                # --- Pipeline ETL Séquentiel ---
                raw_text, converter = conversion.run_conversion(file_path)
                if raw_text is None:
                    logging.error(f"Échec conversion: {filename}")
                    failed_files.append(filename)
                    continue

                entry_date, entities_dict = extraction.run_extraction(raw_text, filename)

                log_entry = transformation.transform_to_log_entry(
                    raw_text=raw_text,
                    source_filename=filename,
                    source_file_path=file_path,
                    converter_used=converter,
                    entry_date=entry_date,
                    extracted_entities_dict=entities_dict
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
                # --- Fin Pipeline ETL ---
                processed_count += 1
            else:
                logging.warning(f"Ignoré (pas un fichier): {filename}")

        msg = f"Ingestion terminée. {processed_count} fichiers traités."
        if failed_files:
            msg += f" Échecs pour: {', '.join(failed_files)}"
        logging.info(msg)
        return IngestResponse(message=msg, processed_files=processed_count, failed_files=failed_files)

    except Exception as e:
        logging.error(f"Erreur majeure lors de l'ingestion depuis {request.directory_path}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur serveur lors de l'ingestion: {e}")


@app.post("/generate/{us_id}", response_model=GenerateResponse, status_code=200)
async def generate_report_chapter(
    us_id: str = Path(..., title="ID de la User Story", description="L'ID de la US (ex: US001)"),
    request: GenerateRequest = GenerateRequest() # Utilise les valeurs par défaut
    ):
    """
    Lance la génération d'un chapitre spécifique via le workflow LangGraph.
    L'exécution est synchrone dans cet exemple d'API simple.
    """
    logging.info(f"Requête de génération reçue pour US ID: {us_id} avec max_retries={request.max_retries}")

    # Exécuter le workflow LangGraph directement
    # Pour une API plus robuste, utiliser BackgroundTasks ou un worker séparé
    result = run_chapter_generation(us_id=us_id, max_retries=request.max_retries)

    if result and result.get("status") == "success":
        logging.info(f"Génération réussie pour {us_id}.")
        return GenerateResponse(
            message=f"Chapitre pour {us_id} généré avec succès.",
            us_id=us_id,
            result=result
        )
    elif result:
        logging.error(f"Échec de la génération pour {us_id}: {result.get('error', 'Raison inconnue')}")
        # Retourner 200 mais avec un statut d'échec dans le corps
        return GenerateResponse(
            message=f"Échec de la génération du chapitre pour {us_id}.",
            us_id=us_id,
            result=result
        )
    else:
         logging.error(f"Erreur inattendue lors de la génération pour {us_id}.")
         raise HTTPException(status_code=500, detail=f"Erreur serveur inattendue lors de la génération pour {us_id}.")


# Les endpoints /status/{job_id} et /download/{report_id} basés sur Celery sont supprimés.
# Le statut est implicite dans la réponse de /generate (synchrone ici).
# Le téléchargement nécessiterait un mécanisme de stockage/récupération des résultats
# basé sur l'us_id ou un ID de rapport généré.

# Exemple simple pour récupérer le contenu (si stocké quelque part)
# Ceci est très basique et suppose un stockage simple (ex: fichier JSON par US ID)
@app.get("/report/{us_id}", response_model=Dict[str, Any])
async def get_report_content(us_id: str):
    """Récupère le dernier résultat de génération pour une US ID (simulation)."""
    # Simuler la lecture d'un fichier où le résultat est stocké
    result_file = f"./generated_reports/{us_id}_result.json" # Chemin exemple
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Aucun rapport trouvé pour US ID: {us_id}")
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Erreur lors de la lecture du rapport: {e}")

# End of file