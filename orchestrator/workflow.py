# --- AGENT_VF/orchestrator/workflow.py ---
"""
Définition du workflow LangGraph pour la génération de chapitres.
"""
import logging
import json
import uuid
import os
from typing import TypedDict, List, Optional, Dict, Any

from langgraph.graph import StateGraph, END
# Importation corrigée pour SqliteSaver, en supposant que langgraph-checkpoint-sqlite est installé
# et rend SqliteSaver disponible sous langgraph.checkpoint.sqlite
from langgraph.checkpoint.sqlite import SqliteSaver 

from AGENT_VF.rag import retriever
from AGENT_VF.writer import client as writer_client
from AGENT_VF.validation import validator
# from AGENT_VF.core.models import LogEntry # Pas utilisé directement ici

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GenerationState(TypedDict):
    """État du graphe pour la génération d'un chapitre."""
    us_id: str
    prompt_template: str
    acceptance_criteria: Dict[str, Any]
    initial_query: str
    context: Optional[List[Dict]]
    current_prompt: str
    generation: Optional[str]
    validation_errors: Optional[List[str]]
    retries_left: int
    final_chapter: Optional[str]

def retrieve_context(state: GenerationState) -> Dict[str, Any]:
    logging.info(f"Noeud RAG: Récupération de contexte pour US {state['us_id']}")
    query = state.get("initial_query", f"Informations pertinentes pour {state['us_id']}")
    rag_k = state.get("acceptance_criteria", {}).get("rag_k", 5)
    filter_criteria = state.get("acceptance_criteria", {}).get("rag_filter")

    retrieved_docs_with_scores = retriever.retrieve(query=query, k=rag_k, filter_criteria=filter_criteria)
    
    context_for_state = []
    # rag_score_threshold: L2 distance, lower is better. Default 1.0 is quite permissive.
    score_threshold = state.get("acceptance_criteria", {}).get("rag_score_threshold", 1.0) 
    
    if retrieved_docs_with_scores:
        for doc_info in retrieved_docs_with_scores:
            doc_score = doc_info.get("score", float('inf'))
            if doc_score <= score_threshold: 
                 context_for_state.append(doc_info)
            else:
                logging.info(f"Noeud RAG: Document écarté car score {doc_score} > seuil {score_threshold} (L2 distance)")
    
    logging.info(f"Noeud RAG: {len(context_for_state)} chunks pertinents retenus après filtrage par score (seuil L2: {score_threshold}).")
    if not context_for_state and retrieved_docs_with_scores:
        logging.warning(f"Noeud RAG: Tous les documents récupérés ont été filtrés par le seuil de score. Le contexte sera vide.")
    return {"context": context_for_state}

def construct_prompt(state: GenerationState) -> Dict[str, Any]:
    logging.info(f"Noeud Prompt: Construction du prompt pour US {state['us_id']}")
    template = state["prompt_template"]
    context_parts = []
    if state.get("context"):
        for i, doc_dict in enumerate(state["context"]):
            if isinstance(doc_dict, dict) and "chunk_text" in doc_dict:
                context_parts.append(f"Contexte {i+1}:\n{doc_dict['chunk_text']}")
            else:
                logging.warning(f"Élément de contexte inattendu ignoré: {doc_dict}")
    context_str = "\n---\n".join(context_parts) if context_parts else "Aucun contexte pertinent n'a été trouvé."

    refinement_prefix = ""
    current_retries_left = state["retries_left"]
    if state.get("validation_errors"): 
        errors_str = "\n - ".join(state["validation_errors"])
        refinement_prefix = (
            f"La tentative précédente a échoué à la validation pour les raisons suivantes:\n - {errors_str}\n"
            f"Veuillez corriger ces points et régénérer le chapitre.\n"
            f"Texte précédent (pour référence):\n'''\n{state.get('generation', '')}\n'''\n\n"
            "Nouvelle tentative:\n"
        )
        current_retries_left -= 1 

    final_prompt_content = template
    final_prompt_content = final_prompt_content.replace("{{ us_id }}", state.get("us_id", "[US_ID MANQUANT]"))
    final_prompt_content = final_prompt_content.replace("{{ chapter_title }}", state.get("acceptance_criteria", {}).get("chapter_title", "[TITRE_CHAPITRE MANQUANT]"))
    final_prompt_content = final_prompt_content.replace("{{ rag_context }}", context_str)
    
    for key, value in state.get("acceptance_criteria", {}).get("prompt_placeholders", {}).items():
        final_prompt_content = final_prompt_content.replace(f"{{{{ {key} }}}}", str(value))

    final_prompt_with_refinement = refinement_prefix + final_prompt_content
    import re
    final_prompt_with_refinement = re.sub(r"\{\{\s*[\w\.]+\s*\}\}", "[DONNÉE MANQUANTE]", final_prompt_with_refinement)

    logging.info(f"Noeud Prompt: Prompt final construit (longueur: {len(final_prompt_with_refinement)}). Tentatives restantes: {current_retries_left}")
    return {"current_prompt": final_prompt_with_refinement, "retries_left": current_retries_left}

def generate_chapter(state: GenerationState) -> Dict[str, Any]:
    logging.info(f"Noeud Writer: Génération du chapitre pour US {state['us_id']}")
    prompt = state.get("current_prompt")
    if not prompt:
        logging.error("Aucun prompt à envoyer au LLM.")
        return {"generation": "[Erreur: Prompt manquant]", "validation_errors": ["Prompt manquant pour la génération."]}
    generated_text = writer_client.generate(prompt)
    logging.info(f"Noeud Writer: Génération terminée (longueur: {len(generated_text)}).")
    return {"generation": generated_text}

def validate_chapter(state: GenerationState) -> Dict[str, Any]:
    logging.info(f"Noeud Validation: Validation du chapitre pour US {state['us_id']}")
    generated_text = state.get("generation")
    criteria = state.get("acceptance_criteria", {})
    min_words = criteria.get("min_words", 50)
    required_sections = criteria.get("required_sections", [])
    errors = []
    if not generated_text or generated_text.startswith("[Erreur"):
        errors.append("La génération a échoué ou retourné une erreur interne.")
        logging.error("Validation échouée: Génération LLM invalide.")
        return {"validation_errors": errors, "final_chapter": None}
    if not validator.check_length(generated_text, min_words):
        errors.append(f"Longueur insuffisante (minimum {min_words} mots).")
    structure_errors = validator.check_structure(generated_text, required_sections)
    errors.extend(structure_errors)
    if not errors:
        logging.info(f"Noeud Validation: Chapitre pour US {state['us_id']} validé avec succès.")
        return {"validation_errors": [], "final_chapter": generated_text}
    else:
        logging.warning(f"Noeud Validation: Échec de la validation pour US {state['us_id']}. Erreurs: {errors}")
        return {"validation_errors": errors, "final_chapter": None}

def should_retry(state: GenerationState) -> str:
    logging.info(f"Condition: Vérification de la nécessité de réessayer pour US {state['us_id']}")
    errors = state.get("validation_errors")
    retries_left = state.get("retries_left", 0) 
    if errors and retries_left >= 0: 
        logging.info(f"Condition: Validation échouée, {retries_left} tentatives restantes. -> retry")
        return "retry"
    elif errors:
        logging.error(f"Condition: Validation échouée et plus de tentatives pour US {state['us_id']}. -> end_failure")
        return "end_failure"
    else:
        logging.info(f"Condition: Validation réussie pour US {state['us_id']}. -> end_success")
        return "end_success"

def create_generation_workflow():
    workflow_builder = StateGraph(GenerationState)
    workflow_builder.add_node("retrieve_context", retrieve_context)
    workflow_builder.add_node("construct_prompt", construct_prompt)
    workflow_builder.add_node("generate_chapter", generate_chapter)
    workflow_builder.add_node("validate_chapter", validate_chapter)
    workflow_builder.set_entry_point("retrieve_context")
    workflow_builder.add_edge("retrieve_context", "construct_prompt")
    workflow_builder.add_edge("construct_prompt", "generate_chapter")
    workflow_builder.add_edge("generate_chapter", "validate_chapter")
    workflow_builder.add_conditional_edges(
        "validate_chapter",
        should_retry,
        {"retry": "construct_prompt", "end_success": END, "end_failure": END}
    )
    
    db_file = os.getenv("LANGGRAPH_DB_FILE", ":memory:")
    checkpointer = SqliteSaver.from_conn_string(db_file)
    
    app = workflow_builder.compile(checkpointer=checkpointer)
    logging.info(f"Workflow LangGraph compilé avec persistance SqliteSaver ({db_file}).")
    return app

generation_app = create_generation_workflow()

def run_chapter_generation(us_id: str, max_retries: int = 3) -> Dict[str, Any]:
    logging.info(f"Lancement du workflow pour US {us_id} avec max {max_retries} tentatives initiales.")
    try:
        with open("prompts_spec.json", 'r', encoding='utf-8') as f:
            prompts_data = json.load(f)
        us_spec = next((item for item in prompts_data if item["us_id"] == us_id), None)
        if not us_spec:
            msg = f"Spécifications non trouvées pour US ID: {us_id}"
            logging.error(msg)
            return {"us_id": us_id, "status": "error", "error": msg}

        prompt_template = us_spec.get("prompt_template", "")
        acceptance_criteria = us_spec.get("acceptance_criteria", {})
        initial_query = us_spec.get("initial_rag_query", f"Informations clés pour rédiger le chapitre '{us_spec.get('chapter_title', us_id)}' (US: {us_id})")
        
        acceptance_criteria.setdefault("rag_k", 5)
        acceptance_criteria.setdefault("rag_score_threshold", 1.0)
        acceptance_criteria.setdefault("chapter_title", us_id)
        acceptance_criteria.setdefault("prompt_placeholders", {})


    except FileNotFoundError:
        msg = "Fichier prompts_spec.json non trouvé."
        logging.error(msg)
        return {"us_id": us_id, "status": "error", "error": msg}
    except Exception as e:
        msg = f"Erreur lors du chargement des spécifications pour {us_id}: {str(e)}"
        logging.error(msg, exc_info=True)
        return {"us_id": us_id, "status": "error", "error": msg}

    initial_state_dict: GenerationState = {
        "us_id": us_id,
        "prompt_template": prompt_template,
        "acceptance_criteria": acceptance_criteria,
        "initial_query": initial_query,
        "context": None, "current_prompt": "", "generation": None,
        "validation_errors": None, "retries_left": max_retries,
        "final_chapter": None,
    }
    thread_id = f"gen_workflow_{us_id}_{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}
    final_output_map = None

    try:
        for event_output in generation_app.stream(initial_state_dict, config=config, stream_mode="values"):
            final_output_map = event_output
        
        if final_output_map:
            if final_output_map.get("final_chapter"):
                logging.info(f"Workflow terminé avec succès pour US {us_id}.")
                reports_dir = os.getenv("GENERATED_REPORTS_DIR", "./generated_reports")
                os.makedirs(reports_dir, exist_ok=True)
                result_file_path = os.path.join(reports_dir, f"{us_id}_result.json")
                
                report_data_to_save = {
                    "us_id": us_id,
                    "chapter_title": final_output_map.get("acceptance_criteria", {}).get("chapter_title", us_id),
                    "chapter_text": final_output_map["final_chapter"],
                    "status": "success",
                    "generation_details": {
                        "retries_left_at_end": final_output_map.get("retries_left"),
                        "context_used_count": len(final_output_map.get("context") or [])
                    }
                }
                try:
                    with open(result_file_path, 'w', encoding='utf-8') as f:
                        json.dump(report_data_to_save, f, ensure_ascii=False, indent=4)
                    logging.info(f"Rapport pour {us_id} sauvegardé dans {result_file_path}")
                except Exception as e_save:
                    logging.error(f"Erreur lors de la sauvegarde du rapport pour {us_id}: {e_save}")
                return report_data_to_save
            else:
                error_msg = "Max retries reached"
                if final_output_map.get("validation_errors"):
                    error_msg = f"Validation failed: {final_output_map['validation_errors']}"
                logging.error(f"Workflow terminé en échec pour US {us_id}: {error_msg}")
                return {"us_id": us_id, "status": "failed", "error": error_msg, "details": final_output_map}
        else:
            msg = "Aucun état final produit par le workflow."
            logging.error(f"Workflow pour US {us_id} n'a pas produit d'état final valide. {msg}")
            return {"us_id": us_id, "status": "error", "error": msg}
    except Exception as e:
        msg = f"Erreur inattendue lors de l'exécution du workflow pour {us_id}: {str(e)}"
        logging.error(msg, exc_info=True)
        return {"us_id": us_id, "status": "error", "error": msg}