# --- AGENT_VF/orchestrator/workflow.py ---
"""
Définition du workflow LangGraph pour la génération de chapitres.
Remplace l'orchestration basée sur Celery.
"""
import logging
import json
from typing import TypedDict, List, Optional, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver # Pour la persistance en mémoire (simple)
# Alternative: utiliser SqliteSaver, PostgresSaver pour persistance fichier/DB
# from langgraph.checkpoint.sqlite import SqliteSaver

# Importer les composants nécessaires
from rag import retriever
from writer import client as writer_client
from validation import validator
from core.models import LogEntry # Si besoin de manipuler des LogEntry ici

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Définition de l'état du Graphe ---

class GenerationState(TypedDict):
    """État du graphe pour la génération d'un chapitre."""
    us_id: str                      # ID de la User Story
    prompt_template: str            # Template de prompt pour l'US
    acceptance_criteria: Dict[str, Any] # Critères d'acceptation (ex: min_words)
    initial_query: str              # Requête initiale pour le RAG
    context: Optional[List[Dict]]   # Contexte récupéré par le RAG
    current_prompt: str             # Prompt complet envoyé au LLM
    generation: Optional[str]       # Texte généré par le LLM
    validation_errors: Optional[List[str]] # Erreurs de validation
    retries_left: int               # Compteur pour la boucle de réécriture
    final_chapter: Optional[str]    # Chapitre final validé

# --- Nœuds du Graphe ---

def retrieve_context(state: GenerationState) -> GenerationState:
    """Récupère le contexte pertinent pour la User Story."""
    logging.info(f"Noeud RAG: Récupération de contexte pour US {state['us_id']}")
    query = state.get("initial_query", f"Informations pertinentes pour {state['us_id']}")
    k = state.get("rag_k", 5) # Nombre de chunks à récupérer (configurable)
    # Ajouter des filtres si nécessaire, ex: basé sur l'US ID ou des tags
    # filter_criteria = {"us_id": state['us_id']}
    filter_criteria = None

    retrieved_docs = retriever.retrieve(query=query, k=k, filter_criteria=filter_criteria)
    logging.info(f"Noeud RAG: {len(retrieved_docs)} chunks récupérés.")

    return {"context": retrieved_docs}

def construct_prompt(state: GenerationState) -> GenerationState:
    """Construit le prompt final pour le LLM."""
    logging.info(f"Noeud Prompt: Construction du prompt pour US {state['us_id']}")
    template = state["prompt_template"]
    context_str = "\n---\n".join([doc.get("chunk_text", "") for doc in state.get("context", [])])

    # Logique de raffinement du prompt si des erreurs de validation existent
    refinement_prefix = ""
    if state.get("validation_errors"):
        errors = "\n - ".join(state["validation_errors"])
        refinement_prefix = (
            f"La tentative précédente a échoué à la validation pour les raisons suivantes:\n - {errors}\n"
            f"Veuillez corriger ces points et régénérer le chapitre.\n"
            f"Texte précédent (pour référence):\n'''\n{state.get('generation', '')}\n'''\n\n"
            "Nouvelle tentative:\n"
        )

    # Remplacer les placeholders dans le template
    # Ceci est une simplification, une vraie application utiliserait un moteur de template (ex: Jinja2)
    # ou les capacités de templating de LangChain.
    # Pour le stub, on fait une substitution simple.
    # Il faudrait charger les données de company_analysis, mission_context etc. quelque part.
    # Pour l'instant, on injecte juste le contexte RAG.
    final_prompt = template # Utiliser le template chargé initialement
    # Remplacer un placeholder générique comme {{ rag_context }}
    final_prompt = final_prompt.replace("{{ rag_context }}", context_str)
    # Ajouter le préfixe de raffinement si nécessaire
    final_prompt = refinement_prefix + final_prompt

    # Supprimer les placeholders non remplis (simplification)
    import re
    final_prompt = re.sub(r"\{\{\s*\w+\s*\}\}", "[DONNÉE MANQUANTE]", final_prompt)

    logging.info(f"Noeud Prompt: Prompt final construit (longueur: {len(final_prompt)}).")
    return {"current_prompt": final_prompt}

def generate_chapter(state: GenerationState) -> GenerationState:
    """Génère le chapitre en utilisant le client LLM."""
    logging.info(f"Noeud Writer: Génération du chapitre pour US {state['us_id']}")
    prompt = state.get("current_prompt")
    if not prompt:
        logging.error("Aucun prompt à envoyer au LLM.")
        return {"generation": "[Erreur: Prompt manquant]", "validation_errors": ["Prompt manquant pour la génération."]}

    # Utiliser le client Ollama configuré
    generated_text = writer_client.generate(prompt)

    logging.info(f"Noeud Writer: Génération terminée (longueur: {len(generated_text)}).")
    return {"generation": generated_text}

def validate_chapter(state: GenerationState) -> GenerationState:
    """Valide le chapitre généré."""
    logging.info(f"Noeud Validation: Validation du chapitre pour US {state['us_id']}")
    generated_text = state.get("generation")
    criteria = state.get("acceptance_criteria", {})
    min_words = criteria.get("min_words", 50) # Valeur par défaut
    required_sections = criteria.get("required_sections", [])

    errors = []
    if not generated_text or generated_text.startswith("[Erreur"):
        errors.append("La génération a échoué ou retourné une erreur.")
        logging.error("Validation échouée: Génération LLM invalide.")
        return {"validation_errors": errors}

    # Validation de la longueur
    if not validator.check_length(generated_text, min_words):
        errors.append(f"Longueur insuffisante (minimum {min_words} mots).")

    # Validation de la structure
    structure_errors = validator.check_structure(generated_text, required_sections)
    errors.extend(structure_errors)

    if not errors:
        logging.info(f"Noeud Validation: Chapitre pour US {state['us_id']} validé avec succès.")
        return {"validation_errors": [], "final_chapter": generated_text} # Stocker le résultat final
    else:
        logging.warning(f"Noeud Validation: Échec de la validation pour US {state['us_id']}. Erreurs: {errors}")
        return {"validation_errors": errors}

# --- Logique Conditionnelle ---

def should_retry(state: GenerationState) -> str:
    """Décide s'il faut réessayer la génération ou terminer."""
    logging.info(f"Condition: Vérification de la nécessité de réessayer pour US {state['us_id']}")
    errors = state.get("validation_errors")
    retries = state.get("retries_left", 0)

    if errors and retries > 0:
        logging.info(f"Condition: Validation échouée, {retries} tentatives restantes. -> retry")
        # Décrémenter les essais restants avant de boucler
        return "retry" # Nom de l'arc pour réessayer
    elif errors:
        logging.error(f"Condition: Validation échouée et plus de tentatives pour US {state['us_id']}. -> end_failure")
        return "end_failure" # Nom de l'arc pour terminer en échec
    else:
        logging.info(f"Condition: Validation réussie pour US {state['us_id']}. -> end_success")
        return "end_success" # Nom de l'arc pour terminer avec succès

# --- Construction du Graphe ---

def create_generation_workflow():
    """Crée et compile le workflow LangGraph."""
    workflow = StateGraph(GenerationState)

    # Ajouter les nœuds
    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("construct_prompt", construct_prompt)
    workflow.add_node("generate_chapter", generate_chapter)
    workflow.add_node("validate_chapter", validate_chapter)

    # Définir le point d'entrée
    workflow.set_entry_point("retrieve_context")

    # Ajouter les arêtes
    workflow.add_edge("retrieve_context", "construct_prompt")
    workflow.add_edge("construct_prompt", "generate_chapter")
    workflow.add_edge("generate_chapter", "validate_chapter")

    # Ajouter l'arête conditionnelle pour la boucle de validation/réécriture
    workflow.add_conditional_edges(
        "validate_chapter",
        should_retry,
        {
            "retry": "construct_prompt", # Retourner à la construction du prompt pour raffinement
            "end_success": END,          # Terminer si succès
            "end_failure": END           # Terminer si échec après max retries
        }
    )

    # Compiler le graphe avec persistance en mémoire (pour commencer)
    # Pour une persistance plus robuste, utiliser SqliteSaver ou PostgresSaver
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    logging.info("Workflow LangGraph compilé avec persistance en mémoire.")
    return app

# Instance du graphe compilé (peut être initialisée au démarrage de l'API/worker)
generation_app = create_generation_workflow()

# --- Fonction pour exécuter le workflow ---

def run_chapter_generation(us_id: str, max_retries: int = 3) -> Optional[Dict[str, Any]]:
    """
    Exécute le workflow de génération pour une US donnée.

    Args:
        us_id (str): L'ID de la User Story.
        max_retries (int): Nombre maximum de tentatives de réécriture.

    Returns:
        Optional[Dict[str, Any]]: Le résultat final du workflow ou None si échec.
    """
    logging.info(f"Lancement du workflow pour US {us_id} avec max {max_retries} tentatives.")

    # Charger les spécifications du prompt et les critères pour l'US
    try:
        with open("prompts_spec.json", 'r', encoding='utf-8') as f:
            prompts_data = json.load(f)
        us_spec = next((item for item in prompts_data if item["us_id"] == us_id), None)

        if not us_spec:
            logging.error(f"Spécifications non trouvées pour US ID: {us_id}")
            return None

        prompt_template = us_spec.get("prompt_template", "")
        acceptance_criteria = us_spec.get("acceptance_criteria", {})
        initial_query = f"Informations clés pour rédiger le chapitre '{us_spec.get('chapter', us_id)}' (US: {us_id})"

    except FileNotFoundError:
        logging.error("Fichier prompts_spec.json non trouvé.")
        return None
    except Exception as e:
        logging.error(f"Erreur lors du chargement des spécifications pour {us_id}: {e}", exc_info=True)
        return None

    initial_state: GenerationState = {
        "us_id": us_id,
        "prompt_template": prompt_template,
        "acceptance_criteria": acceptance_criteria,
        "initial_query": initial_query,
        "context": None,
        "current_prompt": "",
        "generation": None,
        "validation_errors": None,
        "retries_left": max_retries,
        "final_chapter": None,
    }

    # Utiliser un ID de thread unique pour chaque exécution pour la persistance
    config = {"configurable": {"thread_id": f"gen_{us_id}_{uuid.uuid4()}"}}

    final_state = None
    try:
        # Exécuter le graphe de manière synchrone pour cet exemple
        # Pour une exécution asynchrone (ex: dans FastAPI), utiliser .astream() ou .ainvoke()
        for event in generation_app.stream(initial_state, config=config):
            # On peut inspecter les événements ici si besoin
            # print(f"Event: {event}")
            if END in event:
                final_state = event[END]
                break # Sortir de la boucle une fois terminé

        if final_state and final_state.get("final_chapter"):
            logging.info(f"Workflow terminé avec succès pour US {us_id}.")
            return {"us_id": us_id, "chapter_text": final_state["final_chapter"], "status": "success"}
        else:
            logging.error(f"Workflow terminé en échec pour US {us_id}.")
            error_msg = "Max retries reached" if final_state else "Unknown error"
            if final_state and final_state.get("validation_errors"):
                 error_msg = f"Validation failed: {final_state['validation_errors']}"
            return {"us_id": us_id, "status": "failed", "error": error_msg}

    except Exception as e:
        logging.error(f"Erreur inattendue lors de l'exécution du workflow pour {us_id}: {e}", exc_info=True)
        return {"us_id": us_id, "status": "error", "error": str(e)}

# End of file