# --- AGENT_VF/writer/client.py ---
"""
Wrapper pour le client LLM local via Ollama.
"""
import logging
import os
from typing import List, Dict, Any, Optional # <--- Vérifier que cet import est bien présent et en premier (après les commentaires/docstring)

# Importer le client Ollama de LangChain et les types de messages
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Lire l'URL de base d'Ollama depuis les variables d'environnement ou utiliser localhost par défaut
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
# Spécifier le modèle Gemma 7B (ou un autre modèle local disponible via Ollama)
DEFAULT_MODEL = "gemma:7b" # Fallback plus courant
LLM_MODEL_NAME = os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)

# --- Initialisation du client (Lazy) ---
ollama_client = None

def get_ollama_client():
    """Initialise ou retourne le client ChatOllama."""
    global ollama_client
    if ollama_client is None:
        try:
            logging.info(f"Initialisation du client ChatOllama: base_url='{OLLAMA_BASE_URL}', model='{LLM_MODEL_NAME}'")
            ollama_client = ChatOllama(
                base_url=OLLAMA_BASE_URL,
                model=LLM_MODEL_NAME,
                temperature=0.7, # Ajuster la température si nécessaire
            )
            logging.info("Client ChatOllama initialisé avec succès.")
        except Exception as e:
            logging.error(f"Erreur lors de l'initialisation du client ChatOllama: {e}", exc_info=True)
            logging.error(f"Vérifiez que le serveur Ollama est lancé ({OLLAMA_BASE_URL}) et que le modèle '{LLM_MODEL_NAME}' est disponible (`ollama list`).")
            raise SystemExit("Échec critique: Impossible d'initialiser le client LLM Ollama.")
    return ollama_client

# Ligne 49 (ou proche) où l'erreur se produit
def generate(prompt: str, system_prompt: Optional[str] = "You are a helpful assistant.") -> str:
    """
    Génère du texte à partir d'un prompt en utilisant le LLM local via Ollama.

    Args:
        prompt (str): Le prompt utilisateur.
        system_prompt (Optional[str]): Le prompt système (instructions générales).

    Returns:
        str: Le texte généré par le LLM.
    """
    logging.info(f"Génération de texte (Ollama) pour le prompt: '{prompt[:100]}...'")
    if not prompt:
        logging.warning("Prompt vide fourni au générateur Ollama.")
        return ""

    client = get_ollama_client()
    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=prompt))

    try:
        response = client.invoke(messages)
        generated_text = response.content
        logging.info("Génération de texte (Ollama) réussie.")
        return generated_text

    except Exception as e:
        logging.error(f"Erreur lors de la génération de texte avec Ollama: {e}", exc_info=True)
        return f"[Erreur lors de la génération Ollama: {e}]"

# End of file