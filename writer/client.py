"""
Wrapper pour le client LLM local via Ollama.
"""
import logging
import os
from typing import List, Dict, Any, Optional
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = "gemma:7b"
LLM_MODEL_NAME = os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)
ollama_client_instance = None

def get_ollama_client():
    """Initialise ou retourne le client ChatOllama."""
    global ollama_client_instance
    
    if ollama_client_instance is None:
        try:
            logging.info(f"Initialisation du client ChatOllama: base_url='{OLLAMA_BASE_URL}', model='{LLM_MODEL_NAME}'")
            ollama_client_instance = ChatOllama(
                base_url=OLLAMA_BASE_URL,
                model=LLM_MODEL_NAME,
                temperature=0.7,
            )
            
            try:
                logging.info(f"Test de connexion au serveur Ollama et au modèle {LLM_MODEL_NAME}...")
                ollama_client_instance.invoke([HumanMessage(content="Qui es-tu?")])
                logging.info("Client ChatOllama initialisé et connexion testée avec succès.")
            except Exception as conn_test_e:
                logging.error(f"Test de connexion Ollama échoué: {conn_test_e}", exc_info=False)
                logging.error(f"Vérifiez que le serveur Ollama est lancé ({OLLAMA_BASE_URL}) et que le modèle '{LLM_MODEL_NAME}' est disponible (ollama list).")
        except Exception as e:
            logging.error(f"Erreur majeure lors de l'initialisation du client ChatOllama: {e}", exc_info=True)
            ollama_client_instance = None
            
    if ollama_client_instance is None:
        raise RuntimeError(f"Client Ollama n'a pas pu être initialisé. Vérifiez les logs. Base URL: {OLLAMA_BASE_URL}, Model: {LLM_MODEL_NAME}")
        
    return ollama_client_instance

def generate(prompt: str, system_prompt: Optional[str] = "You are a helpful assistant.") -> str:
    """
    Génère du texte à partir d'un prompt en utilisant le LLM local via Ollama.
    """
    logging.info(f"Génération de texte (Ollama) pour le prompt: '{prompt[:100]}...'")
    
    if not prompt:
        logging.warning("Prompt vide fourni au générateur Ollama.")
        return ""
        
    try:
        client = get_ollama_client()
    except RuntimeError as e:
        logging.error(f"Impossible d'obtenir le client Ollama: {e}")
        return f"[Erreur: Client Ollama non disponible - {e}]"

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
        return f"[Erreur lors de la génération Ollama: {str(e)}]"