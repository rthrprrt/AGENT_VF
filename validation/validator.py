# --- AGENT_VF/validation/validator.py ---
"""
Module pour valider le contenu généré (structure, longueur, etc.).
Peut être utilisé comme Tool LangChain si nécessaire.
"""
import logging
import re
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool # Optionnel: pour transformer en Tool

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# @tool # Décommenter pour utiliser comme Tool LangChain
def check_structure(text: str, required_sections: Optional[List[str]] = None) -> List[str]:
    """
    Vérifie si la structure attendue (ex: sections spécifiques H1/H2) est présente.

    Args:
        text (str): Le texte généré à vérifier.
        required_sections (Optional[List[str]]): Liste des titres de section requis.
                                                 Le matching est simple et insensible à la casse.

    Returns:
        List[str]: Une liste de messages d'erreur décrivant les problèmes de structure.
                   Liste vide si la structure est valide.
    """
    logging.info(f"Vérification de la structure du texte. Sections requises: {required_sections}")
    errors = []
    if not text:
        errors.append("Le texte est vide.")
        return errors

    if required_sections:
        text_lower = text.lower()
        for section_title in required_sections:
            # Recherche simple de titres (peut être améliorée avec regex plus précis)
            # Ex: chercher '# titre' ou '## titre'
            title_pattern_h1 = r"^#\s+" + re.escape(section_title.lower())
            title_pattern_h2 = r"^##\s+" + re.escape(section_title.lower())
            if not (re.search(title_pattern_h1, text_lower, re.MULTILINE) or
                    re.search(title_pattern_h2, text_lower, re.MULTILINE)):
                errors.append(f"Section manquante ou titre mal formaté: '{section_title}'")

    # Ajouter d'autres vérifications structurelles si nécessaire (ex: présence de listes, tableaux...)

    if not errors:
        logging.info("Structure du texte validée avec succès.")
    else:
        logging.warning(f"Problèmes de structure détectés: {errors}")

    return errors

# @tool # Décommenter pour utiliser comme Tool LangChain
def check_length(text: str, min_words: int) -> bool:
    """
    Vérifie si le texte atteint un nombre minimum de mots.

    Args:
        text (str): Le texte généré.
        min_words (int): Le nombre minimum de mots requis.

    Returns:
        bool: True si le texte a au moins min_words mots, False sinon.
    """
    logging.info(f"Vérification de la longueur du texte (minimum {min_words} mots).")
    if not text:
        logging.warning("Texte vide fourni pour la vérification de longueur.")
        return False

    # Simple split par espace pour compter les mots
    word_count = len(re.findall(r'\b\w+\b', text))
    is_long_enough = word_count >= min_words

    if is_long_enough:
        logging.info(f"Longueur du texte validée ({word_count} mots).")
    else:
        logging.warning(f"Longueur du texte insuffisante ({word_count} mots, minimum {min_words}).")

    return is_long_enough

# End of file