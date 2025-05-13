# --- AGENT_VF/etl/cleaning.py ---
"""
Wrapper pour le script de nettoyage de texte.
"""
import logging

# Supposons que vos scripts originaux sont accessibles
try:
    from etl_scripts import cleaning as original_cleaning
except ImportError:
    logging.warning("Impossible d'importer 'etl_scripts.cleaning'.")
    # Fonction factice
    class original_cleaning:
        @staticmethod
        def clean_text(raw_text: str) -> str:
            logging.warning("Fonction factice 'clean_text' appelée.")
            return raw_text.strip() if raw_text else ""

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_cleaning(raw_text: str) -> str:
    """
    Exécute le script de nettoyage sur le texte brut fourni.

    Args:
        raw_text (str): Le texte brut à nettoyer.

    Returns:
        str: Le texte nettoyé.
    """
    logging.info("Lancement du nettoyage de texte.")
    if not raw_text:
        logging.warning("Texte brut vide fourni pour le nettoyage.")
        return ""
    try:
        # Appel de la fonction principale de votre script original
        cleaned_text = original_cleaning.clean_text(raw_text)
        logging.info("Nettoyage de texte terminé.")
        return cleaned_text
    except Exception as e:
        logging.error(f"Erreur inattendue lors du nettoyage: {e}", exc_info=True)
        # Retourner le texte brut en cas d'erreur pour ne pas bloquer ? Ou lever l'erreur ?
        # Pour un stub, on retourne le texte brut.
        return raw_text

# End of file