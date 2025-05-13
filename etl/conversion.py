# --- AGENT_VF/etl/conversion.py ---
"""
Wrapper pour la conversion de documents bruts (y compris DOCX) en texte.
"""
import os
import logging
from typing import Optional, Tuple, Dict, List

# Importer les bibliothèques nécessaires
try:
    import docx # python-docx
except ImportError:
    logging.warning("python-docx non trouvé. La conversion DOCX échouera. Installer avec 'pip install python-docx'")
    docx = None

try:
    import fitz # PyMuPDF
except ImportError:
    logging.warning("PyMuPDF non trouvé. La conversion PDF échouera. Installer avec 'pip install PyMuPDF'")
    fitz = None

# --- Configuration du Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Fonctions de Conversion Spécifiques ---

def convert_docx_to_text(file_path: str) -> str:
    """Convertit un fichier DOCX en texte brut."""
    if not docx:
        raise ImportError("Le module 'docx' (python-docx) est requis pour lire les fichiers .docx.")
    try:
        doc = docx.Document(file_path)
        full_text = [para.text for para in doc.paragraphs]
        # Joindre avec double saut de ligne pour préserver la structure des paragraphes
        return '\n\n'.join(full_text)
    except Exception as e:
        logging.error(f"Erreur lors de la conversion de {file_path} (DOCX): {e}")
        raise # Propage l'erreur

def convert_pdf_to_text(file_path: str) -> str:
    """Convertit un fichier PDF en texte brut en utilisant PyMuPDF."""
    if not fitz:
         raise ImportError("Le module 'fitz' (PyMuPDF) est requis pour lire les fichiers .pdf.")
    full_text = []
    try:
        with fitz.open(file_path) as doc:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                full_text.append(page.get_text("text"))
        combined_text = '\n\n'.join(full_text).strip() # Utiliser double saut de ligne
        if len(combined_text) < 50:
             logging.warning(f"Peu de texte extrait de {file_path}. Est-ce un PDF image?")
        return combined_text
    except Exception as e:
        if "password" in str(e).lower():
             logging.error(f"Le fichier PDF {file_path} est protégé par mot de passe.")
        else:
            logging.error(f"Erreur lors de la conversion de {file_path} (PDF): {e}")
        raise

def read_text_file(file_path: str) -> str:
    """Lit un fichier texte brut (txt, md)."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logging.error(f"Erreur lors de la lecture de {file_path} (Text): {e}")
        raise

# --- Fonction Principale de Conversion (Wrapper) ---

def run_conversion(file_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Exécute le script de conversion approprié pour un fichier donné.

    Args:
        file_path (str): Le chemin vers le fichier à convertir.

    Returns:
        Tuple[Optional[str], Optional[str]]: Le texte extrait et le nom du convertisseur utilisé,
                                             ou (None, None) en cas d'échec ou format non supporté.
    """
    logging.info(f"Tentative de conversion pour: {file_path}")
    _, file_extension = os.path.splitext(file_path.lower())
    text_content = None
    converter_used = None

    try:
        if file_extension == '.docx':
            text_content = convert_docx_to_text(file_path)
            converter_used = 'python-docx'
        elif file_extension == '.pdf':
            text_content = convert_pdf_to_text(file_path)
            converter_used = 'PyMuPDF'
        elif file_extension in ['.txt', '.md']:
            text_content = read_text_file(file_path)
            converter_used = 'direct_read'
        else:
            logging.warning(f"Format de fichier non supporté: {file_extension} pour {file_path}")
            return None, None # Format non géré

        logging.info(f"Fichier {file_path} converti avec succès via {converter_used}.")
        return text_content, converter_used

    except ImportError as e:
         logging.error(f"Dépendance manquante pour convertir {file_path}: {e}")
         return None, None
    except Exception as e:
        # Les erreurs spécifiques sont déjà loggées dans les fonctions filles
        logging.error(f"Échec final de la conversion pour {file_path}: {e}", exc_info=False) # exc_info=False pour éviter redondance
        return None, None # Indique un échec

# --- Fonction Utilitaire pour Traiter un Répertoire ---

def process_directory(directory_path: str, extensions: List[str] = ['.docx', '.pdf', '.txt', '.md']) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    """
    Scanne un répertoire et tente de convertir les fichiers avec les extensions spécifiées.

    Args:
        directory_path (str): Chemin du répertoire à scanner.
        extensions (List[str]): Liste des extensions de fichiers à traiter (en minuscules).

    Returns:
        Dict[str, Tuple[Optional[str], Optional[str]]]: Dictionnaire mappant les chemins
            des fichiers traités aux résultats de run_conversion (texte, convertisseur).
            Inclut les échecs avec (None, None).
    """
    results = {}
    logging.info(f"Scan du répertoire '{directory_path}' pour les fichiers {extensions}...")
    if not os.path.isdir(directory_path):
        logging.error(f"Le répertoire spécifié n'existe pas: {directory_path}")
        return results

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            _, file_extension = os.path.splitext(filename.lower())
            if file_extension in extensions:
                results[file_path] = run_conversion(file_path)
            else:
                logging.debug(f"Ignoré (extension non supportée): {filename}")
        else:
            logging.debug(f"Ignoré (n'est pas un fichier): {filename}")

    logging.info(f"Scan terminé. {len(results)} fichiers traités dans '{directory_path}'.")
    return results

# --- Exemple d'utilisation (peut être appelé par l'orchestrateur ou un script principal) ---
# if __name__ == '__main__':
#     # Assurez-vous que le répertoire data/logs existe et contient des fichiers
#     log_directory = "../data/logs" # Adaptez le chemin relatif si nécessaire
#     if not os.path.exists(log_directory):
#         os.makedirs(log_directory)
#         # Créez quelques fichiers dummy pour tester si besoin
#         with open(os.path.join(log_directory, "test.txt"), "w") as f: f.write("Contenu texte.")
#         # Ajoutez un fichier .docx vide ou simple nommé test.docx dans data/logs
#         # Ajoutez un fichier .pdf vide ou simple nommé test.pdf dans data/logs

#     conversion_results = process_directory(log_directory)
#     print("\n--- Résultats de la Conversion du Répertoire ---")
#     for f_path, (text, conv) in conversion_results.items():
#         status = "Échec" if text is None else f"Succès ({conv})"
#         print(f"- {os.path.basename(f_path)}: {status}")
#         # if text:
#         #     print(f"  Extrait: {text[:50]}...")

# End of file