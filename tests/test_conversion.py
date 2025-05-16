# --- AGENT_VF/etl/conversion.py ---
"""
Wrapper pour la conversion de documents bruts (y compris DOCX) en texte.
Inclut une fonction pour utiliser 'unstructured' si nécessaire.
"""
import os
import logging
from typing import Optional, Tuple, Dict, List, Any
import mimetypes 

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

try:
    from unstructured.partition.auto import partition as unstructured_partition_auto
    from unstructured.documents.elements import Element, Image as UnstructuredImage
except ImportError:
    logging.warning("unstructured non trouvé. La conversion via unstructured échouera. Installer avec 'pip install \"unstructured[local-inference]\"' ou une variante adaptée.")
    unstructured_partition_auto = None
    UnstructuredImage = None # type: ignore # Pour que le code ne plante pas si unstructured n'est pas là
    Element = type('Element', (object,), {}) # type: ignore # Dummy class pour type hinting


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_docx_to_text_pydocx(file_path: str) -> str:
    """Convertit un fichier DOCX en texte brut."""
    if not docx:
        raise ImportError("Le module 'docx' (python-docx) est requis pour lire les fichiers .docx.")
    try:
        doc = docx.Document(file_path)
        full_text = [para.text for para in doc.paragraphs]
        return '\n\n'.join(full_text)
    except Exception as e:
        logging.error(f"Erreur lors de la conversion de {file_path} (DOCX via python-docx): {e}")
        raise

def convert_pdf_to_text_pymupdf(file_path: str) -> str:
    """Convertit un fichier PDF en texte brut en utilisant PyMuPDF."""
    if not fitz:
         raise ImportError("Le module 'fitz' (PyMuPDF) est requis pour lire les fichiers .pdf.")
    full_text = []
    try:
        with fitz.open(file_path) as doc:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                full_text.append(page.get_text("text"))
        combined_text = '\n\n'.join(full_text).strip()
        if len(combined_text) < 50:
             logging.warning(f"Peu de texte extrait de {file_path} (longueur: {len(combined_text)} via PyMuPDF). Est-ce un PDF image ou vide?")
        return combined_text
    except Exception as e:
        if "password" in str(e).lower():
             logging.error(f"Le fichier PDF {file_path} est protégé par mot de passe (PyMuPDF).")
        else:
            logging.error(f"Erreur lors de la conversion de {file_path} (PDF via PyMuPDF): {e}")
        raise

def read_text_file(file_path: str) -> str:
    """Lit un fichier texte brut (txt, md)."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logging.error(f"Erreur lors de la lecture de {file_path} (Text): {e}")
        raise

def convert_with_unstructured(file_path: str, unstructured_output_dir: Optional[str] = None) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    Convertit un fichier en utilisant unstructured.partition.auto.partition.
    Retourne le texte concaténé et une liste de dictionnaires représentant les éléments Unstructured.
    Si unstructured_output_dir est fourni, tente de sauvegarder les images extraites.
    """
    if not unstructured_partition_auto or not UnstructuredImage or not Element:
        logging.error("La bibliothèque 'unstructured' ou ses composants (Element, Image) ne sont pas correctement importés/installés.")
        return None, []
    
    elements_as_dicts: List[Dict[str, Any]] = []
    full_text_parts: List[str] = []
    
    try:
        logging.info(f"Tentative de conversion avec unstructured pour: {file_path}")
        
        # La stratégie d'extraction d'images peut nécessiter des paramètres supplémentaires pour `partition`
        # comme strategy="hi_res" et des dépendances comme `detectron2`.
        # Ici, nous utilisons la stratégie par défaut et vérifions les éléments Image.
        elements: List[Element] = unstructured_partition_auto(filename=file_path) # type: ignore

        image_count = 0
        output_dir_for_images = None
        if unstructured_output_dir:
            output_dir_for_images = os.path.join(unstructured_output_dir, "images_unstructured_output") # Nom de sous-dossier plus explicite
            os.makedirs(output_dir_for_images, exist_ok=True)
            logging.info(f"Répertoire pour images (si applicable par unstructured) créé/vérifié: {output_dir_for_images}")

        for i, element in enumerate(elements):
            element_dict = element.to_dict() 
            elements_as_dicts.append(element_dict)
            
            if hasattr(element, 'text') and element.text:
                full_text_parts.append(element.text)

            if UnstructuredImage and isinstance(element, UnstructuredImage) and output_dir_for_images:
                # Tenter de sauvegarder l'image
                img_bytes = getattr(element, 'image_bytes', None)
                img_format = getattr(element, 'image_format', None)

                if img_bytes:
                    if not img_format: # Essayer de deviner le format
                        # Utiliser le nom de fichier original de l'image si disponible dans les métadonnées
                        original_img_filename = element.metadata.filename if hasattr(element, 'metadata') and hasattr(element.metadata, 'filename') else None
                        if original_img_filename:
                             _, ext = os.path.splitext(original_img_filename)
                             img_format = ext.lstrip('.') if ext else 'png' # Fallback
                        else: # Fallback si pas de nom de fichier original
                            img_format = 'png' 
                    
                    image_filename = f"image_{os.path.basename(file_path)}_{image_count}_{i}.{img_format}"
                    image_path = os.path.join(output_dir_for_images, image_filename)
                    try:
                        with open(image_path, "wb") as f:
                            f.write(img_bytes)
                        logging.info(f"Image extraite sauvegardée: {image_path}")
                        image_count += 1
                    except Exception as e_img:
                        logging.error(f"Erreur lors de la sauvegarde de l'image {image_path}: {e_img}")
                else:
                    logging.warning(f"Élément Image détecté mais image_bytes manquant ou vide pour l'élément {i} de {file_path}")

        concatenated_text = "\n\n".join(full_text_parts).strip()

        if not concatenated_text and not elements_as_dicts:
            logging.warning(f"Unstructured n'a retourné aucun texte ni élément pour {file_path}.")
            return None, [] # Retourne None pour le texte si rien n'est extrait
            
        logging.info(f"Conversion avec unstructured réussie pour {file_path}. {len(elements_as_dicts)} éléments trouvés, {image_count} images sauvegardées.")
        return concatenated_text, elements_as_dicts

    except Exception as e:
        logging.error(f"Erreur lors de la conversion avec unstructured pour {file_path}: {e}", exc_info=True)
        return None, []


def run_conversion(
    file_path: str,
    use_unstructured: bool = False, 
    unstructured_output_dir: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[str]]:
    """
    Exécute le script de conversion approprié pour un fichier donné.
    Retourne le texte concaténé et le nom du convertisseur.
    """
    logging.info(f"Tentative de conversion pour: {file_path} (use_unstructured: {use_unstructured}, unstructured_output_dir: {unstructured_output_dir})")
    _, file_extension = os.path.splitext(file_path.lower())
    text_content: Optional[str] = None
    converter_used: Optional[str] = None
    # raw_unstructured_elements: List[Dict[str, Any]] = [] # Non retourné par cette fonction pour l'instant

    try:
        if use_unstructured and unstructured_partition_auto:
            logging.info(f"Priorisation de Unstructured pour {file_path} car use_unstructured=True.")
            text_content, _ = convert_with_unstructured(file_path, unstructured_output_dir)
            if text_content is not None: # Unstructured peut retourner une chaîne vide si le doc est vide mais parsé
                 converter_used = 'unstructured'
            else:
                 logging.warning(f"Conversion avec unstructured (prioritaire) a échoué ou n'a retourné aucun texte pour {file_path}.")
                 return None, None # Échec clair si unstructured était prioritaire et n'a rien retourné
        elif file_extension == '.docx':
            text_content = convert_docx_to_text_pydocx(file_path)
            converter_used = 'pydocx' # Standardisation du nom
        elif file_extension == '.pdf':
            text_content = convert_pdf_to_text_pymupdf(file_path)
            converter_used = 'PyMuPDF'
        elif file_extension in ['.txt', '.md']:
            text_content = read_text_file(file_path)
            converter_used = 'direct_read'
        elif unstructured_partition_auto and is_unstructured_compatible(file_path): # Fallback
            logging.info(f"Tentative de fallback avec Unstructured pour {file_path}.")
            text_content, _ = convert_with_unstructured(file_path, unstructured_output_dir)
            if text_content is not None:
                 converter_used = 'unstructured'
            else:
                 logging.warning(f"Conversion avec unstructured (fallback) a échoué ou n'a retourné aucun texte pour {file_path}.")
                 return None, None
        else:
            logging.warning(f"Format de fichier non supporté ou convertisseur 'unstructured' non disponible: {file_extension} pour {file_path}")
            return None, None

        if text_content is not None:
            logging.info(f"Fichier {file_path} converti avec succès via {converter_used}.")
        
        return text_content, converter_used

    except ImportError as e:
         logging.error(f"Dépendance manquante pour convertir {file_path}: {e}")
         return None, None
    except Exception as e:
        logging.error(f"Échec final de la conversion pour {file_path}: {e}", exc_info=False)
        return None, None

def process_directory(
    directory_path: str,
    extensions: Optional[List[str]] = None,
    use_unstructured_processing: bool = False, 
    unstructured_base_output_dir: Optional[str] = None
    ) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    """
    Scanne un répertoire et tente de convertir les fichiers.
    """
    if extensions is None:
        extensions = ['.docx', '.pdf', '.txt', '.md'] 
        
    results = {}
    logging.info(f"Scan du répertoire '{directory_path}'. use_unstructured_processing: {use_unstructured_processing}, unstructured_base_output_dir: {unstructured_base_output_dir}")

    if not os.path.isdir(directory_path):
        logging.error(f"Le répertoire spécifié n'existe pas: {directory_path}")
        return results

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            text_content, converter = run_conversion(
                file_path,
                use_unstructured=use_unstructured_processing,
                unstructured_output_dir=unstructured_base_output_dir 
            )
            results[file_path] = (text_content, converter) # Stocker même si None pour tracer les échecs
        else:
            logging.debug(f"Ignoré (n'est pas un fichier): {filename}")
    
    successful_conversions = sum(1 for text, _ in results.values() if text is not None)
    logging.info(f"Scan terminé. {successful_conversions}/{len(results)} fichiers dont le contenu a pu être extrait dans '{directory_path}'.")
    return results