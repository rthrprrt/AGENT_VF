# --- AGENT_VF/etl/conversion.py ---
"""
Wrapper pour la conversion de documents bruts (y compris DOCX) en texte.
Inclut une fonction pour utiliser 'unstructured' si nécessaire.
"""
import os
import logging
from typing import Optional, Tuple, Dict, List, Any
import mimetypes # Pour la sauvegarde d'images avec unstructured

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
    # Image est importé pour vérifier le type d'élément et accéder à image_bytes
except ImportError:
    logging.warning("unstructured non trouvé. La conversion via unstructured échouera. Installer avec 'pip install \"unstructured[local-inference]\"' ou une variante adaptée.")
    unstructured_partition_auto = None
    UnstructuredImage = None # type: ignore
    Element = None # type: ignore


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
    Retourne le texte concaténé et une liste d'éléments structurés (dictionnaires).
    Si unstructured_output_dir est fourni, tente de sauvegarder les images extraites.
    """
    if not unstructured_partition_auto or not UnstructuredImage or not Element:
        logging.error("La bibliothèque 'unstructured' n'est pas correctement installée ou ses composants n'ont pu être importés.")
        return None, []
    
    elements_as_dicts: List[Dict[str, Any]] = []
    full_text_parts: List[str] = []
    
    try:
        logging.info(f"Tentative de conversion avec unstructured pour: {file_path}")
        
        # Unstructured peut nécessiter des stratégies spécifiques pour l'extraction d'images.
        # Par défaut, partition() se concentre sur le texte. Pour les images, 'hi_res' est souvent utilisé.
        # Pour cet exemple, nous allons utiliser la partition par défaut et vérifier si des éléments Image sont produits.
        elements: List[Element] = unstructured_partition_auto(filename=file_path) # type: ignore

        image_count = 0
        if unstructured_output_dir:
            output_dir_for_images = os.path.join(unstructured_output_dir, "images")
            os.makedirs(output_dir_for_images, exist_ok=True)
            logging.info(f"Répertoire pour images (si applicable par unstructured) créé/vérifié: {output_dir_for_images}")

        for i, element in enumerate(elements):
            element_dict = element.to_dict() # Convertit l'élément Unstructured en dictionnaire
            elements_as_dicts.append(element_dict)
            
            if hasattr(element, 'text') and element.text:
                full_text_parts.append(element.text)

            # Gestion de la sauvegarde d'images si l'élément est de type Image
            # et que unstructured_output_dir est fourni.
            if UnstructuredImage and isinstance(element, UnstructuredImage) and unstructured_output_dir:
                if hasattr(element, 'image_bytes') and element.image_bytes: # type: ignore
                    image_bytes = element.image_bytes # type: ignore
                    image_format = getattr(element, 'image_format', None) # type: ignore
                    
                    if not image_format and hasattr(element, 'metadata') and hasattr(element.metadata, 'filename'):
                        _, ext = os.path.splitext(element.metadata.filename) # type: ignore
                        image_format = ext.lstrip('.') if ext else 'png' # Fallback
                    elif not image_format:
                         # Essayer de deviner à partir des bytes si possible, sinon fallback
                        content_type = mimetypes.guess_type(f"dummy.{image_format}")[0] if image_format else None
                        if not content_type and image_bytes:
                            # Simple magic number check (très basique)
                            if image_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
                                image_format = 'png'
                            elif image_bytes.startswith(b'\xff\xd8\xff'):
                                image_format = 'jpg'
                        image_format = image_format or 'png' # Default to png

                    image_filename = f"image_{image_count}_{i}.{image_format}"
                    image_path = os.path.join(output_dir_for_images, image_filename)
                    try:
                        with open(image_path, "wb") as f:
                            f.write(image_bytes)
                        logging.info(f"Image extraite sauvegardée: {image_path}")
                        image_count += 1
                    except Exception as e_img:
                        logging.error(f"Erreur lors de la sauvegarde de l'image {image_path}: {e_img}")
                else:
                    logging.warning(f"Élément Image détecté mais image_bytes manquant ou vide pour l'élément {i} de {file_path}")

        concatenated_text = "\n\n".join(full_text_parts).strip()

        if not concatenated_text and not elements_as_dicts: # Si rien n'a été extrait
            logging.warning(f"Unstructured n'a retourné aucun texte ni élément pour {file_path}.")
            return None, []
            
        logging.info(f"Conversion avec unstructured réussie pour {file_path}. {len(elements_as_dicts)} éléments trouvés, {image_count} images sauvegardées.")
        return concatenated_text, elements_as_dicts # Retourne le texte et la liste des dicts d'éléments

    except Exception as e:
        logging.error(f"Erreur lors de la conversion avec unstructured pour {file_path}: {e}", exc_info=True)
        return None, []


def run_conversion(
    file_path: str,
    use_unstructured: bool = False, 
    unstructured_output_dir: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[str]]: # Retourne (texte_concaténé, nom_convertisseur)
    """
    Exécute le script de conversion approprié pour un fichier donné.
    Utilise `use_unstructured` pour prioriser Unstructured si disponible.
    Retourne le texte concaténé des éléments si Unstructured est utilisé.
    """
    logging.info(f"Tentative de conversion pour: {file_path} (use_unstructured: {use_unstructured}, unstructured_output_dir: {unstructured_output_dir})")
    _, file_extension = os.path.splitext(file_path.lower())
    text_content: Optional[str] = None
    converter_used: Optional[str] = None
    # unstructured_elements: List[Dict[str, Any]] = [] # Non retourné directement par run_conversion pour l'instant

    try:
        if use_unstructured and unstructured_partition_auto:
            logging.info(f"Priorisation de Unstructured pour {file_path} car use_unstructured=True.")
            text_content, _ = convert_with_unstructured(file_path, unstructured_output_dir) # _ pour les éléments bruts
            if text_content is not None: # Peut être une chaîne vide si le doc est vide mais parsé
                 converter_used = 'unstructured'
            else:
                 logging.warning(f"Conversion avec unstructured (prioritaire) a échoué ou n'a retourné aucun texte pour {file_path}.")
                 return None, None
        elif file_extension == '.docx':
            text_content = convert_docx_to_text_pydocx(file_path)
            converter_used = 'pydocx' # Standardisé le nom du convertisseur
        elif file_extension == '.pdf':
            text_content = convert_pdf_to_text_pymupdf(file_path)
            converter_used = 'PyMuPDF'
        elif file_extension in ['.txt', '.md']:
            text_content = read_text_file(file_path)
            converter_used = 'direct_read'
        elif unstructured_partition_auto: # Fallback à unstructured si disponible
            logging.info(f"Tentative de fallback avec Unstructured pour {file_path}.")
            text_content, _ = convert_with_unstructured(file_path, unstructured_output_dir) # _ pour les éléments bruts
            if text_content is not None:
                 converter_used = 'unstructured'
            else:
                 logging.warning(f"Conversion avec unstructured (fallback) a échoué ou n'a retourné aucun texte pour {file_path}.")
                 return None, None
        else:
            logging.warning(f"Format de fichier non supporté ou convertisseur 'unstructured' non disponible: {file_extension} pour {file_path}")
            return None, None

        if text_content is not None: # Peut être une chaîne vide
            logging.info(f"Fichier {file_path} converti avec succès via {converter_used}.")
        # Si text_content est None ici, cela signifie qu'un convertisseur a été tenté mais a échoué à retourner du texte.
        # Cela est déjà loggé par la fonction de conversion spécifique ou par le bloc unstructured.
        
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
            # On stocke le résultat même si text_content est None pour tracer les échecs
            results[file_path] = (text_content, converter)
        else:
            logging.debug(f"Ignoré (n'est pas un fichier): {filename}")

    successful_conversions = sum(1 for text, _ in results.values() if text is not None)
    logging.info(f"Scan terminé. {successful_conversions}/{len(results)} fichiers convertis avec succès dans '{directory_path}'.")
    return results