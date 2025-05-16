# --- tests/conftest.py ---
import pytest
import os
import shutil
import importlib
from typing import List, Dict, Any, Optional

# Fixture pour mocker les dépendances Unstructured si non installées ou pour tests unitaires
@pytest.fixture
def mock_unstructured_dependencies(monkeypatch):
    """
    Mocks unstructured dependencies (partition_auto, Element, UnstructuredImage)
    if the 'unstructured' library is not found or to force mocking for unit tests.
    Tests needing this fixture should request it as an argument.
    """
    unstructured_spec = importlib.util.find_spec("unstructured")

    class DummyElement:
        def __init__(self, text: str = "", category: str = "Text", metadata: Optional[Dict[str, Any]] = None, element_id: str = "dummy_id"):
            self.text = text
            self.category = category
            self.metadata = metadata if metadata is not None else {}
            self.id = element_id # Ajout d'un ID pour la méthode to_dict

        def to_dict(self) -> Dict[str, Any]:
            return {
                "text": self.text,
                "type": self.category,
                "metadata": self.metadata,
                "element_id": self.id
            }

    class DummyImage(DummyElement):
        def __init__(self, text: str = "", image_bytes: bytes = b"", image_format: str = "png", metadata: Optional[Dict[str, Any]] = None):
            super().__init__(text=text, category="Image", metadata=metadata)
            self.image_bytes = image_bytes
            self.image_format = image_format # Attribut pour le format d'image

        # to_dict peut être hérité ou surchargé si besoin d'inclure image_format/bytes
        def to_dict(self) -> Dict[str, Any]:
            data = super().to_dict()
            data["image_format"] = self.image_format
            # Ne pas inclure image_bytes dans le dict par défaut car potentiellement volumineux
            return data


    def dummy_unstructured_partition_auto(filename: str, **kwargs) -> List[DummyElement]:
        if "error_case" in filename: # Pour simuler un échec de partitionnement
            return []
        # Simuler le retour d'un élément texte et d'un élément image
        return [
            DummyElement(text="Mocked text content from " + filename, category="NarrativeText"),
            DummyImage(image_bytes=b"dummy_image_bytes", image_format="png", metadata={"filename": "mocked_image.png"})
        ]

    # On mocke toujours pour les tests unitaires qui utilisent cette fixture,
    # pour assurer la prédictibilité indépendamment de l'installation réelle d'unstructured.
    # Les tests d'intégration peuvent choisir de ne pas utiliser cette fixture.
    
    # Patch AGENT_VF.etl.conversion où ces noms sont attendus
    monkeypatch.setattr("AGENT_VF.etl.conversion.unstructured_partition_auto", dummy_unstructured_partition_auto)
    monkeypatch.setattr("AGENT_VF.etl.conversion.Element", DummyElement)
    monkeypatch.setattr("AGENT_VF.etl.conversion.UnstructuredImage", DummyImage)
    
    # Retourner les mocks peut être utile pour certains tests pour vérifier les appels
    return {
        "partition": dummy_unstructured_partition_auto,
        "Element": DummyElement,
        "Image": DummyImage
    }

# Fixture pour un chemin de base de données ChromaDB temporaire et isolé pour chaque test
@pytest.fixture(autouse=True) # autouse=True pour l'appliquer à tous les tests
def temporary_chroma_db_path(monkeypatch, tmp_path_factory):
    """
    Crée un répertoire temporaire pour la base de données ChromaDB pour chaque session de test
    et configure la variable d'environnement CHROMA_DB_PATH pour l'utiliser.
    Nettoie le répertoire après les tests.
    """
    # Créer un sous-répertoire unique pour cette session de test dans le répertoire temporaire de pytest
    db_path = tmp_path_factory.mktemp("chroma_dbs_session")
    
    # Utiliser un sous-répertoire par test si tmp_path est utilisé (scope function)
    # Pour tmp_path_factory (scope session), on a un seul chemin pour la session.
    # Si on veut un par test, il faut utiliser tmp_path (scope function)
    # Pour cet exemple, on utilise un chemin unique pour la session.
    
    # Patch la variable d'environnement que etl.indexation.VECTOR_DB_PATH utilise
    monkeypatch.setenv("CHROMA_DB_PATH", str(db_path))
    
    # Patch la variable globale directement dans le module indexation pour s'assurer qu'elle est mise à jour
    # car elle est lue au moment de l'import du module.
    # Cela est nécessaire si les tests modifient l'environnement APRÈS l'import initial du module.
    if importlib.util.find_spec("AGENT_VF.etl.indexation"):
        indexation_module = importlib.import_module("AGENT_VF.etl.indexation")
        monkeypatch.setattr(indexation_module, "VECTOR_DB_PATH", str(db_path))
        # Réinitialiser la collection globale pour forcer la réinitialisation avec le nouveau chemin
        monkeypatch.setattr(indexation_module, "vector_db_collection", None)


    yield str(db_path) # Le chemin peut être utilisé par les tests s'ils en ont besoin

    # Le nettoyage est géré par tmp_path_factory à la fin de la session.
    # Si un nettoyage manuel plus agressif était nécessaire:
    # shutil.rmtree(str(db_path), ignore_errors=True)