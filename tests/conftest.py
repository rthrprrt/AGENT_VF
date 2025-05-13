# tests/conftest.py
import pytest
from unittest.mock import MagicMock
# Importez ici des classes ou fonctions nécessaires pour les mocks ou setup
# from AGENT_VF.rag.retriever import Retriever
# from langchain_core.vectorstores import VectorStore
# from langchain_core.documents import Document

# Exemple de fixture pour un VectorStore mocké
# @pytest.fixture
# def mock_vector_store() -> MagicMock:
#     mock = MagicMock(spec=VectorStore)
#     # Configurez le comportement du mock si nécessaire
#     doc1 = Document(page_content="Ceci est le doc 1.", metadata={"id": "doc1"})
#     doc2 = Document(page_content="Contenu du document 2.", metadata={"id": "doc2"})
#     mock.similarity_search.return_value = [doc1, doc2]
#     return mock

# Exemple de fixture pour le client Ollama (pourrait être utilisé dans les tests d'intégration)
# @pytest.fixture(scope="session") # Scope session pour ne l'instancier qu'une fois
# def ollama_llm():
#     # Assurez-vous qu'Ollama tourne et est accessible
#     try:
#         from langchain_community.llms import Ollama
#         llm = Ollama(model="gemma:12b", base_url="http://localhost:11434") # Ajustez base_url si nécessaire
#         # Petit test de connexion rapide
#         llm.invoke("Bonjour")
#         return llm
#     except Exception as e:
#         pytest.skip(f"Ollama (gemma:12b) n'est pas accessible. Skipping integration tests. Error: {e}")