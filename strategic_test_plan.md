Plan de Tests Stratégiques
Ce plan détaille les tests unitaires et d'intégration pour chaque module de l'agent AGENT_VF.
1. Module RAG (rag/retriever.py)
Objectif Général : Valider la capacité du retriever à identifier et retourner les documents les plus pertinents depuis la base vectorielle (FAISS/Chroma) en fonction d'une requête.
Tests Unitaires (test_rag.py)
Nom : test_retriever_initialization
Objectif : Vérifier que le Retriever s'initialise correctement avec une base vectorielle (mockée).
Données : Mock de l'objet VectorStore (ex: MagicMock(spec=VectorStore)).
Méthode : Unitaire.
KPI : Initialisation sans erreur. L'attribut vector_store de l'instance est assigné.
Nom : test_retriever_similarity_search_calls_vectorstore
Objectif : S'assurer que la méthode de recherche (ex: get_relevant_documents) appelle bien la méthode similarity_search (ou équivalent) du VectorStore mocké avec la bonne requête.
Données : Instance du Retriever avec VectorStore mocké, requête string simple.
Méthode : Unitaire.
KPI : La méthode similarity_search du mock est appelée exactement une fois avec la requête fournie.
Tests d'Intégration (test_rag.py)
Nom : test_retriever_integration_finds_relevant_docs
Objectif : Vérifier que le Retriever, connecté à une vraie base vectorielle (petite, contrôlée, ex: FAISS local), retourne les documents attendus pour une requête spécifique.
Données : Une petite base FAISS/Chroma pré-remplie avec 3-5 documents connus. Une requête dont la réponse attendue est connue.
Méthode : Intégration.
KPI : Retourne les k=3 documents attendus (vérification par ID ou contenu partiel). Score de similarité moyen des documents retournés > 0.75 (si l'API le permet). Temps de réponse < 500 ms.
Nom : test_retriever_integration_empty_query
Objectif : Vérifier le comportement avec une requête vide ou non pertinente.
Données : Base réelle, requête vide ou très générique ("le").
Méthode : Intégration.
KPI : Retourne une liste vide ou des documents avec un score de similarité bas (< 0.5). Ne lève pas d'exception.
Nom : test_retriever_integration_non_ascii_query
Objectif : Vérifier la robustesse avec des caractères spéciaux ou non-ASCII.
Données : Base réelle, requête avec accents, emojis, etc. ("rapport généré ?").
Méthode : Intégration.
KPI : Exécution sans erreur. Retourne des documents (si pertinents) ou une liste vide.
2. Module Writer (writer/client.py)
Objectif Général : Valider la capacité du Writer à générer du texte structuré et pertinent en utilisant le LLM (Ollama gemma:12b) basé sur un contexte et une requête.
Tests Unitaires (test_writer.py)
Nom : test_writer_initialization
Objectif : Vérifier l'initialisation correcte du client Writer (configuration Ollama).
Données : Paramètres de configuration (URL Ollama, nom modèle).
Méthode : Unitaire.
KPI : Initialisation sans erreur. Les attributs de configuration sont correctement assignés.
Nom : test_writer_prompt_formatting
Objectif : S'assurer que le prompt envoyé au LLM est correctement formaté à partir du contexte et de la requête.
Données : Contexte (liste de Document LangChain ou string), requête string. Mock de la méthode d'appel LLM (Ollama._call ou équivalent).
Méthode : Unitaire.
KPI : La méthode d'appel LLM mockée reçoit un prompt (string) contenant à la fois le contexte et la requête formatés selon le template attendu.
Nom : test_writer_parses_llm_output
Objectif : Vérifier que la sortie brute du LLM (mockée) est correctement parsée (ex: si un OutputParser LangChain est utilisé).
Données : Sortie LLM mockée (string JSON, string simple). Instance du Writer avec OutputParser si applicable.
Méthode : Unitaire.
KPI : La méthode de génération retourne le format attendu (dict si JSON, string sinon) basé sur la sortie mockée.
Tests d'Intégration (test_writer.py)
Nom : test_writer_integration_generates_structured_report
Objectif : Valider que le Writer, en appelant le vrai LLM Ollama, génère une réponse structurée (ex: JSON) pour une requête et un contexte donnés. Ne pas mocker le LLM.
Données : Contexte réaliste (ex: 2-3 extraits de documents), requête claire ("Génère un résumé structuré en JSON avec les points clés").
Méthode : Intégration.
KPI : La réponse est une chaîne de caractères parsable en JSON. Temps de réponse < 5 secondes. Le contenu JSON contient les clés attendues (ex: "titre", "résumé", "points_clés").
Nom : test_writer_integration_generates_sufficient_length
Objectif : Vérifier que la réponse générée atteint une longueur minimale. Ne pas mocker le LLM.
Données : Contexte et requête visant une réponse textuelle longue ("Écris un rapport détaillé sur ...").
Méthode : Intégration.
KPI : Longueur du contenu textuel généré > 300 mots. Temps de réponse < 8 secondes (peut être plus long pour des textes longs).
Nom : test_writer_integration_handles_no_context
Objectif : Vérifier le comportement lorsque aucun contexte n'est fourni. Ne pas mocker le LLM.
Données : Contexte vide ou None, requête simple ("Bonjour").
Méthode : Intégration.
KPI : Génère une réponse cohérente (peut-être générique ou une demande de clarification). Ne lève pas d'exception. Temps de réponse < 3 secondes.
3. Module Validation (validation/validator.py)
Objectif Général : Valider la capacité du Validator à vérifier la conformité de la sortie du Writer par rapport à des règles prédéfinies (structure, longueur, contenu).
Tests Unitaires (test_validator.py)
Nom : test_validator_structure_check_valid
Objectif : Vérifier que le validateur accepte une structure JSON correcte.
Données : Un dictionnaire Python ou une string JSON correspondant à la structure attendue.
Méthode : Unitaire.
KPI : La méthode de validation retourne True ou un état de succès. Précision = 100% sur ce cas.
Nom : test_validator_structure_check_invalid_missing_key
Objectif : Vérifier que le validateur rejette une structure JSON avec une clé manquante.
Données : Un dictionnaire/JSON où une clé requise manque.
Méthode : Unitaire.
KPI : La méthode de validation retourne False ou un état d'échec indiquant une erreur de structure. Précision = 100% sur ce cas.
Nom : test_validator_length_check_valid
Objectif : Vérifier que le validateur accepte un texte de longueur suffisante.
Données : Une string dont la longueur (en mots ou caractères) est supérieure au seuil défini.
Méthode : Unitaire.
KPI : La méthode de validation retourne True ou succès. Précision = 100%.
Nom : test_validator_length_check_invalid_too_short
Objectif : Vérifier que le validateur rejette un texte trop court.
Données : Une string dont la longueur est inférieure au seuil.
Méthode : Unitaire.
KPI : La méthode de validation retourne False ou échec indiquant une erreur de longueur. Précision = 100%.
Nom : test_validator_prompt_adherence_check_valid
Objectif : Vérifier que le validateur détecte la présence de mots-clés ou de formats spécifiques demandés dans le prompt (si applicable).
Données : Texte contenant les éléments requis (ex: "Conclusion:", formatage spécifique). Prompt original (ou ses contraintes).
Méthode : Unitaire.
KPI : La méthode de validation retourne True ou succès. Précision = 100%.
Nom : test_validator_prompt_adherence_check_invalid
Objectif : Vérifier que le validateur détecte l'absence d'éléments requis par le prompt.
Données : Texte ne contenant pas les éléments requis. Prompt original.
Méthode : Unitaire.
KPI : La méthode de validation retourne False ou échec indiquant un non-respect du prompt. Précision = 100%.
Tests d'Intégration (test_validator.py)
Nom : test_validator_integration_accuracy_on_real_outputs
Objectif : Évaluer la précision globale du validateur sur un jeu de sorties réelles (ou réalistes) du Writer, certaines valides, d'autres invalides.
Données : Un set de 10-20 exemples de sorties (strings, JSON strings) générées précédemment ou créées manuellement pour simuler des cas valides et invalides (structure, longueur, prompt).
Méthode : Intégration (car teste l'ensemble des règles sur des données complexes).
KPI : Précision globale de détection (TP+TN / Total) > 90%. Rappel sur les erreurs (TP / TP+FN) > 85%.
4. Module Orchestrator (orchestrator/graph.py)
Objectif Général : Valider le bon fonctionnement du graphe LangGraph, l'enchaînement correct des nœuds (RAG, Writer, Validator) et la gestion des boucles/conditions.
Tests Unitaires (test_orchestrator.py)
Nom : test_graph_compilation
Objectif : Vérifier que le graphe LangGraph se compile sans erreur.
Données : Définition du graphe avec des fonctions mockées pour chaque nœud (RAG, Writer, Validator).
Méthode : Unitaire.
KPI : L'appel à graph.compile() réussit sans exception. L'objet graphe compilé est retourné.
Nom : test_graph_state_update_per_node
Objectif : Vérifier que l'état du graphe est correctement mis à jour après l'exécution de chaque nœud (mocké).
Données : Graphe compilé avec nœuds mockés. État initial. Sorties mockées pour chaque nœud.
Méthode : Unitaire.
KPI : Après l'appel simulé d'un nœud (ex: RAG), l'état contient les clés attendues (ex: documents). Après le Writer, l'état contient generation. Après le Validator, l'état contient validation_result.
Nom : test_graph_conditional_edge_routing
Objectif : Vérifier que les arêtes conditionnelles (basées sur la sortie du Validator) dirigent correctement le flux (vers la fin ou vers une boucle/réécriture).
Données : Graphe compilé avec nœuds mockés. Sortie mockée du Validator (une fois valide, une fois invalide).
Méthode : Unitaire.
KPI : Si Validator retourne "valide", le prochain nœud exécuté est le nœud final (__end__). Si Validator retourne "invalide", le prochain nœud est celui de la boucle (ex: re-Writer ou RAG). Vérification via l'historique d'appel des mocks ou l'état final.
Tests d'Intégration (test_orchestrator.py)
Nom : test_graph_integration_full_run_success
Objectif : Exécuter le graphe complet avec les vrais composants (ou au moins Writer/Validator réels) sur une requête simple et vérifier qu'il se termine avec succès.
Données : Requête simple ("Fais un résumé de ce document : [texte court]"). Composants RAG (peut être mocké si la base est complexe à setup, sinon réel), Writer (réel), Validator (réel).
Méthode : Intégration.
KPI : Le graphe s'exécute jusqu'à la fin (__end__) sans exception. L'état final contient une génération validée. 100% des nœuds attendus dans un flux nominal sont atteints (vérifiable via graph.stream() ou logs LangGraph). Temps d'exécution total < 10 secondes.
Nom : test_graph_integration_handles_validation_failure_and_loops
Objectif : Simuler un échec de validation et vérifier que le graphe boucle correctement (si une boucle de correction est implémentée).
Données : Requête pouvant mener à une sortie initialement invalide (ex: trop courte). Vrais composants Writer/Validator. Potentiellement mocker le Validator pour forcer un échec au premier passage.
Méthode : Intégration.
KPI : Le nœud Validator est appelé au moins deux fois. Le nœud Writer (ou le nœud de correction) est appelé après le premier échec de validation. Le graphe se termine éventuellement (soit par succès après correction, soit par un nombre max de tentatives).
Nom : test_graph_integration_state_consistency
Objectif : Vérifier que les informations clés (requête initiale, documents récupérés, génération, résultat validation) sont présentes et cohérentes dans l'état final du graphe après une exécution réussie.
Données : Requête simple. Vrais composants.
Méthode : Intégration.
KPI : L'état final (graph.invoke(...) ou dernier état de graph.stream(...)) contient la requête originale, une liste de documents (si RAG utilisé), la génération finale, et un indicateur de validation positive.
5. Module API (api/main.py)
Objectif Général : Valider que l'API FastAPI expose correctement les fonctionnalités de l'agent, gère les requêtes HTTP et retourne les réponses appropriées.
Tests Unitaires (test_api.py)
Nom : test_api_endpoint_exists
Objectif : Vérifier que l'endpoint principal (ex: /generate) est défini et accessible.
Données : TestClient de FastAPI.
Méthode : Unitaire (teste la configuration de FastAPI, pas le traitement).
KPI : Une requête OPTIONS ou GET (si défini) sur l'endpoint ne retourne pas 404.
Nom : test_api_request_validation_valid
Objectif : Vérifier que l'API accepte une requête valide (conforme au modèle Pydantic).
Données : TestClient. Payload JSON valide pour l'endpoint. Mock de la fonction de l'orchestrateur appelée par l'endpoint.
Méthode : Unitaire.
KPI : La requête retourne un code de succès (200 OK). Le mock de l'orchestrateur est appelé avec les données parsées.
Nom : test_api_request_validation_invalid
Objectif : Vérifier que l'API rejette une requête invalide (champ manquant, mauvais type).
Données : TestClient. Payload JSON invalide.
Méthode : Unitaire.
KPI : La requête retourne un code 422 Unprocessable Entity. Le corps de la réponse contient les détails de l'erreur de validation Pydantic.
Nom : test_api_response_formatting
Objectif : Vérifier que la réponse de l'API est correctement formatée (JSON) lorsque l'orchestrateur retourne un résultat.
Données : TestClient. Payload valide. Mock de l'orchestrateur retournant un résultat simulé (dict).
Méthode : Unitaire.
KPI : La réponse a le statut 200 OK. Le Content-Type est application/json. Le corps de la réponse JSON correspond au retour du mock de l'orchestrateur.
Tests d'Intégration (test_api.py)
Nom : test_api_integration_generate_endpoint_success
Objectif : Envoyer une requête valide à l'endpoint /generate et vérifier qu'il déclenche l'orchestrateur (réel) et retourne une réponse réussie.
Données : TestClient. Payload JSON valide avec une requête simple. L'agent complet (orchestrateur, RAG, Writer, Validator) doit être fonctionnel en arrière-plan.
Méthode : Intégration.
KPI : Réponse avec statut 200 OK. Temps de réponse total de l'API < 300 ms (hors temps de génération LLM, qui dépend du test d'orchestration). Le corps de la réponse contient le rapport généré par l'agent.
Nom : test_api_integration_generate_endpoint_handles_internal_error
Objectif : Vérifier que l'API retourne une erreur 500 si l'orchestrateur lève une exception inattendue.
Données : TestClient. Payload valide. Mocker l'appel à l'orchestrateur pour qu'il lève une Exception.
Méthode : Intégration (teste le handling d'erreur de l'API face à une défaillance de la logique métier).
KPI : Réponse avec statut 500 Internal Server Error. Le corps de la réponse contient un message d'erreur générique.
Nom : test_api_integration_health_check_endpoint (Si un endpoint /health existe)
Objectif : Vérifier que l'endpoint de health check répond correctement.
Données : TestClient.
Méthode : Intégration (vérifie la disponibilité de base de l'API).
KPI : Réponse 200 OK en < 100 ms. Corps de la réponse indiquant un statut "OK" ou "Healthy".

Instructions d'utilisation
Structure du projet :
.
├── AGENT_VF/
│   ├── rag/
│   │   └── retriever.py
│   ├── writer/
│   │   └── client.py
│   ├── validation/
│   │   └── validator.py
│   ├── orchestrator/
│   │   └── graph.py
│   ├── api/
│   │   └── main.py
│   └── ... (autres fichiers .py, __init__.py)
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── pytest.ini
│   ├── test_rag.py
│   ├── test_writer.py
│   ├── test_validator.py
│   ├── test_orchestrator.py
│   ├── test_api.py
│   └── run_tests.py
├── requirements.txt
└── requirements-dev.txt # Ajouter pytest, typer, fastapi[all], etc.
Use code with caution.
Dépendances : Assure-toi d'installer les dépendances de développement :
pip install -r requirements-dev.txt
# requirements-dev.txt devrait contenir au moins:
# pytest
# typer[all]
# langchain langchain-community langchain-core # Ou les spécifiques nécessaires
# faiss-cpu # ou faiss-gpu ou chromadb
# fastapi uvicorn[standard]
# python-dotenv # Si utilisé pour la config
# ollama # Si l'API python est utilisée, sinon juste s'assurer que le service tourne
Use code with caution.
Bash
Configuration :
Vérifie que le service Ollama avec le modèle gemma:12b (ou gemma:7b utilisé dans les tests pour plus de rapidité) est en cours d'exécution et accessible (par défaut http://localhost:11434). Ajuste OLLAMA_BASE_URL si nécessaire (via variable d'env ou dans le code/fixtures).
Adapte les imports et les noms de classes/fonctions dans les fichiers de test pour correspondre exactement à ton code dans AGENT_VF/. Les classes Retriever, Writer, Validator, MockOllama, etc., dans les tests sont des placeholders à remplacer/adapter.
Implémente la logique de création/chargement de la base vectorielle dans la fixture real_vector_store (test_rag.py).
Implémente la logique de récupération du graphe compilé réel dans la fixture compiled_agent_graph (test_orchestrator.py).
Assure-toi que la vraie application FastAPI (AGENT_VF.api.main.app) est importable pour les tests d'intégration API.
Exécution :
python tests/run_tests.py : Lance tous les tests.
python tests/run_tests.py --marker unit : Lance uniquement les tests unitaires.
python tests/run_tests.py --marker integration : Lance uniquement les tests d'intégration (nécessite Ollama, etc.).
python tests/run_tests.py -k test_api_integration : Lance les tests dont le nom contient test_api_integration.
python tests/run_tests.py -v : Mode verbeux.
python tests/run_tests.py -x : Arrête à la première erreur.