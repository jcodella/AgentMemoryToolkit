"""AgentMemory: local and cloud agent memory management.

Thin orchestrator that composes :class:`CosmosMemoryStore`,
:class:`EmbeddingsClient`, and :class:`ProcessingClient` for Cosmos DB
CRUD, vector search, and Azure Durable Functions processing.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

from .cosmos_memory_client import CosmosMemoryStore
from .embeddings import EmbeddingsClient
from .exceptions import CosmosNotConnectedError, MemoryNotFoundError, ValidationError
from .models import MemoryRecord
from .processing import ProcessingClient
from ._utils import VALID_ROLES, VALID_TYPES, _make_memory, _resolve_embedding_dimensions

logger = logging.getLogger(__name__)


class AgentMemory:
    """Manages agent memories with local storage and Cosmos DB.

    Authentication uses ``azure-identity`` by default.  If no explicit
    credential is passed for Cosmos DB or AI Foundry, a
    ``DefaultAzureCredential`` is created automatically.

    Parameters
    ----------
    cosmos_endpoint : str, optional
        The Cosmos DB account endpoint URL.
    cosmos_credential : TokenCredential, optional
        Azure credential for Cosmos DB.
    cosmos_database : str, optional
        Cosmos DB database name.
    cosmos_container : str, optional
        Cosmos DB container name.
    ai_foundry_endpoint : str, optional
        Azure OpenAI endpoint URL for embeddings.
    ai_foundry_credential : TokenCredential, optional
        Azure credential for the AI Foundry endpoint.
    ai_foundry_api_key : str, optional
        API key for Azure OpenAI (takes precedence over credential).
    embedding_model : str, optional
        Embedding model deployment name (default ``text-embedding-3-large``).
    embedding_dimensions : int, optional
        Dimensionality of embedding vectors.
    adf_endpoint : str, optional
        Base URL for the Azure Durable Functions API.
    adf_key : str, optional
        Function-level key for authenticating to the Azure Function.
    use_default_credential : bool, optional
        Automatically create ``DefaultAzureCredential`` when ``True``.
    """

    def __init__(
        self,
        cosmos_endpoint: Optional[str] = None,
        cosmos_credential: Optional[Any] = None,
        cosmos_database: Optional[str] = None,
        cosmos_container: Optional[str] = None,
        ai_foundry_endpoint: Optional[str] = None,
        ai_foundry_credential: Optional[Any] = None,
        ai_foundry_api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-3-large",
        embedding_dimensions: Optional[int] = None,
        adf_endpoint: Optional[str] = None,
        adf_key: Optional[str] = None,
        use_default_credential: bool = True,
    ) -> None:
        # Local store
        self.local_memory: list[dict[str, Any]] = []

        # Store kwargs directly
        self._cosmos_endpoint = cosmos_endpoint
        self._cosmos_credential = cosmos_credential
        self._cosmos_database = cosmos_database or "ai_memory"
        self._cosmos_container = cosmos_container or "memories"

        self._ai_foundry_endpoint = ai_foundry_endpoint
        self._ai_foundry_credential = ai_foundry_credential
        self._ai_foundry_api_key = ai_foundry_api_key
        self._embedding_model = embedding_model
        self._embedding_dimensions = _resolve_embedding_dimensions(embedding_dimensions)

        self._adf_endpoint = adf_endpoint
        self._adf_key = adf_key

        # Resolve credentials via DefaultAzureCredential when needed
        if use_default_credential:
            needs_cosmos = self._cosmos_credential is None
            needs_embed = self._ai_foundry_credential is None
            if needs_cosmos or needs_embed:
                try:
                    from azure.identity import DefaultAzureCredential
                    _default = DefaultAzureCredential()
                except ImportError:
                    _default = None
                if needs_cosmos:
                    self._cosmos_credential = _default
                if needs_embed:
                    self._ai_foundry_credential = _default

        # Sub-clients (cosmos store created on connect)
        self._cosmos_store: Optional[CosmosMemoryStore] = None
        self._embeddings_client = EmbeddingsClient(
            endpoint=self._ai_foundry_endpoint,
            credential=self._ai_foundry_credential,
            api_key=self._ai_foundry_api_key,
            model=self._embedding_model,
            dimensions=self._embedding_dimensions,
        )
        self._processing_client = ProcessingClient(
            endpoint=self._adf_endpoint,
            key=self._adf_key,
        )

        logger.info("AgentMemory initialized")

    # ------------------------------------------------------------------
    # Local operations
    # ------------------------------------------------------------------

    def add_local(
        self,
        user_id: str,
        role: str,
        content: str,
        memory_type: str = "turn",
        agent_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        thread_id: Optional[str] = None,
    ) -> None:
        """Add a new memory to the local store."""
        memory = _make_memory(
            user_id=user_id,
            role=role,
            content=content,
            memory_type=memory_type,
            agent_id=agent_id,
            metadata=metadata,
            thread_id=thread_id,
        )
        self.local_memory.append(memory)
        logger.debug("add_local id=%s role=%s type=%s", memory["id"], role, memory_type)

    def get_local(
        self,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
        role: Optional[str] = None,
        memory_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Retrieve memories from the local store.

        All filter parameters are optional. When none are provided every
        memory is returned. Filters are combined with AND logic.
        """
        logger.debug(
            "get_local memory_id=%s user_id=%s role=%s type=%s",
            memory_id, user_id, role, memory_type,
        )
        results = self.local_memory

        if memory_id is not None:
            results = [m for m in results if m["id"] == memory_id]
        if user_id is not None:
            results = [m for m in results if m["user_id"] == user_id]
        if role is not None:
            results = [m for m in results if m["role"] == role]
        if memory_type is not None:
            results = [m for m in results if m["type"] == memory_type]

        return results

    def update_local(
        self,
        memory_id: str,
        content: Optional[str] = None,
        role: Optional[str] = None,
        memory_type: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Update an existing memory in the local store.

        Only the fields that are provided (not ``None``) will be updated.

        Raises
        ------
        MemoryNotFoundError
            If no memory with the given id exists.
        ValidationError
            If an invalid role or memory_type is provided.
        """
        for memory in self.local_memory:
            if memory["id"] == memory_id:
                if content is not None:
                    memory["content"] = content
                if role is not None:
                    if role not in VALID_ROLES:
                        raise ValidationError(f"role must be one of {VALID_ROLES}, got '{role}'")
                    memory["role"] = role
                if memory_type is not None:
                    if memory_type not in VALID_TYPES:
                        raise ValidationError(f"type must be one of {VALID_TYPES}, got '{memory_type}'")
                    memory["type"] = memory_type
                if metadata is not None:
                    memory["metadata"] = metadata
                memory["updated_at"] = datetime.now(timezone.utc).isoformat()
                return

        raise MemoryNotFoundError(memory_id=memory_id)

    def delete_local(self, memory_id: str) -> None:
        """Delete a memory from the local store by id.

        Raises
        ------
        MemoryNotFoundError
            If no memory with the given id exists.
        """
        for i, memory in enumerate(self.local_memory):
            if memory["id"] == memory_id:
                self.local_memory.pop(i)
                return

        raise MemoryNotFoundError(memory_id=memory_id)

    # ------------------------------------------------------------------
    # Cosmos DB connection
    # ------------------------------------------------------------------

    def connect_cosmos(
        self,
        endpoint: Optional[str] = None,
        credential: Optional[Any] = None,
        database: Optional[str] = None,
        container: Optional[str] = None,
    ) -> None:
        """Establish a connection to a Cosmos DB container.

        Parameters override whatever was set in ``__init__``.  After this
        call the Cosmos CRUD methods are ready to use.
        """
        self._cosmos_endpoint = endpoint or self._cosmos_endpoint
        self._cosmos_credential = credential or self._cosmos_credential
        self._cosmos_database = database or self._cosmos_database
        self._cosmos_container = container or self._cosmos_container

        self._cosmos_store = CosmosMemoryStore(
            endpoint=self._cosmos_endpoint,
            credential=self._cosmos_credential,
            database=self._cosmos_database,
            container=self._cosmos_container,
        )
        self._cosmos_store.connect()

    def create_memory_store(
        self,
        database: Optional[str] = None,
        container: Optional[str] = None,
        endpoint: Optional[str] = None,
        credential: Optional[Any] = None,
        embedding_dimensions: Optional[int] = None,
        embedding_data_type: Optional[str] = None,
        distance_function: Optional[str] = None,
        full_text_language: Optional[str] = None,
    ) -> None:
        """Create the Cosmos DB database and container for memories.

        After successful creation the instance is connected and ready
        for CRUD operations.
        """
        self._cosmos_endpoint = endpoint or self._cosmos_endpoint
        self._cosmos_credential = credential or self._cosmos_credential
        self._cosmos_database = database or self._cosmos_database
        self._cosmos_container = container or self._cosmos_container

        self._cosmos_store = CosmosMemoryStore(
            endpoint=self._cosmos_endpoint,
            credential=self._cosmos_credential,
            database=self._cosmos_database,
            container=self._cosmos_container,
        )
        self._cosmos_store.create_store(
            embedding_dimensions=embedding_dimensions or self._embedding_dimensions or 1536,
            embedding_data_type=embedding_data_type or "float32",
            distance_function=distance_function or "cosine",
            full_text_language=full_text_language or "en-US",
        )

    def _require_cosmos(self) -> None:
        """Raise if Cosmos DB is not connected."""
        if self._cosmos_store is None:
            raise CosmosNotConnectedError()
        self._cosmos_store._require_connected()

    # ------------------------------------------------------------------
    # Cosmos DB operations
    # ------------------------------------------------------------------

    def add_cosmos(
        self,
        user_id: str,
        role: str,
        content: str,
        memory_type: str = "turn",
        metadata: Optional[dict[str, Any]] = None,
        thread_id: Optional[str] = None,
    ) -> None:
        """Add a memory to Cosmos DB."""
        self._require_cosmos()
        kwargs: dict[str, Any] = {
            "user_id": user_id,
            "role": role,
            "content": content,
            "memory_type": memory_type,
            "metadata": metadata or {},
        }
        if thread_id is not None:
            kwargs["thread_id"] = thread_id
        record = MemoryRecord(**kwargs)
        self._cosmos_store.upsert(record)
        logger.info("add_cosmos id=%s role=%s type=%s", record.id, role, memory_type)

    def push_to_cosmos(self) -> None:
        """Insert all local memories into Cosmos DB.

        Each local memory is inserted as-is, preserving its existing
        ``id``, ``thread_id``, timestamps, and metadata.
        """
        self._require_cosmos()
        logger.info("push_to_cosmos count=%d", len(self.local_memory))
        records = [MemoryRecord.from_cosmos_dict(dict(m)) for m in self.local_memory]
        self._cosmos_store.upsert_batch(records)

    def get_memories(
        self,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        role: Optional[str] = None,
        memory_type: Optional[str] = None,
        recent_k: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Retrieve memories from Cosmos DB with optional filters.

        Args:
            memory_id: Filter by memory id.
            user_id: Filter by user id.
            thread_id: Filter by thread id.
            role: Filter by role.
            memory_type: Filter by type (raw, summary, fact, etc.).
            recent_k: If specified, return only the *k* most recent documents
                (ordered by ``_ts`` descending, then reversed to chronological).
        """
        self._require_cosmos()
        logger.debug(
            "get_memories filters: memory_id=%s user_id=%s thread_id=%s role=%s type=%s recent_k=%s",
            memory_id, user_id, thread_id, role, memory_type, recent_k,
        )
        results = self._cosmos_store.get_memories(
            memory_id=memory_id,
            user_id=user_id,
            thread_id=thread_id,
            role=role,
            memory_type=memory_type,
            recent_k=recent_k,
        )
        if not results:
            logger.warning("get_memories returned empty results")
        return results

    def update_cosmos(
        self,
        memory_id: str,
        content: Optional[str] = None,
        role: Optional[str] = None,
        memory_type: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Update a memory in Cosmos DB."""
        self._require_cosmos()
        self._cosmos_store.update(
            memory_id=memory_id,
            content=content,
            role=role,
            memory_type=memory_type,
            metadata=metadata,
        )

    def delete_cosmos(self, memory_id: str, thread_id: str, user_id: str) -> None:
        """Delete a memory from Cosmos DB."""
        self._require_cosmos()
        self._cosmos_store.delete(
            memory_id=memory_id,
            user_id=user_id,
            thread_id=thread_id,
        )

    def search_cosmos(
        self,
        search_terms: str,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
        role: Optional[str] = None,
        memory_type: Optional[str] = None,
        thread_id: Optional[str] = None,
        hybrid_search: bool = False,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search memories in Cosmos DB using vector similarity.

        1. Embeds *search_terms* via the configured embedding model.
        2. Runs a vector similarity query against the Cosmos DB container.
        3. Optionally filters by the remaining keyword parameters.
        4. Returns up to *top_k* results ordered by similarity.
        """
        self._require_cosmos()
        logger.info(
            "search_cosmos terms_len=%d top_k=%d hybrid_search=%s",
            len(search_terms),
            top_k,
            hybrid_search,
        )
        logger.debug(
            "search_cosmos search_terms=%s",
            search_terms[:50] + "..." if len(search_terms) > 50 else search_terms,
        )
        query_vector = self._embeddings_client.generate(search_terms)
        results = self._cosmos_store.vector_search(
            query_vector=query_vector,
            user_id=user_id,
            role=role,
            memory_type=memory_type,
            thread_id=thread_id,
            hybrid_search=hybrid_search,
            search_terms=search_terms,
            top_k=top_k,
        )
        # Post-filter by memory_id (not supported directly by vector_search)
        if memory_id is not None:
            results = [r for r in results if r.get("id") == memory_id]
        if not results:
            logger.warning(
                "search_cosmos returned empty results (terms_len=%d)",
                len(search_terms),
            )
        return results

    def get_thread(
        self,
        thread_id: str,
        user_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        recent_k: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Retrieve an entire thread from Cosmos DB.

        Returns memories sorted in chronological order (oldest first).
        """
        self._require_cosmos()
        return self._cosmos_store.get_thread(
            thread_id=thread_id,
            user_id=user_id,
            memory_type=memory_type,
            recent_k=recent_k,
        )

    def get_user_summary(self, user_id: str) -> list[dict[str, Any]]:
        """Retrieve user summary documents from Cosmos DB, newest first."""
        self._require_cosmos()
        return self._cosmos_store.get_user_summary(user_id=user_id)

    # ------------------------------------------------------------------
    # Processing (Azure Durable Functions)
    # ------------------------------------------------------------------

    def generate_thread_summary(
        self,
        user_id: str,
        thread_id: str,
        recent_k: Optional[int] = None,
        poll_interval: float = 2.0,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        """Trigger the Azure Durable Function to generate a thread summary."""
        logger.info(
            "generate_thread_summary started user_id=%s thread_id=%s",
            user_id,
            thread_id,
        )
        return self._processing_client.generate_thread_summary(
            user_id=user_id,
            thread_id=thread_id,
            recent_k=recent_k,
            poll_interval=poll_interval,
            timeout=timeout,
        )

    def extract_facts(
        self,
        user_id: str,
        thread_id: str,
        recent_k: Optional[int] = None,
        poll_interval: float = 2.0,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        """Trigger the Azure Durable Function to extract facts from a thread."""
        logger.info(
            "extract_facts started user_id=%s thread_id=%s",
            user_id,
            thread_id,
        )
        return self._processing_client.extract_facts(
            user_id=user_id,
            thread_id=thread_id,
            recent_k=recent_k,
            poll_interval=poll_interval,
            timeout=timeout,
        )

    def generate_user_summary(
        self,
        user_id: str,
        thread_ids: Optional[list[str]] = None,
        recent_k: Optional[int] = None,
        poll_interval: float = 2.0,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        """Trigger the Azure Durable Function to generate a cross-thread user summary."""
        logger.info("generate_user_summary started user_id=%s", user_id)
        return self._processing_client.generate_user_summary(
            user_id=user_id,
            thread_ids=thread_ids,
            recent_k=recent_k,
            poll_interval=poll_interval,
            timeout=timeout,
        )
