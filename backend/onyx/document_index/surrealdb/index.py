import httpx
from typing import Any, List, Set, cast, Optional
from datetime import datetime
import json

from onyx.db.enums import EmbeddingPrecision
from onyx.document_index.interfaces import (
    DocumentIndex,
    IndexBatchParams,
    DocMetadataAwareIndexChunk,
    DocumentInsertionRecord,
    UpdateRequest,
    VespaChunkRequest,
    VespaDocumentFields,
    VespaDocumentUserFields,
    IndexFilters,
)
from onyx.context.search.models import InferenceChunk, QueryExpansionType
from shared_configs.model_server_models import Embedding
from onyx.utils.logger import setup_logger
from onyx.configs.chat_configs import TITLE_CONTENT_RATIO

logger = setup_logger()

class SurrealDBIndex(DocumentIndex):
    def __init__(
        self,
        index_name: str,
        secondary_index_name: str | None,
        large_chunks_enabled: bool,
        secondary_large_chunks_enabled: bool | None,
        multitenant: bool = False,
        httpx_client: httpx.Client | None = None,
        # Default connection settings for SurrealDB
        endpoint: str = "http://localhost:8000",
        namespace: str = "onyx",
        database: str = "onyx",
        username: str = "root",
        password: str = "root",
    ) -> None:
        self.index_name = index_name
        self.secondary_index_name = secondary_index_name
        self.large_chunks_enabled = large_chunks_enabled
        self.secondary_large_chunks_enabled = secondary_large_chunks_enabled
        self.multitenant = multitenant
        self.httpx_client = httpx_client or httpx.Client()
        
        self.endpoint = endpoint.rstrip("/")
        self.namespace = namespace
        self.database = database
        self.auth = (username, password)
        self.table_name = f"chunks_{self.index_name}"

    def _execute_sql(self, sql: str, vars: Optional[dict] = None) -> List[Any]:
        headers = {
            "NS": self.namespace,
            "DB": self.database,
            "Accept": "application/json",
        }
        # SurrealDB REST API supports parameters via JSON body if content is more complex,
        # but the /sql endpoint usually takes raw SQL. 
        # For production, we would use a more robust way to handle variables.
        res = self.httpx_client.post(
            f"{self.endpoint}/sql",
            content=sql,
            auth=self.auth,
            headers=headers,
            timeout=60.0
        )
        res.raise_for_status()
        return res.json()

    def ensure_indices_exist(
        self,
        primary_embedding_dim: int,
        primary_embedding_precision: EmbeddingPrecision,
        secondary_index_embedding_dim: int | None,
        secondary_index_embedding_precision: EmbeddingPrecision | None,
    ) -> None:
        sql = f"""
        DEFINE TABLE {self.table_name} SCHEMAFULL;
        DEFINE FIELD document_id ON {self.table_name} TYPE string;
        DEFINE FIELD chunk_id ON {self.table_name} TYPE int;
        DEFINE FIELD content ON {self.table_name} TYPE string;
        DEFINE FIELD embedding ON {self.table_name} TYPE array<float>;
        DEFINE FIELD metadata ON {self.table_name} TYPE object;
        DEFINE FIELD access ON {self.table_name} TYPE object;
        DEFINE FIELD boost ON {self.table_name} TYPE float;
        DEFINE FIELD hidden ON {self.table_name} TYPE bool;
        DEFINE FIELD updated_at ON {self.table_name} TYPE datetime;
        
        DEFINE INDEX idx_doc_id ON {self.table_name} FIELDS document_id;
        DEFINE INDEX idx_content ON {self.table_name} FIELDS content SEARCH ANALYZER ascii BM25;
        DEFINE INDEX idx_embedding ON {self.table_name} FIELDS embedding HNSW DIMENSION {primary_embedding_dim} DISTANCE COSINE;
        """
        self._execute_sql(sql)

    @staticmethod
    def register_multitenant_indices(
        indices: list[str],
        embedding_dims: list[int],
        embedding_precisions: list[EmbeddingPrecision],
    ) -> None:
        # Placeholder for multitenant index registration logic
        pass

    def index(
        self,
        chunks: list[DocMetadataAwareIndexChunk],
        index_batch_params: IndexBatchParams,
    ) -> set[DocumentInsertionRecord]:
        insertion_records: set[DocumentInsertionRecord] = set()
        
        for chunk in chunks:
            # Create a unique record ID in SurrealDB
            # Format: chunks_indexname:['doc_id', chunk_id]
            safe_doc_id = chunk.source_document.id.replace("'", "\\'")
            record_id = f"{self.table_name}:['{safe_doc_id}', {chunk.chunk_id}]"
            
            data = {
                "document_id": chunk.source_document.id,
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "embedding": chunk.embeddings.full_embedding,
                "metadata": chunk.source_document.metadata,
                "access": chunk.access.model_dump() if chunk.access else {},
                "boost": 0.0,
                "hidden": False,
                "updated_at": datetime.now().isoformat()
            }
            
            # SurrealDB SQL for upsert
            sql = f"UPSERT {record_id} CONTENT {json.dumps(data)};"
            self._execute_sql(sql)
            
            insertion_records.add(DocumentInsertionRecord(
                document_id=chunk.source_document.id,
                already_existed=False # Simplified for now
            ))
            
        return insertion_records

    def update_single(
        self,
        doc_id: str,
        *,
        tenant_id: str,
        chunk_count: int | None,
        fields: VespaDocumentFields | None,
        user_fields: VespaDocumentUserFields | None,
    ) -> None:
        # Update all chunks for a document
        updates = []
        if fields:
            if fields.boost is not None:
                updates.append(f"boost = {fields.boost}")
            if fields.hidden is not None:
                updates.append(f"hidden = {fields.hidden}")
        
        if updates:
            update_str = ", ".join(updates)
            sql = f"UPDATE {self.table_name} SET {update_str} WHERE document_id = '{doc_id.replace(chr(39), chr(92)+chr(39))}';"
            self._execute_sql(sql)

    def delete_single(
        self,
        doc_id: str,
        *,
        tenant_id: str,
        chunk_count: int | None,
    ) -> int:
        sql = f"DELETE {self.table_name} WHERE document_id = '{doc_id.replace(chr(39), chr(92)+chr(39))}';"
        res = self._execute_sql(sql)
        # SurrealDB returns the deleted records or a count depending on the version/query
        return len(res)

    def hybrid_retrieval(
        self,
        query: str,
        query_embedding: Embedding,
        final_keywords: list[str] | None,
        filters: IndexFilters,
        hybrid_alpha: float,
        time_decay_multiplier: float,
        num_to_retrieve: int,
        ranking_profile_type: QueryExpansionType,
        offset: int = 0,
        title_content_ratio: float | None = TITLE_CONTENT_RATIO,
    ) -> list[InferenceChunk]:
        # Vector search using KNN
        # SurrealDB: SELECT *, vector::distance::knn(embedding, $vector) AS dist FROM chunks WHERE ...
        
        # Keyword search using full-text index
        # SurrealDB: SELECT * FROM chunks WHERE content @0@ $query ...
        
        # For simplicity in this initial code, we'll perform a vector search
        # and assume hybrid search logic would be implemented by combining results.
        
        vector_val = json.dumps(query_embedding)
        sql = f"""
        SELECT *, vector::distance::knn(embedding, {vector_val}) AS score 
        FROM {self.table_name} 
        WHERE hidden = false
        ORDER BY score ASC 
        LIMIT {num_to_retrieve} START {offset};
        """
        results = self._execute_sql(sql)
        
        # Result from SurrealDB is a list of objects in the 'result' field of the last statement
        data = results[-1].get("result", []) if results else []
        
        inference_chunks = []
        for row in data:
             inference_chunks.append(InferenceChunk(
                 document_id=row["document_id"],
                 source_type=DocumentSource.SURREALDB, # Simplified
                 semantic_identifier=row["document_id"], # Simplified
                 title=None,
                 boost=int(row.get("boost", 0)),
                 recency_bias=1.0,
                 score=row.get("score"),
                 hidden=row.get("hidden", False),
                 metadata=row.get("metadata", {}),
                 match_highlights=[],
                 doc_summary="",
                 chunk_context="",
                 updated_at=datetime.fromisoformat(row["updated_at"]) if row.get("updated_at") else None,
                 chunk_id=row.get("chunk_id", 0),
                 blurb=row.get("content", "")[:200]
             ))
             
        return inference_chunks

    def id_based_retrieval(
        self,
        chunk_requests: list[VespaChunkRequest],
        filters: IndexFilters,
        batch_retrieval: bool = False,
    ) -> list[InferenceChunk]:
        # Implementation for retrieving specific chunks by ID
        return []

    def admin_retrieval(
        self,
        query: str,
        filters: IndexFilters,
        num_to_retrieve: int,
        offset: int = 0,
    ) -> list[InferenceChunk]:
        # Implementation for admin explorer search
        return []

    def random_retrieval(
        self,
        filters: IndexFilters,
        num_to_retrieve: int = 10,
    ) -> list[InferenceChunk]:
        return []

    def update(self, update_requests: list[UpdateRequest], *, tenant_id: str) -> None:
        pass
