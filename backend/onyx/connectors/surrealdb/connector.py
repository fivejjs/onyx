from typing import Any
from collections.abc import Generator
import httpx

from onyx.configs.app_configs import INDEX_BATCH_SIZE
from onyx.configs.constants import DocumentSource
from onyx.connectors.interfaces import GenerateDocumentsOutput, LoadConnector
from onyx.connectors.models import Document, Section, TextSection
from onyx.utils.logger import setup_logger
from onyx.connectors.cross_connector_utils.miscellaneous_utils import process_in_batches

logger = setup_logger()

class SurrealDBConnector(LoadConnector):
    def __init__(
        self,
        endpoint: str,
        namespace: str,
        database: str,
        surreal_query: str,
        batch_size: int = INDEX_BATCH_SIZE,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.namespace = namespace
        self.database = database
        self.surreal_query = surreal_query
        self.batch_size = batch_size
        
        self.username: str | None = None
        self.password: str | None = None

    def load_credentials(self, credentials: dict[str, Any]) -> dict[str, Any] | None:
        self.username = credentials.get("username")
        self.password = credentials.get("password")
        return None

    def _execute_query(self) -> list[dict[str, Any]]:
        auth = None
        if self.username and self.password:
            auth = (self.username, self.password)
            
        headers = {
            "Accept": "application/json",
            "NS": self.namespace,
            "DB": self.database,
        }
        
        url = f"{self.endpoint}/sql"
        
        with httpx.Client(auth=auth) as client:
            response = client.post(
                url,
                content=self.surreal_query,
                headers=headers,
                timeout=60.0,
            )
            response.raise_for_status()
            
            # SurrealDB returns a list of results, one for each statement in the query
            results = response.json()
            if not results:
                return []
            
            # We assume the last result contains the data we want
            last_result = results[-1]
            if last_result.get("status") != "OK":
                raise RuntimeError(f"SurrealDB query failed: {last_result.get('detail')}")
                
            data = last_result.get("result", [])
            if not isinstance(data, list):
                data = [data]
            return data

    def _map_to_document(self, row: dict[str, Any]) -> Document:
        # Expected fields in the row:
        # id: unique identifier for the document
        # content: the main text content
        # title: (optional) title of the document
        # link: (optional) link to the document
        # metadata: (optional) dict of metadata
        
        doc_id = str(row.get("id", ""))
        if not doc_id:
            raise ValueError("Row missing 'id' field")
            
        content = row.get("content", "")
        if not content:
             # Try fallback to 'text' if 'content' is missing
             content = row.get("text", "")

        title = row.get("title", row.get("name", doc_id))
        link = row.get("link", "")
        metadata = row.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
            
        return Document(
            id=doc_id,
            sections=[TextSection(text=content, link=link)],
            source=DocumentSource.SURREALDB,
            semantic_identifier=title,
            metadata=metadata,
        )

    def load_from_state(self) -> GenerateDocumentsOutput:
        rows = self._execute_query()
        yield from process_in_batches(
            objects=rows,
            process_function=self._map_to_document,
            batch_size=self.batch_size,
        )
