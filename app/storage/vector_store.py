"""
Storage layer — ChromaDB vector store for RAG
"""

import re
from typing import List, Dict, Any, Optional

import chromadb
from sentence_transformers import SentenceTransformer


class ChromaStore:
    def __init__(self, persist_directory: str = "./chroma_data"):
        self.persist_directory = persist_directory
        self._client = chromadb.PersistentClient(
            path=persist_directory,
        )
        self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self._collections: Dict[str, Any] = {}

    def _sanitize_name(self, name: str) -> str:
        """Sanitize name to be ChromaDB-compatible (3-512 chars, alphanumeric._-)."""
        # Replace spaces with underscores, keep only valid chars
        sanitized = re.sub(r'[^a-zA-Z0-9._-]', '_', name)
        sanitized = sanitized.strip('_')
        # Ensure length constraints
        if len(sanitized) < 3:
            sanitized = "research_" + sanitized.ljust(3, 'x')
        if len(sanitized) > 512:
            sanitized = sanitized[:512].rstrip('_')
        return sanitized

    async def _get_collection(self, collection_name: str):
        """Get or create a collection."""
        sanitized_name = self._sanitize_name(collection_name)
        if sanitized_name not in self._collections:
            try:
                collection = self._client.get_collection(name=sanitized_name)
            except Exception:
                collection = self._client.create_collection(
                    name=sanitized_name,
                    metadata={"description": f"Research for {collection_name}", "original_name": collection_name}
                )
            self._collections[sanitized_name] = collection
        return self._collections[sanitized_name]

    async def add_documents(
        self,
        documents: List[str],
        ids: List[str],
        metadatas: List[Dict[str, Any]],
        collection_name: str = "research",
    ) -> None:
        """Add documents to the collection."""
        collection = await self._get_collection(collection_name)

        # Generate embeddings
        embeddings = self._embedding_model.encode(documents).tolist()

        # Add to ChromaDB
        collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    async def query(
        self,
        collection_name: str,
        query_text: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Query the collection."""
        collection = await self._get_collection(collection_name)

        # Embed query
        query_embedding = self._embedding_model.encode([query_text]).tolist()

        # Query
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas"],
        )

        # Format results
        formatted = []
        if results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                formatted.append({
                    "content": doc,
                    "title": results["metadatas"][0][i].get("title", "Untitled") if results["metadatas"] else "Untitled",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                })

        return formatted

    async def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            collection = await self._get_collection(collection_name)
            return {
                "name": collection_name,
                "count": collection.count(),
                "dimension": 384,  # all-MiniLM-L6-v2 output dimension
            }
        except Exception as e:
            return {"error": str(e)}

    async def list_collections(self) -> List[str]:
        """List all collections."""
        return [c.name for c in self._client.list_collections()]

    async def delete_collection(self, collection_name: str) -> None:
        """Delete a collection."""
        sanitized_name = self._sanitize_name(collection_name)
        self._client.delete_collection(name=sanitized_name)
        if sanitized_name in self._collections:
            del self._collections[sanitized_name]