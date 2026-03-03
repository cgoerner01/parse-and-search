"""
Hybrid Search Service - combines semantic and keyword retrieval over PgvectorDocumentStore.
"""

import os
from haystack import component, Document, Pipeline
from haystack.components.joiners.document_joiner import DocumentJoiner
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.pgvector import (
    PgvectorEmbeddingRetriever,
    PgvectorKeywordRetriever,
)
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack.utils import Secret
from typing import List, Dict, Any, Optional

EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "intfloat/multilingual-e5-large-instruct")


class HybridSearchService:
    """
    Service to perform hybrid search on medical documents.
    Combines semantic (embedding) search with keyword-based retrieval,
    then merges results via reciprocal rank fusion.
    """

    def __init__(
        self,
        document_store: PgvectorDocumentStore = None,
        retriever_top_k: int = 10,
    ):
        self.document_store = document_store or PgvectorDocumentStore(
            connection_string=Secret.from_env_var("PG_CONN_STR"),
            embedding_dimension=1024,
            language="german",
            vector_function="cosine_similarity",
            recreate_table=False,
            search_strategy="exact_nearest_neighbor",
        )

        self.retriever_top_k = retriever_top_k

        # Embedder for query text
        self.query_embedder = SentenceTransformersTextEmbedder(
            model=EMBED_MODEL_ID,
            prefix=(
                "Instruct: Suche in den gegebenen Passagen von Arztbriefen "
                "die Passagen, auf die die Query zutrifft.\nQuery: "
            ),
        )
        self.query_embedder.warm_up()

        # Per-query pipeline (retrievers are stateless; we build once and reuse)
        self.embedding_retriever = PgvectorEmbeddingRetriever(
            document_store=self.document_store,
            top_k=self.retriever_top_k,
        )
        self.keyword_retriever = PgvectorKeywordRetriever(
            document_store=self.document_store,
            top_k=self.retriever_top_k,
        )

        self.search_pipeline = Pipeline()
        self.search_pipeline.add_component("query_embedder", self.query_embedder)
        self.search_pipeline.add_component("embedding_retriever", self.embedding_retriever)
        self.search_pipeline.add_component("keyword_retriever", self.keyword_retriever)
        self.search_pipeline.add_component(
            "joiner", DocumentJoiner(join_mode="reciprocal_rank_fusion")
        )

        self.search_pipeline.connect("query_embedder.embedding", "embedding_retriever")
        self.search_pipeline.connect("embedding_retriever.documents", "joiner")
        self.search_pipeline.connect("keyword_retriever.documents", "joiner")

    def _build_query(self, term: Dict[str, str]) -> str:
        """Turn a search term dict into a natural-language query string."""
        if term["polarity"] == "=":
            return f"{term['term']} {term['value']}"
        else:
            return f"{term['term']} nicht {term['value']}"

    def search(
        self,
        search_terms: List[Dict[str, str]],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Perform hybrid search using multiple search terms.

        Each search term is a dict with:
          - 'term':     the field or concept to search for
          - 'polarity': '=' (include) or '!=' (exclude)
          - 'value':    the value to match

        Results from every term are retrieved independently and then
        merged with reciprocal rank fusion before being returned.

        Args:
            search_terms: List of term dicts.
            top_k:        How many documents to return after final join.
            filters:      Optional Haystack metadata filters forwarded to
                          both retrievers.

        Returns:
            List of Documents, ranked by fused score.
        """
        per_term_docs: List[List[Document]] = []

        for term in search_terms:
            query = self._build_query(term)
            print(f"Searching for: {query!r}")

            params: Dict[str, Any] = {
                "query_embedder": {"text": query},
                "keyword_retriever": {"query": query},
            }
            if filters:
                params["embedding_retriever"] = {"filters": filters}
                params["keyword_retriever"]["filters"] = filters

            result = self.search_pipeline.run(params)
            per_term_docs.append(result["joiner"]["documents"])

        if not per_term_docs:
            return []

        # Final merge across all terms
        final_joiner = DocumentJoiner(join_mode="reciprocal_rank_fusion")
        merged = final_joiner.run(per_term_docs)
        documents = merged["documents"]

        return documents[:top_k]