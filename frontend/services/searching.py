import nltk
import os
from Levenshtein import distance
from haystack import component, Document, Pipeline
from typing import List, Dict
from haystack_integrations.components.retrievers.pgvector import PgvectorEmbeddingRetriever, PgvectorKeywordRetriever
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.joiners.document_joiner import DocumentJoiner

EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "intfloat/multilingual-e5-large-instruct")

class HybridSearchService:
    """
    Service to perform hybrid search on medical documents.
    Combines semantic search with keyword-based filtering.
    """
    
    def __init__(self, document_store: PgvectorDocumentStore = None, keyword_weight: float = 0.5, semantic_weight: float = 0.5):
        self.document_store = document_store or PgvectorDocumentStore(
            embedding_dimension=1024,
            language="german",
            vector_function="cosine_similarity",
            recreate_table=False,
            search_strategy="exact_nearest_neighbor",
        )

        self.query_embedder = SentenceTransformersTextEmbedder(
            model=EMBED_MODEL_ID,
            prefix="Instruct: Suche in den gegebenen Passagen von Arztbriefen die Passagen, auf die die Query zutrifft.\nQuery: ",
        )
        self.query_embedder.warm_up()

        self.embedding_retriever = PgvectorEmbeddingRetriever(
            document_store=self.document_store,
            top_k=10
        )
        self.keyword_retriever = PgvectorKeywordRetriever(
            document_store=self.document_store,
            top_k=10
        )

        self.joiner = DocumentJoiner(join_mode="merge")

        self.query_embedder.warm_up()

        self.search_pipeline = Pipeline()
        self.search_pipeline.add_component("query_embedder", self.query_embedder)
        self.search_pipeline.add_component("embedding_retriever", self.embedding_retriever)
        self.search_pipeline.add_component("keyword_retriever", self.keyword_retriever)
        self.search_pipeline.add_component("joiner", self.joiner)

        self.search_pipeline.connect("query_embedder.embedding", "embedding_retriever")
        self.search_pipeline.connect("embedding_retriever.documents", "joiner")
        self.search_pipeline.connect("keyword_retriever.documents", "joiner")

    def search(self, search_terms: List[Dict[str, str]]) -> Dict[str, List[Document]]:
        """
        Perform hybrid search on the document store using multiple search terms.
        Each search term is a dictionary with 'term', 'polarity', and 'value'.
        """
        #TODO
        queries = [f"{term['term']} {term['value']}" if term['polarity'] == "=" else f"{term['term']} nicht {term['value']}" for term in search_terms]
        results = []
        for query in queries:
            print(f"Searching for term: {query}")
            params = {
                "query_embedder": {"text": query},
                "keyword_retriever": {"query": query},
                #"retriever": {"query": query},
            }
            results.append(self.search_pipeline.run(params))
        
        # combine results
        joiner = DocumentJoiner(join_mode="merge")
        joined_results = joiner.run([r["joiner"]["documents"] for r in results])

        return joined_results["documents"]