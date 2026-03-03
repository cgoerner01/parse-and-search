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

class KeywordDistance:
    """Calculate distance between query and document text based on keyword presence."""

    def __init__(self, distance_threshold: int = 5):
        self.distance_threshold = distance_threshold

    def levenshtein_distance(self, text: str, query: str) -> tuple[int, str]:
        text_lower = text.lower()
        query_lower = query.lower()

        # Check with Levenshtein distance for typos
        text_lower_as_list = text_lower.split()
        query_lower_as_list = query_lower.split()
        query_length = len(query_lower_as_list)
        text_length = len(text_lower_as_list)
        nearest_match = None
        lowest = None

        if query_length == 1:
            for word in text_lower_as_list:
                dist = distance(word, query_lower)
                if not lowest or dist < lowest:
                    lowest = dist
                    nearest_match = word
        else:
            i = 0
            j = query_length
            while j < len(text_lower_as_list) + 1:
                text_slice = text_lower_as_list[i:j]
                text_joined = ' '.join(text_slice)
                dist = distance(text_joined, query_lower)
                if not lowest or dist < lowest:
                    lowest = dist
                    nearest_match = text_joined
                i += 1
                j += 1
        
        return lowest, nearest_match

    def score(self, text: str, query: str) -> float:
        """
        Score a document based on keyword presence.
        Returns a score between 0 and 1.
        1 meaning the query is contained in the text (exact match), 0 meaning there is no match.
        """
        #TODO check that text and query are at least one word, text should be longer than query; maybe through a check method?
        text_lower = text.lower()
        query_lower = query.lower()
        score = 0.0

        # Exact diagnosis match (highest weight)
        if query_lower in text_lower:
            score = 1.0
            # Penalty for negations
            negation_patterns = [
                f"kein {query_lower}",
                f"keine {query_lower}",
                f"{query_lower} ausgeschlossen",
                f"{query_lower} verneint",
                f"nicht {query_lower}",
                f"ohne {query_lower}",
            ]

            for pattern in negation_patterns:
                if pattern in text_lower:
                    score = 0.0
            
        # Levenshtein distance for typos
        else:
            lowest, nearest_match = self.levenshtein_distance(text, query)

            if lowest <= self.distance_threshold:
                # TODO: hyperparameter
                score = 0.8

                # Penalty for negations
                negation_patterns = [
                    f"kein {nearest_match}",
                    f"keine {nearest_match}",
                    f"{nearest_match} ausgeschlossen",
                    f"{nearest_match} verneint",
                    f"nicht {nearest_match}",
                    f"ohne {nearest_match}",
                ]

                for pattern in negation_patterns:
                    if pattern in text_lower:
                        score = 0.0

        return max(0.0, min(1.0, score))
        
@component
class HybridMedicalRetriever:
    """
    Combines semantic search with keyword-based filtering for medical documents.
    """
    
    def __init__(self, document_store, semantic_weight: float = 0.5, keyword_weight: float = 0.5, top_k: int = 30):
        self.document_store = document_store
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.top_k = top_k
        self.keyword_extractor = KeywordDistance()
        
    @component.output_types(documents=List[Document])
    def run(self, query_embedding: List[float], query: str):
        """
        Retrieve documents using hybrid search.
        """
        # Step 1: Semantic retrieval (cast broader net)
        semantic_retriever = PgvectorEmbeddingRetriever(
            document_store=self.document_store, 
            top_k=self.top_k * 2  # Retrieve more for reranking
        )
        semantic_results = semantic_retriever.run(query_embedding=query_embedding)
        documents = semantic_results["documents"]
        
        # Step 2: Keyword-based scoring
        for doc in documents:
            keyword_score = self.keyword_extractor.score(
                doc.content, query
            )
            
            # Combine scores
            semantic_score = doc.score if doc.score else 0.5
            hybrid_score = (self.semantic_weight * semantic_score + 
                          self.keyword_weight * keyword_score)
            
            doc.score = hybrid_score
        
        # Step 3: Re-rank and filter
        documents.sort(key=lambda x: x.score if x.score else 0, reverse=True)
        
        # Return top_k documents
        return {"documents": documents[:self.top_k]}

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
        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight

        self.query_embedder = SentenceTransformersTextEmbedder(
            model=EMBED_MODEL_ID,
            prefix="Instruct: Suche in den gegebenen Passagen von Arztbriefen die Passagen, auf die die Query zutrifft. Query: ",
        )
        #TODO
        self.query_embedder.warm_up()

        self.retriever = HybridMedicalRetriever(
            document_store=self.document_store,
            semantic_weight=self.semantic_weight,
            keyword_weight=self.keyword_weight,
            top_k=10
        )

        self.search_pipeline = Pipeline()
        self.search_pipeline.add_component("query_embedder", self.query_embedder)
        self.search_pipeline.add_component("retriever", self.retriever)

        self.search_pipeline.connect("query_embedder.embedding", "retriever")

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
                "retriever": {"query": query},
            }
            results.append(self.search_pipeline.run(params))
        
        # combine results
        joiner = DocumentJoiner(join_mode="merge")
        joined_results = joiner.run([r["retriever"]["documents"] for r in results])

        return joined_results["documents"]