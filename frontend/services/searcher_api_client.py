"""
Hybrid Search API Client

A Python client library for interacting with the Hybrid Search API.
"""

import requests
from typing import List, Optional, Dict, Any, Literal
from dataclasses import dataclass


# --- Data classes ---

@dataclass
class SearchTerm:
    """A single search term with polarity."""
    term: str
    polarity: Literal["=", "!="]
    value: str


@dataclass
class SearchResultItem:
    """A single result returned by the search API."""
    id: Optional[str]
    content: str
    score: Optional[float]
    metadata: Dict[str, Any]


@dataclass
class SearchResponse:
    """Full response from the search endpoint."""
    search_terms: List[SearchTerm]
    results: List[SearchResultItem]
    total_results: int
    executed_at: str


@dataclass
class HealthResponse:
    """Response from the health endpoint."""
    status: str
    timestamp: str
    document_count: int
    embedding_model: str


# --- Client ---

class SearchAPIClient:
    """Client for the Hybrid Search API."""

    def __init__(self, base_url: str = "http://localhost:8002"):
        """
        Initialize the API client.

        Args:
            base_url: Base URL of the search API server.
        """
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def health_check(self) -> bool:
        """
        Check if the API is healthy.

        Returns:
            True if healthy, False otherwise.
        """
        try:
            response = self.session.get(f"{self.base_url}/health")
            return response.status_code == 200
        except requests.RequestException:
            return False

    def health(self) -> HealthResponse:
        """
        Return detailed health information from the API.

        Returns:
            HealthResponse with status, document count, and model info.
        """
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        data = response.json()
        return HealthResponse(
            status=data["status"],
            timestamp=data["timestamp"],
            document_count=data["document_count"],
            embedding_model=data["embedding_model"],
        )

    def search(
        self,
        search_terms: List[SearchTerm],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> SearchResponse:
        """
        Perform a hybrid search using one or more search terms.

        Args:
            search_terms: List of SearchTerm objects specifying what to look for.
            top_k:        Maximum number of results to return (default: 10).
            filters:      Optional Haystack metadata filters applied to both
                          the embedding and keyword retrievers.

        Returns:
            SearchResponse containing matched documents and metadata.

        Example:
            results = client.search([
                SearchTerm(term="Diagnose",   polarity="=",  value="Diabetes"),
                SearchTerm(term="Medikament", polarity="!=", value="Insulin"),
            ])
        """
        payload: Dict[str, Any] = {
            "search_terms": [
                {"term": t.term, "polarity": t.polarity, "value": t.value}
                for t in search_terms
            ],
            "top_k": top_k,
        }
        if filters is not None:
            payload["filters"] = filters

        response = self.session.post(
            f"{self.base_url}/search",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        return SearchResponse(
            search_terms=[
                SearchTerm(
                    term=t["term"],
                    polarity=t["polarity"],
                    value=t["value"],
                )
                for t in data["search_terms"]
            ],
            results=[
                SearchResultItem(
                    id=r["id"],
                    content=r["content"],
                    score=r["score"],
                    metadata=r["metadata"],
                )
                for r in data["results"]
            ],
            total_results=data["total_results"],
            executed_at=data["executed_at"],
        )