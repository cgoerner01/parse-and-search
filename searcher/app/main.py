"""
FastAPI application for hybrid search over a PgvectorDocumentStore.
Combines semantic embedding search with keyword-based retrieval.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from contextlib import asynccontextmanager
import os
from datetime import datetime

from search_service import HybridSearchService


EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "intfloat/multilingual-e5-large-instruct")

# Global search service instance
search_service: Optional[HybridSearchService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global search_service
    print("Initializing hybrid search service...")
    search_service = HybridSearchService()
    print("Hybrid search service ready!")
    yield
    print("Shutting down...")


app = FastAPI(
    title="Hybrid Search API",
    description="Semantic + keyword hybrid search over indexed medical documents",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic models ---

class SearchTerm(BaseModel):
    term: str = Field(..., description="The field or concept to search for")
    polarity: Literal["=", "!="] = Field(
        default="=",
        description="Whether the term should be present (=) or absent (!=)"
    )
    value: str = Field(..., description="The value to match against the term")


class SearchRequest(BaseModel):
    search_terms: List[SearchTerm] = Field(
        ...,
        min_length=1,
        description="List of search terms to combine in the hybrid search"
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results to return after joining"
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional Haystack metadata filters applied to both retrievers"
    )


class SearchResultItem(BaseModel):
    id: Optional[str]
    content: str
    score: Optional[float]
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    search_terms: List[SearchTerm]
    results: List[SearchResultItem]
    total_results: int
    executed_at: str


# --- Endpoints ---

@app.get("/health", tags=["Health"])
async def health_check():
    try:
        doc_count = search_service.document_store.count_documents()
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "document_count": doc_count,
            "embedding_model": EMBED_MODEL_ID,
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def hybrid_search(request: SearchRequest):
    """
    Perform hybrid search using a list of search terms.

    Each term has:
    - **term**: the field or concept (e.g. `"Diagnose"`)
    - **polarity**: `=` to require it, `!=` to exclude it
    - **value**: the value to match (e.g. `"Diabetes"`)

    Results from all terms are merged via reciprocal rank fusion.

    Example request body:
    ```json
    {
      "search_terms": [
        {"term": "Diagnose", "polarity": "=", "value": "Diabetes"},
        {"term": "Medikament", "polarity": "!=", "value": "Insulin"}
      ],
      "top_k": 10
    }
    ```
    """
    try:
        raw_terms = [t.model_dump() for t in request.search_terms]
        documents = search_service.search(
            search_terms=raw_terms,
            top_k=request.top_k,
            filters=request.filters,
        )

        results = [
            SearchResultItem(
                id=doc.id,
                content=doc.content,
                score=doc.score if hasattr(doc, "score") else None,
                metadata=doc.meta or {},
            )
            for doc in documents
        ]

        return SearchResponse(
            search_terms=request.search_terms,
            results=results,
            total_results=len(results),
            executed_at=datetime.now().isoformat(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Hybrid Search API",
        "version": "1.0.0",
        "endpoints": {
            "POST /search": "Hybrid search with search terms",
            "GET /health": "Health check with document count",
        },
    }