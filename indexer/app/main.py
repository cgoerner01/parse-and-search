"""
FastAPI application for document indexing using Haystack.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
from pathlib import Path
import shutil
import uuid
import os
import json
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

from haystack import Document

from indexer_service import DoclingIndexerService
from haystack import Document

import logging
import sys

logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)

loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
print(loggers)
for logger in loggers:
    logger.setLevel(logging.DEBUG)

jobs_db = {}

#Configuration
CONVERTED_DIR = Path(os.getenv("INDEXED_DOCS_PATH", "/data/outputs"))
EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "intfloat/multilingual-e5-large-instruct")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")

# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting up FastAPI application...")
    global indexing_service
    indexing_service = DoclingIndexerService()
    print("Indexing service initialized")
    yield
    # Shutdown
    print("Shutting down FastAPI application...")

app = FastAPI(
    title="Docling Document Indexer API",
    description="API for indexing documents with Haystack",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class IndexRequest(BaseModel):
    pipeline_type: Literal["simple", "metadata_extractor"] = Field(
        default="simple",
        description="Type of indexing pipeline to use"
    )
    recreate_table: bool = Field(
        default=False,
        description="Whether to recreate the document store table"
    )


class IndexResponse(BaseModel):
    job_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    message: str
    created_at: str

class IndexDocumentStoreConfig(BaseModel):
    connection_string: Optional[str]
    embedding_dimension: Optional[int]
    language: Optional[str]
    vector_function: Optional[Literal["cosine_similarity", "inner_product"]]
    recreate_table: Optional[bool]
    search_strategy: Optional[Literal["exact_nearest_neighbor", "hnsw"]]

class DocumentUploadRequest(BaseModel):
    content: str = Field(..., description="Document text content")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Document metadata")

class StatsResponse(BaseModel):
    total_documents: int
    unique_patients: Optional[int] = None
    index_info: Dict[str, Any]

class JobStatus(BaseModel):
    job_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    message: str
    created_at: str
    completed_at: Optional[str] = None
    document_count: Optional[int] = None
    error: Optional[str] = None

# Helper functions
async def process_indexing(
    job_id: str,
    documents: List[Document],
    pipeline_type: str,
    document_store_config: IndexDocumentStoreConfig = None
):
    """Background task to process document indexing."""
    global indexing_service
    
    try:
        jobs_db[job_id]["status"] = "processing"
        
        indexing_service.set_document_store(**(document_store_config.dict() if document_store_config else {}))
        
        # Initialize appropriate pipeline
        if pipeline_type == "simple":
            indexing_service.init_simple_pipeline()
        elif pipeline_type == "metadata_extractor":
            indexing_service.init_metadata_extractor_pipeline()
        else:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")
        
        # Index documents
        indexing_service.index_documents(documents)
        
        # Update job status
        jobs_db[job_id]["status"] = "completed"
        jobs_db[job_id]["completed_at"] = datetime.now().isoformat()
        jobs_db[job_id]["document_count"] = len(documents)
        jobs_db[job_id]["message"] = f"Successfully indexed {len(documents)} document(s)"
        
    except Exception as e:
        jobs_db[job_id]["status"] = "failed"
        jobs_db[job_id]["completed_at"] = datetime.now().isoformat()
        jobs_db[job_id]["error"] = str(e)
        jobs_db[job_id]["message"] = f"Indexing failed: {str(e)}"

# API Endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Document Indexing API",
        "version": "1.0.0",
        "endpoints": {
            "POST /index/upload": "Upload and index documents from text",
            "POST /index/files": "Upload and index documents from converted files",
            "GET /index/status/{job_id}": "Check indexing job status",
            "GET /stats": "Get index statistics",
            "GET /health": "Health check"
        }
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    try:
        # Check if document store is accessible
        doc_count = indexing_service.document_store.count_documents()
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "document_count": doc_count,
            "embedding_model": EMBED_MODEL_ID
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.post("/index/upload", response_model=IndexResponse, tags=["Indexing"])
async def index_documents_from_text(
    background_tasks: BackgroundTasks,
    documents: List[DocumentUploadRequest],
    pipeline_type: Literal["simple", "metadata_extractor"] = "simple",
    recreate_table: bool = False
):
    """
    Index documents from raw text content.
    
    - **documents**: List of documents with content and optional metadata
    - **pipeline_type**: Type of indexing pipeline
        - `simple`: Basic indexing without metadata extraction
        - `metadata_extractor`: Extract metadata using LLM
    - **document_store_config**: PgvectorDocumentStore config
    """
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Convert to Haystack documents
    haystack_docs = []
    for i, doc_req in enumerate(documents):
        meta = doc_req.metadata or {}
        meta["upload_id"] = job_id
        meta["document_index"] = i
        
        haystack_docs.append(Document(
            content=doc_req.content,
            meta=meta
        ))
    
    # Create job entry
    jobs_db[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "message": f"Indexing job created for {len(haystack_docs)} document(s)",
        "created_at": datetime.now().isoformat(),
        "pipeline_type": pipeline_type,
        "document_count": len(haystack_docs)
    }
    
    # Add background task
    background_tasks.add_task(
        process_indexing,
        job_id,
        haystack_docs,
        pipeline_type,
        document_store_config
    )
    
    return IndexResponse(
        job_id=job_id,
        status="pending",
        message=f"Indexing job submitted for {len(haystack_docs)} document(s)",
        created_at=jobs_db[job_id]["created_at"]
    )


@app.post("/index/files", response_model=IndexResponse, tags=["Indexing"])
async def index_documents_from_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Text files from conversion (.txt)"),
    pipeline_type: Literal["simple", "metadata_extractor"] = "simple",
    document_store_config: Optional[str] = Form(None, description="JSON string of IndexDocumentStoreConfig")
):
    """
    Index documents from converted text files.
    
    - **files**: One or more .txt files from Docling conversion
    - **pipeline_type**: Type of indexing pipeline
    - **document_store_config**: PgvectorDocumentStore config
    """
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded files and read content
    haystack_docs = []
    job_upload_dir = CONVERTED_DIR / job_id
    job_upload_dir.mkdir(parents=True, exist_ok=True)

    # Parse document store config if provided
    document_store_config_obj = None
    if document_store_config:
        try:
            config_dict = json.loads(document_store_config)
            document_store_config_obj = IndexDocumentStoreConfig(**config_dict)
            print(f"Parsed document store config: {document_store_config_obj}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid document_store_config: {str(e)}")
    
    try:
        for file in files:
            if not file.filename.lower().endswith('.txt'):
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} is not a .txt file. Only text files are supported."
                )
            
            # Save file
            file_path = job_upload_dir / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Read content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Create Haystack document
            haystack_docs.append(Document(
                content=content,
                meta={
                    "filename": file.filename,
                    "upload_id": job_id,
                    "file_path": str(file_path)
                }
            ))
    
    except Exception as e:
        # Clean up on error
        shutil.rmtree(job_upload_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")
    
    # Create job entry
    jobs_db[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "message": f"Indexing job created for {len(haystack_docs)} document(s)",
        "created_at": datetime.now().isoformat(),
        "pipeline_type": pipeline_type,
        "document_count": len(haystack_docs)
    }
    
    # Add background task
    background_tasks.add_task(
        process_indexing,
        job_id,
        haystack_docs,
        pipeline_type,
        document_store_config_obj
    )
    
    return IndexResponse(
        job_id=job_id,
        status="pending",
        message=f"Indexing job submitted for {len(haystack_docs)} file(s)",
        created_at=jobs_db[job_id]["created_at"]
    )


@app.get("/index/status/{job_id}", response_model=JobStatus, tags=["Indexing"])
async def get_indexing_status(job_id: str):
    """
    Get the status of an indexing job.
    
    - **job_id**: The job ID returned from the /index endpoints
    """
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job = jobs_db[job_id]
    return JobStatus(**job)

@app.get("/stats", response_model=StatsResponse, tags=["Stats"])
async def get_stats():
    """Get statistics about the indexed documents."""
    try:
        total_docs = indexing_service.document_store.count_documents()
        
        # Try to get unique patient count if metadata extraction was used
        unique_patients = None
        try:
            # This is a simple approximation - in production you'd query the DB directly
            all_docs = indexing_service.document_store.filter_documents()
            patient_ids = set()
            for doc in all_docs:
                if "extracted_metadata" in doc.meta:
                    patient_id = doc.meta["extracted_metadata"].get("patient_id")
                    if patient_id:
                        patient_ids.add(patient_id)
            unique_patients = len(patient_ids) if patient_ids else None
        except Exception:
            pass
        
        return StatsResponse(
            total_documents=total_docs,
            unique_patients=unique_patients,
            index_info={
                "embedding_model": EMBED_MODEL_ID,
                "vector_function": "cosine_similarity"
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


# @app.delete("/index/clear", tags=["Maintenance"])
# async def clear_index():
#     """Clear all documents from the index (recreate table)."""
#     try:
#         global indexing_service
#         indexing_service.set_document_store()
#         indexing_service.init_simple_pipeline()
        
#         return {
#             "status": "success",
#             "message": "Index cleared successfully"
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to clear index: {str(e)}")


@app.get("/jobs", tags=["Indexing"])
async def list_jobs(
    status: Optional[Literal["pending", "processing", "completed", "failed"]] = None
):
    """
    List all indexing jobs, optionally filtered by status.
    
    - **status**: Filter jobs by status (optional)
    """
    jobs = list(jobs_db.values())
    
    if status:
        jobs = [job for job in jobs if job["status"] == status]
    
    return {
        "total": len(jobs),
        "jobs": jobs
    }
