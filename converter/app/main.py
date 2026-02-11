"""
FastAPI application for PDF conversion using Docling.
Supports OCR, deskewing, and VLM-based conversion pipelines.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from pathlib import Path
import shutil
import uuid
import os
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

from converter_service import DoclingConvertingService

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

# Configuration
UPLOAD_DIR = Path(os.getenv("UPLOADED_FILES_PATH", "/data/uploads"))
OUTPUT_DIR = Path(os.getenv("INDEXED_DOCS_PATH", "/data/outputs"))
PREPROCESS_DIR = Path(os.getenv("PREPROCESSED_FILES_PATH", "/data/preprocessed"))
RAPIDOCR_MODELS_PATH = Path(os.getenv("RAPIDOCR_MODELS_PATH", "/data/rapidocr_models"))

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PREPROCESS_DIR.mkdir(parents=True, exist_ok=True)

# Store active jobs
jobs_db = {}


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting up FastAPI application...")
    yield
    # Shutdown
    print("Shutting down FastAPI application...")
    # Clean up temporary files older than 1 hour
    cleanup_old_files()


app = FastAPI(
    title="Docling PDF Converter API",
    description="API for converting PDF documents using Docling with OCR, deskewing, and VLM support",
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
class ConversionRequest(BaseModel):
    pipeline_type: Literal["ocr", "ocr_deskew", "vlm"] = Field(
        default="ocr",
        description="Type of conversion pipeline to use"
    )


class ConversionResponse(BaseModel):
    job_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    message: str
    created_at: str


class JobStatus(BaseModel):
    job_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    message: str
    created_at: str
    completed_at: Optional[str] = None
    output_files: Optional[List[str]] = None
    error: Optional[str] = None


# Helper functions
def cleanup_old_files(max_age_hours: int = 1):
    """Remove files older than max_age_hours from upload and output directories."""
    import time
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    
    for directory in [UPLOAD_DIR, OUTPUT_DIR, PREPROCESS_DIR]:
        for file_path in directory.glob("*"):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    try:
                        file_path.unlink()
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")


async def process_conversion(
    job_id: str,
    file_paths: List[Path],
    pipeline_type: str
):
    """Background task to process PDF conversion."""
    try:
        logging.getLogger("haystack").debug("HAYSTACK DEBUG TEST")
        print("PRINT TEST")
        jobs_db[job_id]["status"] = "processing"
        
        # Create job-specific output directory
        job_output_dir = OUTPUT_DIR / job_id
        job_output_dir.mkdir(parents=True, exist_ok=True)
        
        job_preprocess_dir = PREPROCESS_DIR / job_id
        job_preprocess_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize the service
        service = DoclingConvertingService(
            preprocess_dir=job_preprocess_dir,
            output_dir=job_output_dir,
            rapidocr_models_path=RAPIDOCR_MODELS_PATH
        )
        
        # Initialize appropriate pipeline
        if pipeline_type == "rapidocr":
            service.init_rapidocr_pipeline()
        elif pipeline_type == "tesseract":
            service.init_tesseract_pipeline()
        elif pipeline_type == "easyocr":
            service.init_easyocr_pipeline()
        elif pipeline_type == "vlm":
            service.init_vlm_pipeline()
        else:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")
        
        # Run conversion
        result = service.convert_documents(file_paths)
        
        # Get output files
        output_files = [f.name for f in job_output_dir.glob("*.txt")]
        
        # Update job status
        jobs_db[job_id]["status"] = "completed"
        jobs_db[job_id]["completed_at"] = datetime.now().isoformat()
        jobs_db[job_id]["output_files"] = output_files
        jobs_db[job_id]["message"] = f"Conversion completed successfully. Generated {len(output_files)} output file(s)."
        
    except Exception as e:
        jobs_db[job_id]["status"] = "failed"
        jobs_db[job_id]["completed_at"] = datetime.now().isoformat()
        jobs_db[job_id]["error"] = str(e)
        jobs_db[job_id]["message"] = f"Conversion failed: {str(e)}"


# API Endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Docling PDF Converter API",
        "version": "1.0.0",
        "endpoints": {
            "POST /convert": "Upload and convert PDF files",
            "GET /status/{job_id}": "Check conversion job status",
            "GET /download/{job_id}/{filename}": "Download converted file",
            "GET /health": "Health check"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/convert", response_model=ConversionResponse, tags=["Conversion"])
async def convert_pdfs(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="PDF files to convert"),
    pipeline_type: Literal["rapidocr", "tesseract", "easyocr", "vlm"] = "rapidocr"
):
    """
    Upload and convert PDF files using the specified pipeline.
    
    - **files**: One or more PDF files to convert
    - **pipeline_type**: Type of conversion pipeline
        - `rapidocr`: Standard OCR pipeline with RapidOCR
        - `tesseract`: OCR pipeline using Tesseract
        - `easyocr`: OCR pipeline using EasyOCR
        - `vlm`: Vision Language Model pipeline
    
    Returns a job_id that can be used to check status and download results.
    """
    # Validate files
    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail=f"File {file.filename} is not a PDF. Only PDF files are supported."
            )
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded files
    saved_paths = []
    job_upload_dir = UPLOAD_DIR / job_id
    job_upload_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        for file in files:
            file_path = job_upload_dir / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_paths.append(file_path)
    except Exception as e:
        # Clean up on error
        shutil.rmtree(job_upload_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Error saving files: {str(e)}")
    
    # Create job entry
    jobs_db[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "message": f"Conversion job created for {len(files)} file(s)",
        "created_at": datetime.now().isoformat(),
        "pipeline_type": pipeline_type,
        "file_count": len(files)
    }
    
    # Add background task
    background_tasks.add_task(
        process_conversion,
        job_id,
        saved_paths,
        pipeline_type
    )
    
    return ConversionResponse(
        job_id=job_id,
        status="pending",
        message=f"Conversion job submitted for {len(files)} file(s) using {pipeline_type} pipeline",
        created_at=jobs_db[job_id]["created_at"]
    )


@app.get("/status/{job_id}", response_model=JobStatus, tags=["Conversion"])
async def get_job_status(job_id: str):
    """
    Get the status of a conversion job.
    
    - **job_id**: The job ID returned from the /convert endpoint
    """
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job = jobs_db[job_id]
    return JobStatus(**job)


@app.get("/download/{job_id}/{filename}", tags=["Conversion"])
async def download_file(job_id: str, filename: str):
    """
    Download a converted file.
    
    - **job_id**: The job ID returned from the /convert endpoint
    - **filename**: Name of the output file to download
    """
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job = jobs_db[job_id]
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} is not completed. Current status: {job['status']}"
        )
    
    file_path = OUTPUT_DIR / job_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File {filename} not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="text/plain"
    )


@app.get("/jobs", tags=["Conversion"])
async def list_jobs(
    status: Optional[Literal["pending", "processing", "completed", "failed"]] = None
):
    """
    List all conversion jobs, optionally filtered by status.
    
    - **status**: Filter jobs by status (optional)
    """
    jobs = list(jobs_db.values())
    
    if status:
        jobs = [job for job in jobs if job["status"] == status]
    
    return {
        "total": len(jobs),
        "jobs": jobs
    }


@app.delete("/cleanup", tags=["Maintenance"])
async def cleanup_files(max_age_hours: int = 1):
    """
    Clean up old temporary files.
    
    - **max_age_hours**: Maximum age of files to keep (default: 1 hour)
    """
    try:
        cleanup_old_files(max_age_hours)
        return {
            "status": "success",
            "message": f"Cleaned up files older than {max_age_hours} hour(s)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")