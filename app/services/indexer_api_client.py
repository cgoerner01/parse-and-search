"""
Document Indexing API Client

A Python client library for interacting with the Document Indexing API.
"""

import requests
import time
from pathlib import Path
import json
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
from dataclasses import dataclass, asdict

@dataclass
class IndexJobStatus:
    """Represents the status of an indexing job."""
    job_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    message: str
    created_at: str
    completed_at: Optional[str] = None
    document_count: Optional[int] = None
    error: Optional[str] = None

@dataclass
class IndexDocumentStoreConfig:
    connection_string: Optional[str]
    embedding_dimension: Optional[int]
    language: Optional[str]
    vector_function: Optional[Literal["cosine_similarity", "inner_product"]]
    recreate_table: Optional[bool]
    search_strategy: Optional[Literal["exact_nearest_neighbor", "hnsw"]]


class IndexerAPIClient:
    """Client for the Document Indexing API."""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def index_text_documents(
        self,
        documents: List[Dict[str, Any]],
        pipeline_type: Literal["simple", "metadata_extractor"] = "simple",
        document_store_config: IndexDocumentStoreConfig = None,
        wait: bool = False,
        poll_interval: float = 2.0,
        timeout: Optional[float] = None
    ) -> IndexJobStatus:
        """
        Index documents from raw text content.
        
        Args:
            documents: List of dicts with 'content' and optional 'metadata'
            pipeline_type: Type of indexing pipeline
            document_store_config: IndexDocumentStoreConfig object
            wait: If True, wait for indexing to complete
            poll_interval: Seconds between status checks when waiting
            timeout: Maximum seconds to wait (None for no timeout)
            
        Returns:
            IndexJobStatus object
        """
        response = self.session.post(
            f"{self.base_url}/index/upload",
            json=documents,
            params={
                'pipeline_type': pipeline_type,
                'document_store_config': asdict(document_store_config) if document_store_config else None

            }
        )
        response.raise_for_status()
        
        data = response.json()
        job_status = IndexJobStatus(
            job_id=data['job_id'],
            status=data['status'],
            message=data['message'],
            created_at=data['created_at']
        )
        
        if wait:
            return self._wait_for_indexing(
                job_status.job_id,
                poll_interval,
                timeout
            )
        
        return job_status
    
    def index_files(
        self,
        file_paths: List[Path],
        pipeline_type: Literal["simple", "metadata_extractor"] = "simple",
        document_store_config: IndexDocumentStoreConfig = None,
        wait: bool = False,
        poll_interval: float = 2.0,
        timeout: Optional[float] = None
    ) -> IndexJobStatus:
        """
        Index documents from text files.
        
        Args:
            file_paths: List of paths to .txt files
            pipeline_type: Type of indexing pipeline
            document_store_config: IndexDocumentStoreConfig object
            wait: If True, wait for indexing to complete
            poll_interval: Seconds between status checks when waiting
            timeout: Maximum seconds to wait (None for no timeout)
            
        Returns:
            IndexJobStatus object
        """
        files = [
            ('files', (path.name, open(path, 'rb'), 'text/plain'))
            for path in file_paths
        ]
        
        try:
            response = self.session.post(
                f"{self.base_url}/index/files",
                files=files,
                data={
                    'pipeline_type': pipeline_type,
                    'document_store_config': json.dumps(asdict(document_store_config)) if document_store_config else None
                }
            )
            response.raise_for_status()
            
            data = response.json()
            job_status = IndexJobStatus(
                job_id=data['job_id'],
                status=data['status'],
                message=data['message'],
                created_at=data['created_at']
            )
            
            if wait:
                return self._wait_for_indexing(
                    job_status.job_id,
                    poll_interval,
                    timeout
                )
            
            return job_status
            
        finally:
            for _, (_, file_obj, _) in files:
                file_obj.close()
    
    def get_indexing_status(self, job_id: str) -> IndexJobStatus:
        """
        Get the status of an indexing job.
        
        Args:
            job_id: The job ID
            
        Returns:
            IndexJobStatus object
        """
        response = self.session.get(f"{self.base_url}/index/status/{job_id}")
        response.raise_for_status()
        
        data = response.json()
        return IndexJobStatus(**data)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the indexed documents.
        
        Returns:
            Dictionary with statistics
        """
        response = self.session.get(f"{self.base_url}/stats")
        response.raise_for_status()
        return response.json()
    
    def clear_index(self):
        """Clear all documents from the index."""
        response = self.session.delete(f"{self.base_url}/index/clear")
        response.raise_for_status()
        return response.json()
    
    def list_jobs(
        self,
        status: Optional[Literal["pending", "processing", "completed", "failed"]] = None
    ) -> List[IndexJobStatus]:
        """
        List all indexing jobs, optionally filtered by status.
        
        Args:
            status: Filter by job status (optional)
            
        Returns:
            List of IndexJobStatus objects
        """
        params = {'status': status} if status else {}
        response = self.session.get(f"{self.base_url}/jobs", params=params)
        response.raise_for_status()
        
        data = response.json()
        return [IndexJobStatus(**job) for job in data['jobs']]
    
    def health_check(self) -> bool:
        """
        Check if the API is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            response = self.session.get(f"{self.base_url}/health")
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def _wait_for_indexing(
        self,
        job_id: str,
        poll_interval: float,
        timeout: Optional[float]
    ) -> IndexJobStatus:
        """
        Wait for an indexing job to complete.
        
        Args:
            job_id: The job ID
            poll_interval: Seconds between status checks
            timeout: Maximum seconds to wait (None for no timeout)
            
        Returns:
            IndexJobStatus object when complete
        """
        start_time = time.time()
        
        while True:
            status = self.get_indexing_status(job_id)
            
            if status.status in ["completed", "failed"]:
                return status
            
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    raise TimeoutError(
                        f"Indexing job {job_id} did not complete within {timeout} seconds"
                    )
            
            time.sleep(poll_interval)