"""
PDF Converter API Client

A Python client library for interacting with the Docling-based PDF Converter API.
"""

import requests
import time
from pathlib import Path
from typing import List, Optional, Literal
from dataclasses import dataclass


@dataclass
class JobStatus:
    """Represents the status of a conversion job."""
    job_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    message: str
    created_at: str
    completed_at: Optional[str] = None
    output_files: Optional[List[str]] = None
    error: Optional[str] = None


class PDFConverterAPIClient:
    """Client for the Docling-based PDF Converter API."""
    
    def __init__(self, base_url: str = "http://converter:8002"):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def convert_pdfs(
        self,
        pdf_paths: List[Path],
        pipeline_type: Literal["rapidocr", "tesseract", "easyocr", "vlm"] = "rapidocr",
        wait: bool = False,
        poll_interval: float = 2.0,
        timeout: Optional[float] = None
    ) -> JobStatus:
        """
        Upload and convert PDF files.
        
        Args:
            pdf_paths: List of paths to PDF files
            pipeline_type: Type of conversion pipeline to use
            wait: If True, wait for conversion to complete
            poll_interval: Seconds between status checks when waiting
            timeout: Maximum seconds to wait (None for no timeout)
            
        Returns:
            JobStatus object
            
        Raises:
            requests.RequestException: On API errors
            TimeoutError: If timeout is exceeded while waiting
        """
        # Prepare files for upload
        files = [
            ('files', (path.name, open(path, 'rb'), 'application/pdf'))
            for path in pdf_paths
        ]
        
        try:
            # Submit conversion job
            response = self.session.post(
                f"{self.base_url}/convert",
                files=files,
                params={'pipeline_type': pipeline_type}
            )
            response.raise_for_status()
            
            data = response.json()
            job_status = JobStatus(
                job_id=data['job_id'],
                status=data['status'],
                message=data['message'],
                created_at=data['created_at']
            )
            
            # Wait for completion if requested
            if wait:
                return self._wait_for_completion(
                    job_status.job_id,
                    poll_interval,
                    timeout
                )
            
            return job_status
            
        finally:
            # Close file handles
            for _, (_, file_obj, _) in files:
                file_obj.close()
    
    def get_status(self, job_id: str) -> JobStatus:
        """
        Get the status of a conversion job.
        
        Args:
            job_id: The job ID
            
        Returns:
            JobStatus object
            
        Raises:
            requests.RequestException: On API errors
        """
        response = self.session.get(f"{self.base_url}/status/{job_id}")
        response.raise_for_status()
        
        data = response.json()
        return JobStatus(**data)
    
    def download_file(self, job_id: str, filename: str, output_path: Path):
        """
        Download a converted file.
        
        Args:
            job_id: The job ID
            filename: Name of the file to download
            output_path: Path where to save the downloaded file
            
        Raises:
            requests.RequestException: On API errors
        """
        response = self.session.get(
            f"{self.base_url}/download/{job_id}/{filename}",
            stream=True
        )
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    
    def download_all(self, job_id: str, output_dir: Path) -> List[Path]:
        """
        Download all converted files for a job.
        
        Args:
            job_id: The job ID
            output_dir: Directory where to save downloaded files
            
        Returns:
            List of paths to downloaded files
            
        Raises:
            requests.RequestException: On API errors
        """
        # Get job status to get list of output files
        status = self.get_status(job_id)
        
        if status.status != "completed":
            raise ValueError(f"Job {job_id} is not completed (status: {status.status})")
        
        if not status.output_files:
            return []
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Download all files
        downloaded_paths = []
        for filename in status.output_files:
            output_path = output_dir / filename
            self.download_file(job_id, filename, output_path)
            downloaded_paths.append(output_path)
        
        return downloaded_paths
    
    def list_jobs(
        self,
        status: Optional[Literal["pending", "processing", "completed", "failed"]] = None
    ) -> List[JobStatus]:
        """
        List all jobs, optionally filtered by status.
        
        Args:
            status: Filter by job status (optional)
            
        Returns:
            List of JobStatus objects
            
        Raises:
            requests.RequestException: On API errors
        """
        params = {'status': status} if status else {}
        response = self.session.get(f"{self.base_url}/jobs", params=params)
        response.raise_for_status()
        
        data = response.json()
        return [JobStatus(**job) for job in data['jobs']]
    
    def cleanup(self, max_age_hours: int = 1):
        """
        Trigger cleanup of old temporary files.
        
        Args:
            max_age_hours: Maximum age of files to keep
            
        Raises:
            requests.RequestException: On API errors
        """
        response = self.session.delete(
            f"{self.base_url}/cleanup",
            params={'max_age_hours': max_age_hours}
        )
        response.raise_for_status()
        return response.json()
    
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
    
    def _wait_for_completion(
        self,
        job_id: str,
        poll_interval: float,
        timeout: Optional[float]
    ) -> JobStatus:
        """
        Wait for a job to complete.
        
        Args:
            job_id: The job ID
            poll_interval: Seconds between status checks
            timeout: Maximum seconds to wait (None for no timeout)
            
        Returns:
            JobStatus object when complete
            
        Raises:
            TimeoutError: If timeout is exceeded
            requests.RequestException: On API errors
        """
        start_time = time.time()
        
        while True:
            status = self.get_status(job_id)
            
            if status.status in ["completed", "failed"]:
                return status
            
            # Check timeout
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    raise TimeoutError(
                        f"Job {job_id} did not complete within {timeout} seconds"
                    )
            
            time.sleep(poll_interval)