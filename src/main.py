from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Form, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict
import os
import shutil
import time
import tempfile
import uuid
import json
import secrets
from datetime import datetime, timedelta
from collections import deque, defaultdict
from contextlib import asynccontextmanager
import psutil
import asyncio
import hashlib
from enum import Enum
import logging

from rag import RAGService, process_pdf

ALLOWED_ORIGINS = [
    "http://localhost:7680",
    "https://askbookie-pesu.vercel.app",
]

API_KEYS = {
    os.getenv("ADMIN_API_KEY", "your-admin-secret-key"): {"role": "admin", "name": "admin"},
    os.getenv("USER_API_KEY_1", "your-user-secret-key-1"): {"role": "user", "name": "user_1"},
    os.getenv("USER_API_KEY_2", "your-user-secret-key-2"): {"role": "user", "name": "user_2"},
    os.getenv("USER_API_KEY_3", "your-user-secret-key-3"): {"role": "user", "name": "user_3"},
}

MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_CONCURRENT_JOBS = 3
JOB_RETENTION_HOURS = 24
MAX_LATENCY_SAMPLES = 1000

MAX_RETRIES = 3
RETRY_DELAYS = [60, 300, 900]

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class JobStatusEnum(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    subject: str = Field(..., min_length=1, max_length=100)
    
    @field_validator('query', 'subject')
    @classmethod
    def strip_whitespace(cls, v):
        return v.strip()

class JobStatus(BaseModel):
    status: JobStatusEnum
    filename: str
    subject: str
    timestamp: str
    details: str = ""
    retry_count: int = 0
    error: Optional[str] = None

class Metrics:
    def __init__(self):
        self._lock = asyncio.Lock()
        self.start_time = datetime.now()
        self.per_key_metrics = defaultdict(lambda: {
            "api_calls": 0,
            "questions_asked": 0,
            "uploads_attempted": 0,
            "uploads_successful": 0,
            "failures": 0,
            "latencies": deque(maxlen=MAX_LATENCY_SAMPLES)
        })
        self.auth_failures = 0
    
    async def record_api_call(self, api_key: str):
        async with self._lock:
            self.per_key_metrics[api_key]["api_calls"] += 1
    
    async def record_question(self, api_key: str):
        async with self._lock:
            self.per_key_metrics[api_key]["questions_asked"] += 1
    
    async def record_upload_attempt(self, api_key: str):
        async with self._lock:
            self.per_key_metrics[api_key]["uploads_attempted"] += 1
    
    async def record_upload_success(self, api_key: str):
        async with self._lock:
            self.per_key_metrics[api_key]["uploads_successful"] += 1
    
    async def record_failure(self, api_key: str):
        async with self._lock:
            self.per_key_metrics[api_key]["failures"] += 1
    
    async def record_auth_failure(self):
        async with self._lock:
            self.auth_failures += 1
    
    async def add_latency(self, api_key: str, duration: float):
        async with self._lock:
            self.per_key_metrics[api_key]["latencies"].append(duration)
    
    def get_key_stats(self, api_key: str) -> dict:
        """Get stats for a specific API key"""
        stats = self.per_key_metrics[api_key]
        avg_latency = sum(stats["latencies"]) / len(stats["latencies"]) if stats["latencies"] else 0.0
        
        return {
            "api_calls": stats["api_calls"],
            "questions_asked": stats["questions_asked"],
            "uploads_attempted": stats["uploads_attempted"],
            "uploads_successful": stats["uploads_successful"],
            "failures": stats["failures"],
            "success_rate": round(
                (stats["api_calls"] - stats["failures"]) / stats["api_calls"] * 100, 2
            ) if stats["api_calls"] > 0 else 100.0,
            "upload_success_rate": round(
                stats["uploads_successful"] / stats["uploads_attempted"] * 100, 2
            ) if stats["uploads_attempted"] > 0 else 100.0,
            "average_latency_seconds": round(avg_latency, 4),
        }
    
    def get_all_stats(self) -> dict:
        uptime = (datetime.now() - self.start_time).total_seconds()
        all_stats = {}
        
        for api_key in self.per_key_metrics:
            key_info = API_KEYS.get(api_key, {"name": "unknown"})
            all_stats[key_info["name"]] = self.get_key_stats(api_key)
        
        return {
            "uptime_seconds": round(uptime, 2),
            "auth_failures": self.auth_failures,
            "per_key": all_stats
        }

class JobManager:
    def __init__(self):
        self.jobs: Dict[str, dict] = {}
        self._lock = asyncio.Lock()
        self.active_jobs = 0
    
    async def create_job(self, filename: str, subject: str) -> str:
        async with self._lock:
            if self.active_jobs >= MAX_CONCURRENT_JOBS:
                raise HTTPException(
                    status_code=429,
                    detail=f"Too many concurrent jobs. Max: {MAX_CONCURRENT_JOBS}. Try again later."
                )
            
            job_id = str(uuid.uuid4())
            self.jobs[job_id] = {
                "status": JobStatusEnum.QUEUED,
                "filename": filename,
                "subject": subject,
                "timestamp": datetime.now().isoformat(),
                "details": "Job queued for processing",
                "retry_count": 0,
                "error": None,
                "created_at": datetime.now()
            }
            self.active_jobs += 1
            logger.info(f"Job created: {job_id} - {filename}")
            return job_id
    
    async def update_status(
        self, 
        job_id: str, 
        status: JobStatusEnum, 
        details: str = "",
        error: str = None
    ):
        async with self._lock:
            if job_id in self.jobs:
                old_status = self.jobs[job_id]["status"]
                self.jobs[job_id]["status"] = status
                self.jobs[job_id]["details"] = details
                
                if error:
                    self.jobs[job_id]["error"] = error
                
                if old_status in [JobStatusEnum.QUEUED, JobStatusEnum.PROCESSING, JobStatusEnum.RETRYING]:
                    if status in [JobStatusEnum.COMPLETED, JobStatusEnum.FAILED]:
                        self.active_jobs = max(0, self.active_jobs - 1)
                
                logger.info(f"Job {job_id}: {old_status} -> {status} | {details}")
    
    async def increment_retry(self, job_id: str):
        async with self._lock:
            if job_id in self.jobs:
                self.jobs[job_id]["retry_count"] += 1
    
    async def get_job(self, job_id: str) -> Optional[dict]:
        async with self._lock:
            return self.jobs.get(job_id)
    
    async def cleanup_old_jobs(self):
        async with self._lock:
            cutoff = datetime.now() - timedelta(hours=JOB_RETENTION_HOURS)
            to_delete = [
                job_id for job_id, job in self.jobs.items()
                if job["created_at"] < cutoff and 
                job["status"] in [JobStatusEnum.COMPLETED, JobStatusEnum.FAILED]
            ]
            for job_id in to_delete:
                del self.jobs[job_id]
            
            if to_delete:
                logger.info(f"Cleaned up {len(to_delete)} old jobs")
            return len(to_delete)

metrics = Metrics()
job_manager = JobManager()
rag_service = None
security = HTTPBearer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global rag_service
    
    try:
        rag_service = RAGService()
        logger.info("RAG Service initialized successfully")
        
        cleanup_task = asyncio.create_task(periodic_cleanup())
        logger.info("Background cleanup task started")
        
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        raise
    
    yield
    
    cleanup_task.cancel()
    logger.info("Shutting down gracefully...")

async def periodic_cleanup():
    """Periodically clean up old jobs"""
    while True:
        await asyncio.sleep(3600)
        try:
            deleted = await job_manager.cleanup_old_jobs()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

app = FastAPI(
    title="AskBookie API",
    description="Secure, production-ready API for document Q&A",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
    max_age=600,
)

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    
    if token not in API_KEYS:
        await metrics.record_auth_failure()
        logger.warning(f"Auth failure: Invalid API key attempted")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    return {"key": token, **API_KEYS[token]}

async def process_with_retry(
    file_path: str,
    filename: str,
    job_id: str,
    subject: str,
    api_key: str,
    retry_count: int = 0
):
    try:
        await job_manager.update_status(
            job_id, 
            JobStatusEnum.PROCESSING, 
            f"Processing document (attempt {retry_count + 1}/{MAX_RETRIES + 1})..."
        )
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            process_pdf,
            file_path,
            filename,
            subject,
            None
        )
        
        await job_manager.update_status(
            job_id,
            JobStatusEnum.COMPLETED,
            "Document processed and embedded successfully"
        )
        await metrics.record_upload_success(api_key)
        logger.info(f"Job {job_id} completed successfully")
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Job {job_id} failed: {error_msg}")
        
        if retry_count < MAX_RETRIES:
            await job_manager.increment_retry(job_id)
            retry_delay = RETRY_DELAYS[retry_count]
            
            await job_manager.update_status(
                job_id,
                JobStatusEnum.RETRYING,
                f"Failed, retrying in {retry_delay}s... (attempt {retry_count + 1}/{MAX_RETRIES})",
                error=error_msg
            )
            
            await asyncio.sleep(retry_delay)
            await process_with_retry(file_path, filename, job_id, subject, api_key, retry_count + 1)
        else:
            await job_manager.update_status(
                job_id,
                JobStatusEnum.FAILED,
                f"Failed after {MAX_RETRIES} retries",
                error=error_msg
            )
            await metrics.record_failure(api_key)
    
    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Cleaned up temp file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to delete temp file {file_path}: {e}")

@app.get("/", tags=["Health"])
async def health_check():
    """Health check with system status"""
    return {
        "status": "healthy",
        "service": "AskBookie API",
        "version": "2.0.0",
        "rag_initialized": rag_service is not None,
        "active_jobs": job_manager.active_jobs,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/embed", tags=["Documents"], status_code=202)
async def embed_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    subject: str = Form(...),
    user: dict = Depends(verify_api_key)
):
    api_key = user["key"]
    await metrics.record_api_call(api_key)
    await metrics.record_upload_attempt(api_key)
    start_time = time.time()
    
    logger.info(f"Upload attempt by {user['name']} ({user['role']}) | File: {file.filename}")
    
    if file.content_type != "application/pdf":
        await metrics.record_failure(api_key)
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )
    
    if not subject or not subject.strip():
        await metrics.record_failure(api_key)
        raise HTTPException(
            status_code=400,
            detail="Subject cannot be empty"
        )
    
    try:
        file_content = await file.read()
        if len(file_content) > MAX_FILE_SIZE:
            await metrics.record_failure(api_key)
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE / 1024 / 1024}MB"
            )
        
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name
        
        job_id = await job_manager.create_job(file.filename, subject.strip())
        
        background_tasks.add_task(
            process_with_retry,
            tmp_path,
            file.filename,
            job_id,
            subject.strip(),
            api_key
        )
        
        await metrics.add_latency(api_key, time.time() - start_time)
        
        return {
            "status": "accepted",
            "job_id": job_id,
            "message": f"Document '{file.filename}' queued for processing",
            "check_status_at": f"/jobs/{job_id}",
            "retry_policy": f"Will retry up to {MAX_RETRIES} times on failure"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        await metrics.record_failure(api_key)
        logger.error(f"Upload error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during upload"
        )

@app.post("/ask", tags=["Query"])
async def ask_question(
    query_request: QueryRequest,
    user: dict = Depends(verify_api_key)
):
    api_key = user["key"]
    await metrics.record_api_call(api_key)
    await metrics.record_question(api_key)
    start_time = time.time()
    
    logger.info(f"Query by {user['name']} ({user['role']}) | Subject: {query_request.subject}")
    
    if not rag_service:
        await metrics.record_failure(api_key)
        raise HTTPException(
            status_code=503,
            detail="RAG service not initialized. Please try again later."
        )
    
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            rag_service.ask,
            query_request.query,
            query_request.subject
        )
        
        await metrics.add_latency(api_key, time.time() - start_time)
        
        return response
        
    except HTTPException:
        await metrics.record_failure(api_key)
        raise
    except Exception as e:
        await metrics.record_failure(api_key)
        logger.error(f"Query error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to process query"
        )

@app.get("/jobs/{job_id}", tags=["Jobs"])
async def get_job_status(
    job_id: str,
    user: dict = Depends(verify_api_key)
):
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail="Job not found"
        )
    return job

@app.get("/jobs", tags=["Jobs"])
async def list_jobs(
    limit: int = 50,
    status: Optional[JobStatusEnum] = None,
    user: dict = Depends(verify_api_key)
):
    async with job_manager._lock:
        jobs_list = list(job_manager.jobs.items())
    
    if status:
        jobs_list = [(jid, job) for jid, job in jobs_list if job["status"] == status]
    
    sorted_jobs = sorted(
        jobs_list,
        key=lambda x: x[1]["timestamp"],
        reverse=True
    )
    
    return {
        "total": len(job_manager.jobs),
        "filtered": len(jobs_list),
        "active": job_manager.active_jobs,
        "jobs": dict(sorted_jobs[:limit])
    }

@app.get("/metrics", tags=["Monitoring"])
async def get_metrics(
    user: dict = Depends(verify_api_key)
):
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    cpu_percent = process.cpu_percent(interval=0.1)
    
    if user["role"] == "admin":
        api_metrics = metrics.get_all_stats()
    else:
        api_metrics = {
            "user": user["name"],
            "stats": metrics.get_key_stats(user["key"])
        }
    
    return {
        "api": api_metrics,
        "system": {
            "memory_usage_mb": round(memory_mb, 2),
            "cpu_percent": round(cpu_percent, 2)
        },
        "jobs": {
            "active": job_manager.active_jobs,
            "total": len(job_manager.jobs),
            "max_concurrent": MAX_CONCURRENT_JOBS
        }
    }

@app.delete("/jobs/{job_id}", tags=["Jobs"])
async def delete_job(
    job_id: str,
    user: dict = Depends(verify_api_key)
):
    if user["role"] != "admin":
        raise HTTPException(
            status_code=403,
            detail="Admin access required"
        )
    
    async with job_manager._lock:
        if job_id in job_manager.jobs:
            del job_manager.jobs[job_id]
            logger.info(f"Job {job_id} deleted by admin")
            return {"message": "Job deleted successfully"}
    
    raise HTTPException(status_code=404, detail="Job not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860,
        log_level="info"
    )