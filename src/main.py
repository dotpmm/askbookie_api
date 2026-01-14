import os
import time
import asyncio
import secrets
import hmac
import hashlib
import logging
import tempfile
import atexit
import re
import json
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

load_dotenv()

from rag import RAGService, process_pdf, model_manager, MODEL_OPTIONS, QuotaExhaustedError
import database as db

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:     %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("rag-api")

MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_MIME_TYPES = {"application/pdf"}
UPLOAD_DIR = Path(tempfile.mkdtemp(prefix="rag_uploads_"))
MAX_JSON_BODY = 16 * 1024
MAX_DEFAULT_BODY = 32 * 1024
RATE_LIMITS = {"/ask": 30, "/upload": 2, "default": 50}
FAILED_AUTH_LIMIT = 5
KEY_EXPIRY_DAYS = 90
HMAC_TIME_WINDOW = 300
MAX_CONCURRENT_UPLOADS = 3
SUBJECT_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')
STARTUP_TIME = time.time()

_api_key_configs = [
    ("askbookie-dev", "API_KEY_1", "user"),
    ("askbookie-prod", "API_KEY_2", "user"),
]

API_KEYS = {}
for key_id, env_var, role in _api_key_configs:
    secret = os.getenv(env_var)
    if secret:
        API_KEYS[key_id] = {
            "secret": secret,
            "expires_at": datetime.now(timezone.utc) + timedelta(days=KEY_EXPIRY_DAYS),
            "active": True,
            "role": role,
        }

ADMIN_SECRET = os.getenv("ADMIN_API_KEY")
if ADMIN_SECRET:
    API_KEYS["admin"] = {"secret": ADMIN_SECRET, "expires_at": None, "active": True, "role": "admin"}

if not API_KEYS:
    logger.warning("No API keys configured!")

def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"

def verify_hmac_signature(request: Request) -> Optional[str]:
    key_id = request.headers.get("X-API-Key-Id")
    sig = request.headers.get("X-API-Signature")
    ts = request.headers.get("X-API-Timestamp")
    if not all([key_id, sig, ts]):
        return None
    meta = API_KEYS.get(key_id)
    dummy_secret = "dummy_secret_for_timing_safety"
    secret_to_use = meta["secret"] if meta else dummy_secret
    is_valid_key = meta is not None and meta.get("active", False)
    if meta and meta.get("expires_at"):
        is_valid_key = is_valid_key and meta["expires_at"] > datetime.now(timezone.utc)
    try:
        ts_int = int(ts)
    except (ValueError, TypeError):
        ts_int = 0
        is_valid_key = False
    time_valid = abs(time.time() - ts_int) <= HMAC_TIME_WINDOW
    is_valid_key = is_valid_key and time_valid
    message = f"{ts_int}\n{request.method.upper()}\n{request.url.path}"
    computed = hmac.new(secret_to_use.encode(), message.encode(), hashlib.sha256).hexdigest()
    sig_valid = secrets.compare_digest(computed, sig) if sig else False
    if is_valid_key and sig_valid:
        return key_id
    return None

async def verify_api_key(request: Request) -> str:
    ip = get_client_ip(request)
    if db.check_auth_lockout(ip, FAILED_AUTH_LIMIT):
        raise HTTPException(status_code=429, detail="Too many failed attempts")
    key_id = verify_hmac_signature(request)
    if key_id:
        return key_id
    db.record_failed_auth(ip)
    await asyncio.sleep(0.1)
    raise HTTPException(status_code=401, detail="Unauthorized")

def rate_limited(endpoint: str):
    async def dependency(request: Request):
        key_id = await verify_api_key(request)
        limit = RATE_LIMITS.get(endpoint, RATE_LIMITS["default"])
        if not db.check_rate_limit(key_id, endpoint, limit):
            raise HTTPException(status_code=429, detail="Rate limit exceeded", headers={"Retry-After": "60"})
        request.state.key_id = key_id
        return key_id
    return dependency

def sanitize_subject(subject: str) -> str:
    clean = subject.strip().lower()
    if not SUBJECT_PATTERN.match(clean):
        clean = re.sub(r'[^a-zA-Z0-9_-]', '', clean)
    return clean[:50] if clean else "default"

def get_metrics_summary() -> dict:
    metrics = db.get_metrics_summary()
    metrics["uptime_hours"] = round((time.time() - STARTUP_TIME) / 3600, 2)
    for kid in metrics.get("per_user", {}):
        metrics["per_user"][kid]["role"] = API_KEYS.get(kid, {}).get("role", "user")
    return metrics

async def process_pdf_job(job_id: str, file_path: Path, original_filename: str, subject: str):
    def status_callback(status: str):
        db.update_job_status(job_id, status)
        logger.info(f"Job {job_id}: {status}")
    try:
        db.update_job_status(job_id, "processing")
        logger.info(f"Job {job_id}: Starting RAG processing")
        process_pdf(str(file_path), original_filename, subject, status_callback)
        db.update_job_status(job_id, "done")
        logger.info(f"Job {job_id}: Complete")
    except Exception as e:
        logger.exception(f"Job {job_id} failed")
        db.update_job_status(job_id, "failed", str(e)[:200])
    finally:
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception:
            pass

def cleanup_uploads():
    import shutil
    try:
        shutil.rmtree(UPLOAD_DIR)
        logger.info(f"Cleaned up: {UPLOAD_DIR}")
    except Exception:
        pass

atexit.register(cleanup_uploads)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting service")
    app.state.rag_service = RAGService()
    logger.info("RAG service initialized")
    yield
    logger.info("Shutting down")

app = FastAPI(
    title="AskBookie RAG API",
    version="1.2.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:7860",
        "http://127.0.0.1:7860",
        "https://askbookie.vercel.app",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key-Id", "X-API-Signature", "X-API-Timestamp", "Content-Type"],
    allow_credentials=False,
    max_age=3600,
)

@app.middleware("http")
async def payload_size_guard(request: Request, call_next):
    path = request.url.path
    if path == "/upload":
        return await call_next(request)
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            size = int(content_length)
            limit = MAX_JSON_BODY if path == "/ask" else MAX_DEFAULT_BODY
            if size > limit:
                return JSONResponse(status_code=413, content={"detail": "Payload too large"})
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "Invalid request"})
    return await call_next(request)

@app.middleware("http")
async def security_middleware(request: Request, call_next):
    request_id = secrets.token_hex(8)
    request.state.request_id = request_id
    start_time = time.time()
    response = await call_next(request)
    response.headers.update({
        "X-Request-ID": request_id,
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "ALLOW-FROM https://huggingface.co",
        "Referrer-Policy": "strict-origin-when-cross-origin",
    })
    duration_ms = round((time.time() - start_time) * 1000, 2)
    key_id = getattr(request.state, "key_id", None)
    logger.info(f"{request.method} {request.url.path} {response.status_code} {duration_ms}ms key={key_id} rid={request_id}")
    return response

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail}, headers=getattr(exc, "headers", None))

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled error: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal error"})

class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    subject: str = Field(..., min_length=1, max_length=100)
    context_limit: int = Field(default=5, ge=1, le=20)

    @field_validator("query", "subject")
    @classmethod
    def sanitize(cls, v: str) -> str:
        return v.strip()

@app.post("/ask")
async def ask(request: Request, body: AskRequest, key_id: str = Depends(rate_limited("/ask"))):
    start_time = time.time()
    success = False
    subject = sanitize_subject(body.subject)
    if not subject:
        raise HTTPException(status_code=400, detail="Invalid subject")
    try:
        rag_service: RAGService = request.app.state.rag_service
        result = rag_service.ask(body.query, subject)
        success = True
        latency_ms = (time.time() - start_time) * 1000
        db.store_query_history(
            key_id=key_id,
            subject=subject,
            query=body.query,
            answer=result["answer"],
            sources=result["sources"],
            request_id=request.state.request_id,
            latency_ms=latency_ms
        )
        return {"answer": result["answer"], "sources": result["sources"], "request_id": request.state.request_id}
    except QuotaExhaustedError as e:
        logger.warning(f"LLM quota exhausted: {e}")
        raise HTTPException(status_code=429, detail="LLM quota exhausted. Try again later or switch model.", headers={"Retry-After": "3600"})
    except Exception as e:
        logger.exception(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail="Query failed")
    finally:
        db.record_metric(key_id, "/ask", success, (time.time() - start_time) * 1000)

@app.post("/upload")
async def upload(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    subject: str = Form(..., min_length=1, max_length=100),
    key_id: str = Depends(rate_limited("/upload"))
):
    start_time = time.time()
    success = False
    subject = sanitize_subject(subject)
    if not subject:
        raise HTTPException(status_code=400, detail="Invalid subject")
    if db.count_active_uploads(key_id) >= MAX_CONCURRENT_UPLOADS:
        raise HTTPException(status_code=429, detail="Too many concurrent uploads")
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail="Invalid file type")
    job_id = secrets.token_hex(12)
    temp_path = UPLOAD_DIR / f"{job_id}.pdf"
    size = 0
    try:
        with open(temp_path, "wb") as f:
            first_chunk = True
            while chunk := await file.read(1024 * 1024):
                if first_chunk:
                    if not chunk.startswith(b"%PDF"):
                        raise HTTPException(status_code=400, detail="Invalid PDF")
                    first_chunk = False
                size += len(chunk)
                if size > MAX_FILE_SIZE:
                    raise HTTPException(status_code=413, detail="File too large")
                f.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        db.create_job(job_id, key_id, file.filename, subject, size)
        background_tasks.add_task(process_pdf_job, job_id, temp_path, file.filename, subject)
        logger.info(f"Upload queued: job={job_id}")
        success = True
        return {"job_id": job_id, "status": "queued", "filename": file.filename, "subject": subject, "size": size}
    except HTTPException:
        if temp_path.exists():
            temp_path.unlink()
        raise
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail="Upload failed")
    finally:
        db.record_metric(key_id, "/upload", success, (time.time() - start_time) * 1000)

@app.get("/jobs/{job_id}")
async def get_job_status(request: Request, job_id: str, key_id: str = Depends(rate_limited("default"))):
    job = db.get_job(job_id, key_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.get("/jobs")
async def list_jobs(request: Request, key_id: str = Depends(rate_limited("default"))):
    jobs = db.list_user_jobs(key_id)
    return {"jobs": jobs, "total": len(jobs)}

@app.get("/health")
async def health():
    try:
        metrics = get_metrics_summary()
        metrics["status"] = "healthy"
        metrics["current_model"] = model_manager.current_model_info
        return metrics
    except Exception:
        logger.exception("Metrics error")
        return {"status": "degraded", "uptime_hours": round((time.time() - STARTUP_TIME) / 3600, 2)}

@app.get("/")
async def dashboard():
    dashboard_path = Path(__file__).parent.parent / "assets" / "index.html"
    if dashboard_path.exists():
        return FileResponse(dashboard_path, media_type="text/html")
    return {"service": "AskBookie RAG API", "version": "1.0.0"}

@app.get("/admin/keys")
async def list_keys(request: Request, key_id: str = Depends(rate_limited("default"))):
    if API_KEYS.get(key_id, {}).get("role") != "admin":
        raise HTTPException(status_code=403, detail="Forbidden")
    keys = []
    for kid, meta in API_KEYS.items():
        keys.append({
            "key_id": kid,
            "role": meta.get("role"),
            "active": meta.get("active"),
            "expires_at": meta.get("expires_at").isoformat() if meta.get("expires_at") else None,
        })
    return {"keys": keys}

@app.post("/admin/keys/{target_key_id}/disable")
async def disable_key(request: Request, target_key_id: str, key_id: str = Depends(rate_limited("default"))):
    if API_KEYS.get(key_id, {}).get("role") != "admin":
        raise HTTPException(status_code=403, detail="Forbidden")
    if target_key_id not in API_KEYS:
        raise HTTPException(status_code=404, detail="Not found")
    if target_key_id == "admin":
        raise HTTPException(status_code=400, detail="Cannot disable admin")
    API_KEYS[target_key_id]["active"] = False
    logger.info(f"Key {target_key_id} disabled")
    return {"status": "disabled", "key_id": target_key_id}

@app.post("/admin/keys/{target_key_id}/enable")
async def enable_key(request: Request, target_key_id: str, key_id: str = Depends(rate_limited("default"))):
    if API_KEYS.get(key_id, {}).get("role") != "admin":
        raise HTTPException(status_code=403, detail="Forbidden")
    if target_key_id not in API_KEYS:
        raise HTTPException(status_code=404, detail="Not found")
    API_KEYS[target_key_id]["active"] = True
    logger.info(f"Key {target_key_id} enabled")
    return {"status": "enabled", "key_id": target_key_id}

@app.get("/history")
async def get_query_history(
    request: Request,
    limit: int = 100,
    offset: int = 0,
    key_id: str = Depends(rate_limited("default"))
):
    if API_KEYS.get(key_id, {}).get("role") != "admin":
        raise HTTPException(status_code=403, detail="Forbidden")
    
    history, total = db.get_query_history(limit, offset)
    return {"history": history, "total": total, "limit": limit, "offset": offset}

class ModelSwitchRequest(BaseModel):
    model_id: int = Field(..., ge=1, le=5)

@app.get("/admin/models/current")
async def get_current_model(request: Request, key_id: str = Depends(rate_limited("default"))):
    if API_KEYS.get(key_id, {}).get("role") != "admin":
        raise HTTPException(status_code=403, detail="Forbidden")
    return model_manager.current_model_info


@app.post("/admin/models/switch")
async def switch_model(
    request: Request, 
    body: ModelSwitchRequest, 
    key_id: str = Depends(rate_limited("default"))
):
    if API_KEYS.get(key_id, {}).get("role") != "admin":
        raise HTTPException(status_code=403, detail="Forbidden")
    
    try:
        result = model_manager.switch_model(body.model_id)
        logger.info(f"Admin switched model to {body.model_id}")
        return {
            "status": "success",
            "message": f"Switched to model {body.model_id}",
            "model": result
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)