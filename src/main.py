import os
import time
import asyncio
import secrets
import hmac
import hashlib
import logging
import sqlite3
import tempfile
import atexit
import re
from datetime import datetime, timedelta, timezone
from threading import Lock
from contextlib import contextmanager, asynccontextmanager
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

load_dotenv()

from rag import RAGService, process_pdf

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
    datefmt='%Y-%m-%dT%H:%M:%S'
)
logger = logging.getLogger("rag-api")

DB_PATH = os.getenv("DB_PATH", "rag_security.db")
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_MIME_TYPES = {"application/pdf"}
UPLOAD_DIR = Path(tempfile.mkdtemp(prefix="rag_uploads_"))
MAX_JSON_BODY = 16 * 1024
MAX_DEFAULT_BODY = 32 * 1024
RATE_LIMITS = {"/ask": 30, "/upload": 2, "default": 50}
FAILED_AUTH_LIMIT = 5
FAILED_AUTH_WINDOW = 300
JOB_TTL_SECONDS = 24 * 3600
KEY_EXPIRY_DAYS = 90
HMAC_TIME_WINDOW = 300
MAX_CONCURRENT_UPLOADS = 3
PDF_PROCESS_TIMEOUT = 300
SUBJECT_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')
STARTUP_TIME = time.time()

_db_lock = Lock()

def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn

@contextmanager
def db_transaction():
    with _db_lock:
        conn = get_db_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

def init_database():
    with db_transaction() as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS rate_limits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key_id TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                timestamp REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_rate_key_endpoint ON rate_limits(key_id, endpoint, timestamp);
            CREATE TABLE IF NOT EXISTS failed_auth (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ip TEXT NOT NULL,
                timestamp REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_failed_ip ON failed_auth(ip, timestamp);
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                key_id TEXT NOT NULL,
                filename TEXT,
                subject TEXT,
                size INTEGER,
                status TEXT NOT NULL,
                error TEXT,
                created_at REAL NOT NULL,
                updated_at REAL
            );
            CREATE INDEX IF NOT EXISTS idx_jobs_key ON jobs(key_id, created_at);
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key_id TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                success INTEGER NOT NULL,
                latency_ms REAL NOT NULL,
                timestamp REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_metrics_key ON metrics(key_id, endpoint, timestamp);
        """)

init_database()

_api_key_configs = [
    ("service-a", "SERVICE_A_SECRET", "user"),
    ("service-b", "SERVICE_B_SECRET", "user"),
    ("service-c", "SERVICE_C_SECRET", "user"),
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

def record_failed_auth(ip: str):
    with db_transaction() as conn:
        conn.execute("INSERT INTO failed_auth (ip, timestamp) VALUES (?, ?)", (ip, time.time()))

def check_auth_lockout(ip: str) -> bool:
    cutoff = time.time() - FAILED_AUTH_WINDOW
    with db_transaction() as conn:
        conn.execute("DELETE FROM failed_auth WHERE timestamp < ?", (cutoff,))
        cursor = conn.execute("SELECT COUNT(*) FROM failed_auth WHERE ip = ? AND timestamp > ?", (ip, cutoff))
        return cursor.fetchone()[0] >= FAILED_AUTH_LIMIT

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
    if check_auth_lockout(ip):
        raise HTTPException(status_code=429, detail="Too many failed attempts")
    key_id = verify_hmac_signature(request)
    if key_id:
        return key_id
    record_failed_auth(ip)
    await asyncio.sleep(0.1)
    raise HTTPException(status_code=401, detail="Unauthorized")

def check_rate_limit(key_id: str, endpoint: str) -> bool:
    limit = RATE_LIMITS.get(endpoint, RATE_LIMITS["default"])
    now = time.time()
    window_start = now - 60
    with db_transaction() as conn:
        conn.execute("DELETE FROM rate_limits WHERE timestamp < ?", (window_start,))
        cursor = conn.execute(
            "SELECT COUNT(*) FROM rate_limits WHERE key_id = ? AND endpoint = ? AND timestamp > ?",
            (key_id, endpoint, window_start)
        )
        if cursor.fetchone()[0] >= limit:
            return False
        conn.execute("INSERT INTO rate_limits (key_id, endpoint, timestamp) VALUES (?, ?, ?)", (key_id, endpoint, now))
    return True

def rate_limited(endpoint: str):
    async def dependency(request: Request):
        key_id = await verify_api_key(request)
        if not check_rate_limit(key_id, endpoint):
            raise HTTPException(status_code=429, detail="Rate limit exceeded", headers={"Retry-After": "60"})
        request.state.key_id = key_id
        return key_id
    return dependency

def update_job_status(job_id: str, status: str, error: str = None):
    with db_transaction() as conn:
        conn.execute("UPDATE jobs SET status = ?, error = ?, updated_at = ? WHERE job_id = ?", 
                     (status, error, time.time(), job_id))

def get_job(job_id: str, key_id: str) -> Optional[dict]:
    with db_transaction() as conn:
        cursor = conn.execute("SELECT * FROM jobs WHERE job_id = ? AND key_id = ?", (job_id, key_id))
        row = cursor.fetchone()
    return dict(row) if row else None

def list_user_jobs(key_id: str) -> list:
    cutoff = time.time() - JOB_TTL_SECONDS
    with db_transaction() as conn:
        conn.execute("DELETE FROM jobs WHERE created_at < ?", (cutoff,))
        cursor = conn.execute(
            "SELECT job_id, filename, subject, size, status, error, created_at FROM jobs WHERE key_id = ? ORDER BY created_at DESC",
            (key_id,)
        )
        return [dict(row) for row in cursor.fetchall()]

def count_active_uploads(key_id: str) -> int:
    with db_transaction() as conn:
        cursor = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE key_id = ? AND status IN ('queued', 'processing')",
            (key_id,)
        )
        return cursor.fetchone()[0]

def sanitize_subject(subject: str) -> str:
    clean = subject.strip().lower()
    if not SUBJECT_PATTERN.match(clean):
        clean = re.sub(r'[^a-zA-Z0-9_-]', '', clean)
    return clean[:50] if clean else "default"

def record_metric(key_id: str, endpoint: str, success: bool, latency_ms: float):
    with db_transaction() as conn:
        conn.execute(
            "INSERT INTO metrics (key_id, endpoint, success, latency_ms, timestamp) VALUES (?, ?, ?, ?, ?)",
            (key_id, endpoint, 1 if success else 0, latency_ms, time.time())
        )

def get_metrics_summary() -> dict:
    import psutil
    with db_transaction() as conn:
        total_calls = conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
        total_questions = conn.execute("SELECT COUNT(*) FROM metrics WHERE endpoint = '/ask'").fetchone()[0]
        active_jobs = conn.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued', 'processing')").fetchone()[0]
        per_user = {}
        cursor = conn.execute("""
            SELECT key_id, COUNT(*) as api_calls,
                   SUM(CASE WHEN endpoint = '/ask' THEN 1 ELSE 0 END) as questions_asked,
                   SUM(CASE WHEN endpoint = '/upload' THEN 1 ELSE 0 END) as uploads_attempted,
                   AVG(CASE WHEN success = 1 THEN 100.0 ELSE 0.0 END) as success_rate,
                   AVG(latency_ms) / 1000.0 as average_latency_seconds,
                   SUM(CASE WHEN endpoint = '/ask' AND success = 0 THEN 1 ELSE 0 END) as ask_fails,
                   SUM(CASE WHEN endpoint = '/upload' AND success = 0 THEN 1 ELSE 0 END) as upload_fails
            FROM metrics GROUP BY key_id
        """)
        for row in cursor.fetchall():
            kid = row[0]
            per_user[kid] = {
                "api_calls": row[1],
                "questions_asked": row[2],
                "uploads_attempted": row[3],
                "success_rate": round(row[4], 1) if row[4] else 100,
                "average_latency_seconds": round(row[5], 2) if row[5] else 0,
                "ask_fails": row[6] or 0,
                "upload_fails": row[7] or 0,
                "role": API_KEYS.get(kid, {}).get("role", "user"),
            }
    process = psutil.Process()
    return {
        "uptime_hours": round((time.time() - STARTUP_TIME) / 3600, 2),
        "total_api_calls": total_calls,
        "total_questions": total_questions,
        "active_jobs": active_jobs,
        "memory_mb": round(process.memory_info().rss / (1024 * 1024), 1),
        "per_user": per_user,
    }

async def process_pdf_job(job_id: str, file_path: Path, original_filename: str, subject: str):
    def status_callback(status: str):
        update_job_status(job_id, status)
        logger.info(f"Job {job_id}: {status}")
    try:
        update_job_status(job_id, "processing")
        logger.info(f"Job {job_id}: Starting RAG processing")
        process_pdf(str(file_path), original_filename, subject, status_callback)
        update_job_status(job_id, "done")
        logger.info(f"Job {job_id}: Complete")
    except Exception as e:
        logger.exception(f"Job {job_id} failed")
        update_job_status(job_id, "failed", str(e)[:200])
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
    logger.info(f"Starting service; upload_dir={UPLOAD_DIR}")
    app.state.rag_service = RAGService()
    logger.info("RAG service initialized")
    yield
    logger.info("Shutting down")

app = FastAPI(
    title="AskBookie RAG API",
    version="1.0.0",
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
        "https://ask-bookie.vercel.app",
        "https://askbookie-pesu.vercel.app",
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
        "X-Frame-Options": "DENY",
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
        return {"answer": result["answer"], "sources": result["sources"], "request_id": request.state.request_id}
    except Exception as e:
        logger.exception(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail="Query failed")
    finally:
        record_metric(key_id, "/ask", success, (time.time() - start_time) * 1000)

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
    if count_active_uploads(key_id) >= MAX_CONCURRENT_UPLOADS:
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
        with db_transaction() as conn:
            conn.execute(
                "INSERT INTO jobs (job_id, key_id, filename, subject, size, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (job_id, key_id, file.filename, subject, size, "queued", time.time())
            )
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
        record_metric(key_id, "/upload", success, (time.time() - start_time) * 1000)

@app.get("/jobs/{job_id}")
async def get_job_status(request: Request, job_id: str, key_id: str = Depends(rate_limited("default"))):
    job = get_job(job_id, key_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.get("/jobs")
async def list_jobs(request: Request, key_id: str = Depends(rate_limited("default"))):
    jobs = list_user_jobs(key_id)
    return {"jobs": jobs, "total": len(jobs)}

@app.get("/health")
async def health():
    try:
        metrics = get_metrics_summary()
        metrics["status"] = "healthy"
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)