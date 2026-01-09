import os
import time
import logging
from datetime import datetime, timezone
from typing import Optional, List

from pymongo import MongoClient, DESCENDING
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

logger = logging.getLogger("rag-api")

MONGODB_URI = os.getenv("MONGODB_URI")
_client: Optional[MongoClient] = None
_db = None


def get_database():
    global _client, _db
    
    if _db is not None:
        return _db
    
    try:
        _client = MongoClient(
            MONGODB_URI, 
            serverSelectionTimeoutMS=5000,
            tls=True,
            tlsAllowInvalidCertificates=True
        )
        _client.admin.command('ping')
        _db = _client.askbookie
        logger.info("MongoDB connected successfully")
        _create_indexes(_db)
        return _db
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        logger.error(f"MongoDB connection failed: {e}")
        raise


def _create_indexes(db):
    try:
        db.rate_limits.create_index("timestamp", expireAfterSeconds=60)
        db.rate_limits.create_index([("key_id", 1), ("endpoint", 1), ("timestamp", 1)])
        db.failed_auth.create_index("timestamp", expireAfterSeconds=300)
        db.failed_auth.create_index([("ip", 1), ("timestamp", 1)])
        db.jobs.create_index("created_at", expireAfterSeconds=86400)
        db.jobs.create_index([("key_id", 1), ("created_at", -1)])
        db.jobs.create_index("job_id", unique=True)
        db.metrics.create_index([("key_id", 1), ("endpoint", 1), ("timestamp", -1)])
        db.query_history.create_index([("timestamp", -1)])
        logger.info("MongoDB indexes created")
    except Exception as e:
        logger.warning(f"Index creation warning: {e}")


def check_rate_limit(key_id: str, endpoint: str, limit: int) -> bool:
    db = get_database()
    now = datetime.now(timezone.utc)
    window_start = datetime.fromtimestamp(time.time() - 60, tz=timezone.utc)
    
    count = db.rate_limits.count_documents({
        "key_id": key_id,
        "endpoint": endpoint,
        "timestamp": {"$gte": window_start}
    })
    
    if count >= limit:
        return False
    
    db.rate_limits.insert_one({
        "key_id": key_id,
        "endpoint": endpoint,
        "timestamp": now
    })
    return True


def record_failed_auth(ip: str):
    db = get_database()
    db.failed_auth.insert_one({
        "ip": ip,
        "timestamp": datetime.now(timezone.utc)
    })


def check_auth_lockout(ip: str, limit: int = 5) -> bool:
    db = get_database()
    window_start = datetime.fromtimestamp(time.time() - 300, tz=timezone.utc)
    count = db.failed_auth.count_documents({
        "ip": ip,
        "timestamp": {"$gte": window_start}
    })
    return count >= limit


def create_job(job_id: str, key_id: str, filename: str, subject: str, size: int) -> dict:
    db = get_database()
    job = {
        "job_id": job_id,
        "key_id": key_id,
        "filename": filename,
        "subject": subject,
        "size": size,
        "status": "queued",
        "error": None,
        "created_at": datetime.now(timezone.utc),
        "updated_at": None
    }
    db.jobs.insert_one(job)
    return job


def update_job_status(job_id: str, status: str, error: str = None):
    db = get_database()
    db.jobs.update_one(
        {"job_id": job_id},
        {"$set": {
            "status": status,
            "error": error,
            "updated_at": datetime.now(timezone.utc)
        }}
    )


def get_job(job_id: str, key_id: str) -> Optional[dict]:
    db = get_database()
    job = db.jobs.find_one({"job_id": job_id, "key_id": key_id})
    if job:
        job["_id"] = str(job["_id"])
        job["created_at"] = job["created_at"].timestamp() if job.get("created_at") else None
        job["updated_at"] = job["updated_at"].timestamp() if job.get("updated_at") else None
    return job


def list_user_jobs(key_id: str) -> List[dict]:
    db = get_database()
    jobs = list(db.jobs.find(
        {"key_id": key_id},
        {"_id": 0, "job_id": 1, "filename": 1, "subject": 1, "size": 1, "status": 1, "error": 1, "created_at": 1}
    ).sort("created_at", DESCENDING))
    
    for job in jobs:
        if job.get("created_at"):
            job["created_at"] = job["created_at"].timestamp()
    return jobs


def count_active_uploads(key_id: str) -> int:
    db = get_database()
    return db.jobs.count_documents({
        "key_id": key_id,
        "status": {"$in": ["queued", "processing"]}
    })


def record_metric(key_id: str, endpoint: str, success: bool, latency_ms: float):
    db = get_database()
    db.metrics.insert_one({
        "key_id": key_id,
        "endpoint": endpoint,
        "success": success,
        "latency_ms": latency_ms,
        "timestamp": datetime.now(timezone.utc)
    })


def get_metrics_summary() -> dict:
    import psutil
    
    db = get_database()
    
    total_calls = db.metrics.count_documents({})
    total_questions = db.metrics.count_documents({"endpoint": "/ask"})
    active_jobs = db.jobs.count_documents({"status": {"$in": ["queued", "processing"]}})
    
    pipeline = [
        {"$group": {
            "_id": "$key_id",
            "api_calls": {"$sum": 1},
            "questions_asked": {"$sum": {"$cond": [{"$eq": ["$endpoint", "/ask"]}, 1, 0]}},
            "uploads_attempted": {"$sum": {"$cond": [{"$eq": ["$endpoint", "/upload"]}, 1, 0]}},
            "success_count": {"$sum": {"$cond": ["$success", 1, 0]}},
            "total_latency": {"$sum": "$latency_ms"},
            "ask_fails": {"$sum": {"$cond": [{"$and": [{"$eq": ["$endpoint", "/ask"]}, {"$not": "$success"}]}, 1, 0]}},
            "upload_fails": {"$sum": {"$cond": [{"$and": [{"$eq": ["$endpoint", "/upload"]}, {"$not": "$success"}]}, 1, 0]}}
        }}
    ]
    
    per_user = {}
    for doc in db.metrics.aggregate(pipeline):
        kid = doc["_id"]
        total = doc["api_calls"]
        per_user[kid] = {
            "api_calls": total,
            "questions_asked": doc["questions_asked"],
            "uploads_attempted": doc["uploads_attempted"],
            "success_rate": round((doc["success_count"] / total * 100) if total > 0 else 100, 1),
            "average_latency_seconds": round((doc["total_latency"] / total / 1000) if total > 0 else 0, 2),
            "ask_fails": doc["ask_fails"],
            "upload_fails": doc["upload_fails"],
        }
    
    process = psutil.Process()
    return {
        "total_api_calls": total_calls,
        "total_questions": total_questions,
        "active_jobs": active_jobs,
        "memory_mb": round(process.memory_info().rss / (1024 * 1024), 1),
        "per_user": per_user,
    }


def store_query_history(key_id: str, subject: str, query: str, answer: str,
                        sources: list, request_id: str, latency_ms: float):
    db = get_database()
    db.query_history.insert_one({
        "key_id": key_id,
        "subject": subject,
        "query": query,
        "answer": answer,
        "sources": sources,
        "request_id": request_id,
        "latency_ms": latency_ms,
        "timestamp": datetime.now(timezone.utc)
    })


def get_query_history(limit: int = 100, offset: int = 0) -> tuple[List[dict], int]:
    db = get_database()
    
    total = db.query_history.count_documents({})
    
    history = list(db.query_history.find(
        {},
        {"_id": 0}
    ).sort("timestamp", DESCENDING).skip(offset).limit(limit))
    
    for i, item in enumerate(history):
        item["id"] = offset + i + 1
        if item.get("timestamp"):
            item["timestamp"] = item["timestamp"].timestamp()
    
    return history, total
