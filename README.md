---
title: AskBookie API
emoji: 🔥
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
---

# AskBookie API

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/YOUR_USERNAME/askbookie-api)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-DC382D?logo=qdrant&logoColor=white)](https://qdrant.tech)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C?logo=langchain&logoColor=white)](https://langchain.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-ready RAG (Retrieval Augmented Generation) API for document Q&A using university slide data.
### **Live API**: [https://huggingface.co/spaces/pmmdot/askbookie](https://huggingface.co/spaces/pmmdot/askbookie)

## Quick Start

### Authentication

This API uses HMAC-SHA256 authentication. Every request requires three headers:

| Header | Description |
|--------|-------------|
| `X-API-Key-Id` | Your API key identifier |
| `X-API-Timestamp` | Current Unix timestamp |
| `X-API-Signature` | HMAC-SHA256 signature |

### Python Example

```python
import hmac
import hashlib
import time
import requests

# Your credentials
API_KEY_ID = "your-key-id"
API_SECRET = "your-secret-key"
BASE_URL = "https://huggingface.co/spaces/pmmdot/askbookie"

def get_auth_headers(method: str, path: str) -> dict:
    """Generate HMAC authentication headers."""
    timestamp = str(int(time.time()))
    message = f"{timestamp}\n{method.upper()}\n{path}"
    signature = hmac.new(
        API_SECRET.encode(),
        message.encode(),
        hashlib.sha256
    ).hexdigest()
    
    return {
        "X-API-Key-Id": API_KEY_ID,
        "X-API-Timestamp": timestamp,
        "X-API-Signature": signature,
    }

# Ask a question
def ask_question(query: str, subject: str):
    headers = get_auth_headers("POST", "/ask")
    headers["Content-Type"] = "application/json"
    
    response = requests.post(
        f"{BASE_URL}/ask",
        headers=headers,
        json={"query": query, "subject": subject}
    )
    return response.json()

# Example usage
result = ask_question("Tell me more about the classification of ecosystem", "evs")
print(result["answer"])
print("Sources:", result["sources"])
```

### Upload a PDF

```python
def upload_pdf(file_path: str, subject: str):
    headers = get_auth_headers("POST", "/upload")
    
    with open(file_path, "rb") as f:
        response = requests.post(
            f"{BASE_URL}/upload",
            headers=headers,
            files={"file": f},
            data={"subject": subject}
        )
    return response.json()

# Upload and get job ID
job = upload_pdf("MES.pdf", "mes")
print(f"Job ID: {job['job_id']}, Status: {job['status']}")
```

### Check Job Status

```python
def check_job(job_id: str):
    headers = get_auth_headers("GET", f"/jobs/{job_id}")
    response = requests.get(f"{BASE_URL}/jobs/{job_id}", headers=headers)
    return response.json()

# Poll until complete
status = check_job(job["job_id"])
print(f"Status: {status['status']}")
```

### cURL Example

```bash
# Generate signature (use the Python helper or implement in your language)
TIMESTAMP=$(date +%s)
SIGNATURE=$(echo -n "${TIMESTAMP}\nPOST\n/ask" | openssl dgst -sha256 -hmac "your-secret" | cut -d' ' -f2)

curl -X POST "https://huggingface.co/spaces/pmmdot/askbookie/ask" \
  -H "Content-Type: application/json" \
  -H "X-API-Key-Id: your-key-id" \
  -H "X-API-Timestamp: $TIMESTAMP" \
  -H "X-API-Signature: $SIGNATURE" \
  -d '{"query": "What is the use of re module in python?", "subject": "pcps"}'
```

## Features

- **PDF Upload & Processing** - Upload PDFs and embed them into a Qdrant vector database
- **Intelligent Q&A** - Ask questions about your documents using Gemini 3 Flash or GPT-4o
- **High Throughput LLM Routing** - Multiple free/paid providers with effectively unlimited calls per minute
- **Per-Key Metrics** - Track usage metrics for each API key
- **Background Job Processing** - Async PDF processing with automatic retry logic
- **Admin Controls** - Separate admin and user permissions

## Security Features

- **HMAC-based Authentication** - Time-sensitive HMAC-SHA256 signatures for API authentication
- **Rate Limiting** - Per-endpoint, per-key rate limits (30/min for /ask, 2/min for /upload)
- **Auth Lockout** - Automatic IP-based lockout after 5 failed authentication attempts
- **Constant-Time Comparison** - Timing-safe HMAC verification to prevent timing attacks
- **Payload Size Limits** - 10MB max file size, 16KB max JSON payload for /ask endpoint
- **File Validation** - PDF magic number validation, MIME type checking, and file signature verification
- **Request Tracking** - Unique request IDs for all API calls
- **Security Headers** - X-Content-Type-Options, X-Frame-Options, Referrer-Policy
- **API Key Expiry** - Configurable key expiration (90 days default)
- **Input Sanitization** - Subject name validation and length restrictions on all inputs

## API Endpoints

### Public Endpoints
- `GET /` - Dashboard UI
- `GET /health` - Health check with detailed metrics (uptime, memory usage, per-user stats)

### User Endpoints (Require Authentication)
- `POST /upload` - Upload and process PDF documents (10MB max, async processing)
- `POST /ask` - Ask questions about documents (1-20 context documents, configurable)
- `GET /jobs/{job_id}` - Check upload job status (queued/processing/done/failed)
- `GET /jobs` - List all jobs for the authenticated user with 24h auto-cleanup

### Admin Endpoints (Admin Role Required)
- `GET /history` - View query history with pagination (includes answers, sources, latency)
- `GET /admin/keys` - List all API keys with status and expiration dates
- `POST /admin/keys/{key_id}/enable` - Enable an API key
- `POST /admin/keys/{key_id}/disable` - Disable an API key (cannot disable admin key)

## Tech Stack

- **FastAPI** - Modern Python web framework
- **Qdrant** - Vector database for embeddings
- **LLM Providers** - Gemini 3 Flash and GPT-4o via multiple free/paid providers with effectively unlimited calls per minute
- **SQLite** - lightweight relational store for metadata and jobs
- **HuggingFace Embeddings** - gte-modernbert-base for document embeddings
- **LangChain** - RAG orchestration

## Documentation

Interactive API documentation available at `/docs` when running.
