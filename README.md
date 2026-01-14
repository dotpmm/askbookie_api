# AskBookie API

Production-grade retrieval-augmented generation service for document-based question answering. The system operates on pre-vectorised document clusters stored in Qdrant, with semantic retrieval feeding into instruction-tuned language model inference.

**Base URL**: `https://pmmdot-askbookie.hf.space`  
**Interactive Documentation**: `/docs` (Swagger UI) | `/redoc` (ReDoc)

## Table of Contents

1. [Authentication](#authentication)
2. [Rate Limits](#rate-limits)
3. [Core Endpoints](#core-endpoints)
   - [POST /ask](#post-ask)
   - [POST /upload](#post-upload)
   - [GET /jobs/{job_id}](#get-jobsjob_id)
   - [GET /jobs](#get-jobs)
4. [System Endpoints](#system-endpoints)
   - [GET /health](#get-health)
   - [GET /](#get-)
5. [Admin Endpoints](#admin-endpoints)
   - [GET /history](#get-history)
   - [GET /admin/keys](#get-adminkeys)
   - [POST /admin/keys/{key_id}/enable](#post-adminkeyskeyidenable)
   - [POST /admin/keys/{key_id}/disable](#post-adminkeyskeyiddisable)
   - [GET /admin/models/current](#get-adminmodelscurrent)
   - [POST /admin/models/switch](#post-adminmodelsswitch)
6. [Error Handling](#error-handling)


## Technical Stack

| Component | Technology |
|-----------|------------|
| Web Framework | FastAPI |
| Vector Database | Qdrant Cloud |
| Embedding Model | HuggingFace `gte-modernbert-base` |
| Language Models | Gemini 3 Flash/Pro, GPT-4o-mini, Claude-3-Haiku |
| RAG Orchestration | LangChain |
| Metadata Storage | MongoDB Atlas |
| PDF Processing | PyPDFLoader |

## Available Models

| Model ID | Name | Description |
|----------|------|-------------|
| 1 | Gemini-3-flash | Gemini Primary API Key |
| 2 | Gemini-3-flash (Back-up) | Gemini Secondary API Key |
| 3 | Gemini-3-Pro | Gemini Primary API Key |
| 4 | GPT-4o-mini | DuckDuckGo (Free) |
| 5 | Claude-3-Haiku | DuckDuckGo (Free) |

## Authentication

All endpoints except `/health` and `/` require HMAC-SHA256 request signing. Authentication operates on a rotating key infrastructure with 90-day expiration cycles.

### Required Headers

| Header | Description |
|--------|-------------|
| `X-API-Key-Id` | Unique identifier for your API key |
| `X-API-Timestamp` | Current Unix timestamp (seconds) |
| `X-API-Signature` | HMAC-SHA256 signature of the request |

### Signature Construction

The signature message follows the format:
```
{timestamp}\n{HTTP_METHOD}\n{path}
```

**Python Implementation**:
```python
import hmac
import hashlib
import time

def generate_auth_headers(method: str, path: str, key_id: str, secret: str) -> dict:
    timestamp = str(int(time.time()))
    message = f"{timestamp}\n{method.upper()}\n{path}"
    signature = hmac.new(
        secret.encode(),
        message.encode(),
        hashlib.sha256
    ).hexdigest()
    
    return {
        "X-API-Key-Id": key_id,
        "X-API-Timestamp": timestamp,
        "X-API-Signature": signature
    }
```

**JavaScript Implementation**:
```javascript
const crypto = require('crypto');

function generateAuthHeaders(method, path, keyId, secret) {
    const timestamp = Math.floor(Date.now() / 1000).toString();
    const message = `${timestamp}\n${method.toUpperCase()}\n${path}`;
    const signature = crypto
        .createHmac('sha256', secret)
        .update(message)
        .digest('hex');
    
    return {
        'X-API-Key-Id': keyId,
        'X-API-Timestamp': timestamp,
        'X-API-Signature': signature
    };
}
```

### Security Constraints

- Timestamp tolerance: 300 seconds (5 minutes)
- Failed authentication lockout: 5 attempts per IP (5-minute window)
- Constant-time signature comparison to prevent timing attacks



## Rate Limits

Rate limiting operates on a sliding window of 60 seconds per API key.

| Endpoint | Limit | Window |
|----------|-------|--------|
| `/ask` | 30 requests | 60 seconds |
| `/upload` | 2 requests | 60 seconds |
| All other endpoints | 50 requests | 60 seconds |
| Failed auth attempts | 5 per IP | 5 minutes (lockout) |

When rate limited, responses include the `Retry-After` header:
```http
HTTP/1.1 429 Too Many Requests
Retry-After: 60
Content-Type: application/json

{"detail": "Rate limit exceeded"}
```
## Core Endpoints

### POST /ask

Query the pre-vectorised document corpus. The system performs semantic retrieval against the specified subject partition, then synthesises a response using the active language model.

**Request**:
```http
POST /ask HTTP/1.1
Content-Type: application/json
X-API-Key-Id: your-key-id
X-API-Timestamp: 1705234567
X-API-Signature: a1b2c3d4...

{
    "query": "What are the different types of ecosystems?",
    "subject": "evs",
    "context_limit": 5
}
```

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `query` | string | Yes | 1-1000 chars | Natural language question |
| `subject` | string | Yes | 1-100 chars, alphanumeric with `_-` | Document collection identifier |
| `context_limit` | integer | No | 1-20, default 5 | Number of context chunks for retrieval |

**Response** (200 OK):
```json
{
    "answer": "Ecosystems are classified into two primary categories: terrestrial and aquatic. Terrestrial ecosystems include forests, grasslands, deserts, and tundra. Aquatic ecosystems are subdivided into freshwater (lakes, rivers, wetlands) and marine (oceans, coral reefs, estuaries).",
    "sources": [
        {
            "page": 12,
            "content": "Ecosystems can be broadly categorized into terrestrial and aquatic types...",
            "filename": "evs_chapter3.pdf"
        },
        {
            "page": 15,
            "content": "Marine ecosystems cover approximately 71% of Earth's surface...",
            "filename": "evs_chapter3.pdf"
        }
    ],
    "request_id": "a1b2c3d4e5f6g7h8"
}
```
---
### POST /upload

Ingest a PDF document into the vector index. Documents are validated, chunked by page boundaries, embedded using the HuggingFace `gte-modernbert-base` model, and stored in the specified subject partition. Processing occurs asynchronously.

**Request**:
```http
POST /upload HTTP/1.1
Content-Type: multipart/form-data
X-API-Key-Id: your-key-id
X-API-Timestamp: 1705234567
X-API-Signature: a1b2c3d4...

file: [binary PDF data]
subject: physics
```

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `file` | binary | Yes | PDF, max 10MB, must start with `%PDF` magic bytes | Document to ingest |
| `subject` | string | Yes | 1-100 chars, alphanumeric with `_-` | Target collection identifier |

**Response** (200 OK):
```json
{
    "job_id": "a1b2c3d4e5f6g7h8i9j0k1l2",
    "status": "queued",
    "filename": "thermodynamics_notes.pdf",
    "subject": "physics",
    "size": 2457600
}
```

**Validation Pipeline**:
1. MIME type verification (`application/pdf`)
2. Magic byte validation (must start with `%PDF`)
3. Size constraint (max 10MB)
4. Concurrent upload limit (max 3 per key)
---

### GET /jobs/{job_id}

Retrieve the status of a PDF processing job. Jobs transition through the following states: `queued` → `processing` → `done` | `failed`.

**Request**:
```http
GET /jobs/a1b2c3d4e5f6g7h8i9j0k1l2 HTTP/1.1
X-API-Key-Id: your-key-id
X-API-Timestamp: 1705234567
X-API-Signature: a1b2c3d4...
```

**Response** (200 OK):
```json
{
    "job_id": "a1b2c3d4e5f6g7h8i9j0k1l2",
    "key_id": "your-key-id",
    "filename": "thermodynamics_notes.pdf",
    "subject": "physics",
    "size": 2457600,
    "status": "done",
    "error": null,
    "created_at": 1705234567.0,
    "updated_at": 1705234890.0
}
```

| Status | Description |
|--------|-------------|
| `queued` | Job accepted, awaiting processing |
| `processing` | Document being chunked and embedded |
| `done` | Successfully ingested into vector store |
| `failed` | Processing error (check `error` field) |
---

### GET /jobs

List all jobs associated with the authenticated API key. Results are ordered by creation time (descending).

**Request**:
```http
GET /jobs HTTP/1.1
X-API-Key-Id: your-key-id
X-API-Timestamp: 1705234567
X-API-Signature: a1b2c3d4...
```

**Response** (200 OK):
```json
{
    "jobs": [
        {
            "job_id": "a1b2c3d4e5f6g7h8i9j0k1l2",
            "filename": "thermodynamics_notes.pdf",
            "subject": "physics",
            "size": 2457600,
            "status": "done",
            "error": null,
            "created_at": 1705234567.0
        },
        {
            "job_id": "m1n2o3p4q5r6s7t8u9v0w1x2",
            "filename": "organic_chemistry.pdf",
            "subject": "chemistry",
            "size": 1843200,
            "status": "processing",
            "error": null,
            "created_at": 1705234500.0
        }
    ],
    "total": 2
}
```

## System Endpoints

### GET /health

Service health check endpoint. Returns operational metrics and current model configuration. No authentication required.

**Response** (200 OK):
```json
{
    "status": "healthy",
    "uptime_hours": 48.5,
    "total_api_calls": 15420,
    "total_questions": 12350,
    "active_jobs": 2,
    "memory_mb": 1024.5,
    "current_model": {
        "id": 1,
        "name": "Gemini-3-flash",
        "description": "Gemini Primary API Key"
    },
    "per_user": {
        "askbookie-pesu": {
            "api_calls": 8500,
            "questions_asked": 7200,
            "uploads_attempted": 45,
            "success_rate": 98.5,
            "average_latency_seconds": 1.25,
            "ask_fails": 12,
            "upload_fails": 3,
            "role": "user"
        }
    }
}
```
---
### GET /

Dashboard endpoint. Returns the service dashboard HTML if available, otherwise returns service metadata.

**Response** (200 OK):
```json
{
    "service": "AskBookie RAG API",
    "version": "1.0.0"
}
```


## Admin Endpoints

### GET /history

Retrieve paginated query history across all users. Useful for analytics and audit trails.

**Request**:
```http
GET /history?limit=50&offset=0 HTTP/1.1
X-API-Key-Id: admin
X-API-Timestamp: 1705234567
X-API-Signature: a1b2c3d4...
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | integer | 100 | Number of records to return |
| `offset` | integer | 0 | Pagination offset |

**Response** (200 OK):
```json
{
    "history": [
        {
            "id": 1,
            "key_id": "askbookie-pesu",
            "subject": "physics",
            "query": "What is the first law of thermodynamics?",
            "answer": "The first law of thermodynamics states that energy cannot be created or destroyed...",
            "sources": [...],
            "request_id": "a1b2c3d4e5f6g7h8",
            "latency_ms": 1250.5,
            "timestamp": 1705234567.0
        }
    ],
    "total": 12350,
    "limit": 50,
    "offset": 0
}
```
---
### GET /admin/keys

List all configured API keys with their metadata.

**Response** (200 OK):
```json
{
    "keys": [
        {
            "key_id": "askbookie-pesu",
            "role": "user",
            "active": true,
            "expires_at": "2025-04-14T00:00:00+00:00"
        },
        {
            "key_id": "admin",
            "role": "admin",
            "active": true,
            "expires_at": null
        }
    ]
}
```
---
### POST /admin/keys/{key_id}/enable

Re-enable a disabled API key.

**Response** (200 OK):
```json
{
    "status": "enabled",
    "key_id": "askbookie-pesu"
}
```
---
### POST /admin/keys/{key_id}/disable

Disable an API key. Disabled keys cannot authenticate requests. The admin key cannot be disabled.

**Response** (200 OK):
```json
{
    "status": "disabled",
    "key_id": "askbookie-pesu"
}
```
---
### GET /admin/models/current

Retrieve the currently active language model configuration.

**Response** (200 OK):
```json
{
    "id": 1,
    "name": "Gemini-3-flash",
    "description": "Gemini Primary API Key"
}
```
---
### POST /admin/models/switch

Switch the active language model. The system supports multiple model backends for failover and experimentation.

**Request**:
```http
POST /admin/models/switch HTTP/1.1
Content-Type: application/json
X-API-Key-Id: admin
X-API-Timestamp: 1705234567
X-API-Signature: a1b2c3d4...

{
    "model_id": 2
}
```


**Response** (200 OK):
```json
{
    "status": "success",
    "message": "Switched to model 2",
    "model": {
        "id": 2,
        "name": "Gemini-3-flash(Back-up)",
        "description": "Gemini Secondary API Key"
    }
}
```
---
## Error Handling

All errors return JSON responses with consistent structure:

```json
{
    "detail": "Error description"
}
```
### HTTP Status Codes

| Code | Meaning |
|------|---------|
| 400 | Bad Request - Invalid parameters, malformed JSON, unsupported file type |
| 401 | Unauthorized - Invalid or expired signature, missing auth headers |
| 403 | Forbidden - Insufficient permissions for admin endpoints |
| 404 | Not Found - Job or resource does not exist |
| 413 | Payload Too Large - File exceeds 10MB or JSON exceeds 16KB |
| 429 | Too Many Requests - Rate limit exceeded, auth lockout, or LLM quota exhausted |
| 500 | Internal Server Error - RAG pipeline failure |


---
### cURL Examples

```bash
generate_sig() {
    local method=$1 path=$2 secret=$3
    local ts=$(date +%s)
    local msg="${ts}"$'\n'"${method}"$'\n'"${path}"
    local sig=$(echo -n "$msg" | openssl dgst -sha256 -hmac "$secret" | cut -d' ' -f2)
    echo "-H 'X-API-Key-Id: $KEY_ID' -H 'X-API-Timestamp: $ts' -H 'X-API-Signature: $sig'"
}

curl -X POST https://pmmdot-askbookie.hf.space/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key-Id: your-key-id" \
  -H "X-API-Timestamp: $(date +%s)" \
  -H "X-API-Signature: <computed>" \
  -d '{"query": "What is thermodynamics?", "subject": "physics"}'

curl -X POST https://pmmdot-askbookie.hf.space/upload \
  -H "X-API-Key-Id: your-key-id" \
  -H "X-API-Timestamp: $(date +%s)" \
  -H "X-API-Signature: <computed>" \
  -F "file=@document.pdf" \
  -F "subject=physics"

curl https://pmmdot-askbookie.hf.space/health
```