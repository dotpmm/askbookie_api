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

A production-ready RAG (Retrieval Augmented Generation) API for document Q&A using university slide data.

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

## Environment Variables

Required environment variables:

```
QDRANT_CLUSTER_URL=your-qdrant-cluster-url
QDRANT_API_KEY=your-qdrant-api-key
ADMIN_API_KEY=your-admin-key
USER_API_KEY_1=your-user-key-1
USER_API_KEY_2=your-user-key-2
USER_API_KEY_3=your-user-key-3
```

## Tech Stack

- **FastAPI** - Modern Python web framework
- **Qdrant** - Vector database for embeddings
- **LLM Providers** - Gemini 3 Flash and GPT-4o via multiple free/paid providers with effectively unlimited calls per minute
- **SQLite** - lightweight relational store for metadata and jobs
- **HuggingFace Embeddings** - gte-modernbert-base for document embeddings
- **LangChain** - RAG orchestration

## Documentation

Interactive API documentation available at `/docs` when running.
