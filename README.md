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

## API Endpoints

- `GET /` - Dashboard UI
- `GET /health` - Health check with metrics
- `POST /upload` - Upload and process PDF documents
- `POST /ask` - Ask questions about documents
- `GET /jobs/{job_id}` - Check job status
- `GET /jobs` - List all jobs for the authenticated user
- `GET /history` - View query history (admin only)
- `GET /admin/keys` - List all API keys (admin only)
- `POST /admin/keys/{key_id}/enable` - Enable an API key (admin only)
- `POST /admin/keys/{key_id}/disable` - Disable an API key (admin only)

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
