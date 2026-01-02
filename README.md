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
- **Intelligent Q&A** - Ask questions about your documents using Gemini LLM
- **Per-Key Metrics** - Track usage metrics for each API key
- **Background Job Processing** - Async PDF processing with automatic retry logic
- **Admin Controls** - Separate admin and user permissions

## API Endpoints

- `GET /` - Health check
- `POST /embed` - Upload and process PDF documents
- `POST /ask` - Ask questions about documents
- `GET /jobs/{job_id}` - Check job status
- `GET /jobs` - List all jobs
- `GET /metrics` - View API metrics (admin sees all, users see own)
- `DELETE /jobs/{job_id}` - Delete a job (admin only)

## Environment Variables

Required environment variables:

```
GEMINI_API_KEY=your-gemini-api-key
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
- **Gemini 2.5 Flash** - LLM for answering questions
- **HuggingFace Embeddings** - gte-modernbert-base for document embeddings
- **LangChain** - RAG orchestration

## Documentation

Interactive API documentation available at `/docs` when running.

# testing