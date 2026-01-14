from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from openai import OpenAI
import os
import uuid
import time
import logging
from threading import Lock
from typing import List
from g4f.client import Client
logger = logging.getLogger("rag-api")

PROMPT_TEMPLATE = """
    You are AskBookie, an assistant built on a RAG system using university slide data.

    Your rules:
    1. If the context has the answer, use it.
    2. If the context is related but incomplete, answer from your knowledge but mention the context.
    3. If unrelated, say it's not in context.
    4. Format your answer nicely in Markdown. Use LaTeX for math ($...$ for inline, $$...$$ for block).

    Context:
    {context}

    Question: {question}
"""

g4f_client = Client()

PROMO_PATTERNS = [
    "want best roleplay experience",
    "llmplayground.net",
    "want the best roleplay",
    "best ai roleplay",
]

class QuotaExhaustedError(Exception):
    pass

def clean_response(text: str) -> str:
    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        line_lower = line.lower().strip()
        if any(pattern in line_lower for pattern in PROMO_PATTERNS):
            continue
        if line_lower.startswith("http") and "llmplayground" in line_lower:
            continue
        clean_lines.append(line)
    while clean_lines and not clean_lines[-1].strip():
        clean_lines.pop()
    return '\n'.join(clean_lines)

MODEL_OPTIONS = {
    1: {"name": "Gemini-3-flash", "description": "Gemini Primary API Key"},
    2: {"name": "Gemini-3-flash(Back-up)", "description": "Gemini Secondary API Key"},
    3: {"name": "Gemini-3-Pro", "description": "Gemini Primary API Key"},
    4: {"name": "GPT-4o-mini", "description": "DuckDuckGo (Free)"},
    5: {"name": "Claude-3-Haiku", "description": "DuckDuckGo (Free)"},
}


class ModelManager:
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        self._current_model = 1  
        self._model_lock = Lock()
        
        self._gemini_primary_key = os.getenv("GEMINI_API_KEY")
        self._gemini_secondary_key = os.getenv("GEMINI_2_API_KEY")
        self._gemini_pro_key = os.getenv("GEMINI_API_KEY")
        self._openai_key = os.getenv("OPENAI_API_KEY")
        
        if self._openai_key:
            self._openai_client = OpenAI(
                api_key=self._openai_key,
                base_url="https://api.chatanywhere.tech/v1"
            )
        else:
            self._openai_client = None
        
        logger.info(f"ModelManager initialized with model {self._current_model}")
    
    @property
    def current_model(self) -> int:
        with self._model_lock:
            return self._current_model
    
    @property
    def current_model_info(self) -> dict:
        with self._model_lock:
            return {
                "model_id": self._current_model,
                **MODEL_OPTIONS[self._current_model]
            }
    
    def switch_model(self, model_id: int) -> dict:
        if model_id not in MODEL_OPTIONS:
            raise ValueError(f"Invalid model ID: {model_id}. Valid options: 1-5")
      
        with self._model_lock:
            old_model = self._current_model
            self._current_model = model_id
            logger.info(f"Model switched from {old_model} to {model_id} ({MODEL_OPTIONS[model_id]['name']})")
        
        return self.current_model_info
    
    def call_llm(self, prompt: str) -> str:
        model_id = self.current_model
        
        if model_id == 1:
            return self._call_gemini(prompt, self._gemini_primary_key, "gemini-2.5-flash")
        elif model_id == 2:
            return self._call_gemini(prompt, self._gemini_secondary_key, "gemini-2.5-flash")
        elif model_id == 3:
            return self._call_gemini(prompt, self._gemini_pro_key, "gemini-2.5-pro")
        elif model_id == 4:
            return self._call_gpt4o(prompt)
        elif model_id == 5:
            return self._call_openai(prompt)
    
    def _call_gemini(self, prompt: str, api_key: str, model: str) -> str:
        try:
            llm = ChatGoogleGenerativeAI(model=model, temperature=0, google_api_key=api_key)
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            error_str = str(e).lower()
            if "429" in str(e) or "quota" in error_str or "resource exhausted" in error_str or "rate limit" in error_str:
                logger.error(f"Gemini quota exhausted: {e}")
                raise QuotaExhaustedError("LLM quota exhausted. Please try again later or switch to a different model.")
            raise
    
    def _call_gpt4o(self, prompt: str) -> str:
        from g4f.Provider import DDG
        response = g4f_client.chat.completions.create(
            model="gpt-4o-mini",
            provider=DDG,
            messages=[{"role": "user", "content": prompt}],
        )
        return clean_response(response.choices[0].message.content)
    
    def _call_openai(self, prompt: str) -> str:
        from g4f.Provider import DDG
        response = g4f_client.chat.completions.create(
            model="claude-3-haiku",
            provider=DDG,
            messages=[{"role": "user", "content": prompt}],
        )
        return clean_response(response.choices[0].message.content)

model_manager = ModelManager()

def call_llm(prompt: str) -> str:
    return model_manager.call_llm(prompt)

def get_embedding_function():
    return HuggingFaceEmbeddings(
        model_name="Alibaba-NLP/gte-modernbert-base",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={"normalize_embeddings": True}
    )


class RAGService:
    def __init__(self):
        self.qdrant_url = os.getenv("QDRANT_CLUSTER_URL")
        self.qdrant_key = os.getenv("QDRANT_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="Alibaba-NLP/gte-modernbert-base",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    def ask(self, query_text: str, subject: str):
        clean_subject = subject.strip().lower().replace(" ", "_")
        collection_name = f"askbookie_{clean_subject}"
        
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_key, timeout=120)
                vectorstore = QdrantVectorStore(client=client, collection_name=collection_name, embedding=self.embeddings)
                results = vectorstore.similarity_search_with_score(query_text, k=5)
                break
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise last_error

        top_results = results[:5]
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in top_results])
        
        full_prompt = PROMPT_TEMPLATE.format(context=context_text, question=query_text)
        answer = call_llm(full_prompt)
        
        # answer = call_llm_gemini(full_prompt, self.gemini_key)
        
        sources = [
            f"{doc.metadata.get('source', 'Unknown')}: Slide {doc.metadata.get('slide_number', 'Unknown')}" 
            for doc, _ in top_results
        ]
        
        return {"answer": answer, "sources": sources}


def process_pdf(file_path: str, original_filename: str, subject: str, status_callback=None) -> str:
    qdrant_url = os.getenv("QDRANT_CLUSTER_URL")
    qdrant_key = os.getenv("QDRANT_API_KEY")
    
    if status_callback:
        status_callback("processing_started")

    docs = load_document(file_path)
    if not docs:
        if status_callback:
            status_callback("failed_no_content")
        raise ValueError("No content found in PDF")
    
    if status_callback:
        status_callback("chunking")
    
    processed_docs = create_slide_chunks(docs, subject, original_filename)
    if not processed_docs:
        if status_callback:
            status_callback("failed_no_content")
        raise ValueError("No content found after processing")
   
    if status_callback:
        status_callback("upserting_to_vector_db")
        
    add_to_qdrant(processed_docs, qdrant_url, qdrant_key, subject)
    
    if status_callback:
        status_callback("completed")
        
    return "done"


def load_document(file_path: str) -> List[Document]:
    loader = PyPDFLoader(file_path)
    return loader.load()


def create_slide_chunks(raw_docs: List[Document], subject: str, original_filename: str) -> List[Document]:
    final_docs = []
    total = len(raw_docs)
    slide_counts = {}

    for i, doc in enumerate(raw_docs):
        page_num = doc.metadata.get("page", 0)
        source = doc.metadata.get("source", original_filename)

        if source not in slide_counts:
            slide_counts[source] = 0
        slide_counts[source] += 1
        current_slide_num = slide_counts[source]

        prev_text = raw_docs[i - 1].page_content if i > 0 else ""
        next_text = raw_docs[i + 1].page_content if i < total - 1 else ""
        combined = prev_text + "\n\n" + doc.page_content + "\n\n" + next_text

        slide_metadata = {
            "id": uuid.uuid4().hex,
            "page": page_num,
            "slide_number": current_slide_num,
            "source": original_filename,
            "subject": subject
        }

        final_docs.append(Document(page_content=combined, metadata=slide_metadata))

    return final_docs


def add_to_qdrant(chunks: List[Document], url: str, api_key: str, subject: str):
    clean_subject = subject.strip().lower().replace(" ", "_")
    collection_name = f"askbookie_{clean_subject}"
    embedding = get_embedding_function()
    dim = len(embedding.embed_query("test"))

    max_retries = 3
    last_error = None
    
    for attempt in range(max_retries):
        try:
            client = QdrantClient(url=url, api_key=api_key, timeout=120)
            
            if not client.collection_exists(collection_name):
                client.create_collection(
                    collection_name,
                    vectors_config=qdrant_models.VectorParams(size=dim, distance=qdrant_models.Distance.COSINE)
                )

            vectorstore = QdrantVectorStore(client=client, collection_name=collection_name, embedding=embedding)
            ids = [doc.metadata["id"] for doc in chunks]
            vectorstore.add_documents(chunks, ids=ids)
            return
            
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            raise last_error
