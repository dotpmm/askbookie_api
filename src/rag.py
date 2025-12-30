from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
import os
import uuid
from typing import Optional, List

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
        
        self.client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_key)

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0,
            google_api_key=self.gemini_key
        )
        
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    def ask(self, query_text: str, subject: str):
        clean_subject = subject.strip().lower().replace(" ", "_")
        collection_name = f"askbookie_{clean_subject}"
        
        vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.embeddings
        )

        results = vectorstore.similarity_search_with_score(
            query_text, 
            k=5
        )

        top_results = results[:3]
        
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in top_results])
        
        chain = self.prompt | self.llm
        response = chain.invoke({"context": context_text, "question": query_text})
        
        sources = [
            f"{doc.metadata.get('source', 'Unknown')}: Slide {doc.metadata.get('slide_number', 'Unknown')}" 
            for doc, _ in top_results
        ]
        
        return {
            "answer": response.content,
            "sources": sources
        }


def get_embedding_function():
    return HuggingFaceEmbeddings(
        model_name="Alibaba-NLP/gte-modernbert-base",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={"normalize_embeddings": True}
    )


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
    docs = loader.load()
    return docs


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

        unique_id = uuid.uuid4().hex

        slide_metadata = {
            "id": unique_id,
            "page": page_num,
            "slide_number": current_slide_num,
            "source": original_filename,
            "subject": subject
        }

        final_docs.append(
            Document(page_content=combined, metadata=slide_metadata)
        )

    return final_docs


def add_to_qdrant(chunks: List[Document], url: str, api_key: str, subject: str):
    clean_subject = subject.strip().lower().replace(" ", "_")
    collection_name = f"askbookie_{clean_subject}"

    embedding = get_embedding_function()
    dim = len(embedding.embed_query("pmm the goat!"))

    client = QdrantClient(url=url, api_key=api_key)
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name,
            vectors_config=qdrant_models.VectorParams(size=dim, distance=qdrant_models.Distance.COSINE)
        )

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding
    )
    
    ids = [doc.metadata["id"] for doc in chunks]
    vectorstore.add_documents(chunks, ids=ids)
