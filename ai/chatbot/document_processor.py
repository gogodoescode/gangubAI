"""Document loading and chunking logic for the RAG pipeline."""

import os

import fitz  # PyMuPDF
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pptx import Presentation

from ai.chatbot.config import (
    CHROMA_COLLECTION,
    EMBEDDING_MODEL,
    PERSIST_DIRECTORY,
    RAG_DIR,
)

SUPPORTED_EXTENSIONS = (".pptx", ".pdf")


# ── Chunk a single file ───────────────────────────────────
def process_file_to_chunks(file_path, chunk_size=1000, chunk_overlap=200):
    """Parses PPTX and PDF with text splitting for retrieval."""
    ext = os.path.splitext(file_path)[1].lower()
    file_name = os.path.basename(file_path)
    documents = []

    if ext == ".pptx":
        prs = Presentation(file_path)
        for i, slide in enumerate(prs.slides):
            text = "\n".join([s.text.strip() for s in slide.shapes if hasattr(s, "text") and s.text.strip()])
            if text:
                doc = Document(
                    page_content=text,
                    metadata={"source": file_name, "type": "Slide", "number": i + 1}
                )
                documents.append(doc)

    elif ext == ".pdf":
        doc = fitz.open(file_path)
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            if text:
                page_doc = Document(
                    page_content=text,
                    metadata={"source": file_name, "type": "Page", "number": i + 1}
                )
                documents.append(page_doc)

    # Use RecursiveCharacterTextSplitter – keeps larger units (e.g. paragraphs) together
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )

    # Split documents into smaller chunks
    chunks = text_splitter.split_documents(documents)

    # Add the metadata inside the page content so that the LLM can use it to print
    for chunk in chunks:
        original_content = chunk.page_content
        metadata = chunk.metadata
        chunk.page_content = (
            f"[{metadata['type']} {metadata['number']} from {metadata['source']}]\n"
            f"Content: {original_content}"
        )

    return chunks


# ── Build a fresh vector store from source files ──────────
def setup_gangubai_brain(source_dir=RAG_DIR, persist_directory=PERSIST_DIRECTORY):
    """Index every supported document in *source_dir*."""
    print("📚 Loading and processing documents...")
    all_chunks = []
    for name in sorted(os.listdir(source_dir)):
        if not name.lower().endswith(SUPPORTED_EXTENSIONS):
            continue
        path = os.path.join(source_dir, name)
        print(f"  - Processing {path}...")
        chunks = process_file_to_chunks(path)
        all_chunks.extend(chunks)
        print(f"    ✓ Created {len(chunks)} chunks")

    print(f"\n📊 Total chunks created: {len(all_chunks)}")

    print("🔄 Initializing embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("💾 Creating vector database...")
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        collection_name=CHROMA_COLLECTION,
        embedding=embeddings,
        persist_directory=persist_directory,
    )

    print(f"✅ Vector database created with {len(all_chunks)} documents\n")
    return vectorstore


# ── Load or create the vector store (module-level singleton) ──
def get_vectorstore():
    """Return the Chroma vectorstore, loading from disk or building from scratch."""
    if os.path.exists(PERSIST_DIRECTORY):
        print("Loading existing vector database...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vs = Chroma(
            collection_name=CHROMA_COLLECTION,
            embedding_function=embeddings,
            persist_directory=PERSIST_DIRECTORY,
        )
        print("✅ Vector database loaded\n")
        return vs
    return setup_gangubai_brain()
