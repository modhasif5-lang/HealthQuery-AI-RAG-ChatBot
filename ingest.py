# ─────────────────────────────────────────────
# ingest.py — Download a free medical PDF, chunk, embed, and upsert to Pinecone
# ─────────────────────────────────────────────
"""
This script populates the Pinecone 'medical-index' with medical textbook data.

Usage:
    python ingest.py                      # Downloads & ingests default sample PDF
    python ingest.py --pdf path/to/file.pdf  # Ingests your own PDF
    python ingest.py --pdf-dir data/       # Ingests all PDFs in a directory

It uses the SAME embedding model (BAAI/bge-large-en) as the chatbot,
so query-time and index-time vectors are aligned.
"""

import os
import sys
import argparse
import requests
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

# ── Reuse the same embeddings from the chatbot config ──
from src.config import embeddings, PINECONE_API_KEY


# ── Sample Medical PDFs (Public Domain / Open Access) ──
SAMPLE_PDFS = [
    {
        "name": "General Pathology Lecture Notes",
        "url": "https://cartercenter.org/resources/pdfs/health/ephti/library/lecture_notes/health_science_students/ln_gen_pathology_final.pdf",
    },
    {
        "name": "Human Anatomy & Physiology",
        "url": "https://cartercenter.org/resources/pdfs/health/ephti/library/lecture_notes/health_science_students/LN_Human_Phys_final.pdf",
    },
]


def download_pdf(url: str, save_dir: str = "data") -> str:
    """Download a PDF from URL and save locally."""
    os.makedirs(save_dir, exist_ok=True)
    filename = url.split("/")[-1]
    filepath = os.path.join(save_dir, filename)

    if os.path.exists(filepath):
        print(f"  ✓ Already downloaded: {filename}")
        return filepath

    print(f"  ↓ Downloading: {filename} ...")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0

    with open(filepath, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = (downloaded / total) * 100
                print(f"\r  ↓ Downloading: {filename} ... {pct:.0f}%", end="", flush=True)

    print(f"\n  ✓ Saved: {filepath} ({downloaded / 1024 / 1024:.1f} MB)")
    return filepath


def load_and_chunk_pdf(filepath: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    """Load a PDF and split into chunks for embedding."""
    print(f"  📄 Loading: {filepath}")
    loader = PyPDFLoader(filepath)
    pages = loader.load()
    print(f"  📄 Loaded {len(pages)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n\n\n", "\n\n", "\n", ". ", "; ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.split_documents(pages)
    print(f"  ✂️  Split into {len(chunks)} chunks")
    return chunks


def upsert_to_pinecone(chunks: list, index_name: str = "medical-index", batch_size: int = 50):
    """Embed chunks and upsert vectors into Pinecone."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(index_name)

    # Check current state
    stats = index.describe_index_stats()
    print(f"\n  📊 Index '{index_name}' before ingestion: {stats['total_vector_count']} vectors")

    total = len(chunks)
    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        texts = [doc.page_content for doc in batch]
        metadatas = [doc.metadata for doc in batch]

        # Embed using the same model as the chatbot
        vectors = embeddings.embed_documents(texts)

        # Create upsert records
        records = []
        for j, (vec, text, meta) in enumerate(zip(vectors, texts, metadatas)):
            doc_id = f"doc_{i + j}"
            meta["text"] = text  # Store text in metadata for retrieval
            records.append({"id": doc_id, "values": vec, "metadata": meta})

        index.upsert(vectors=records)
        print(f"  ⬆️  Upserted batch {i // batch_size + 1}/{(total + batch_size - 1) // batch_size} ({min(i + batch_size, total)}/{total} chunks)")

    # Final stats
    stats = index.describe_index_stats()
    print(f"\n  ✅ Index '{index_name}' after ingestion: {stats['total_vector_count']} vectors")


def main():
    parser = argparse.ArgumentParser(description="Ingest medical PDFs into Pinecone")
    parser.add_argument("--pdf", type=str, help="Path to a single PDF file to ingest")
    parser.add_argument("--pdf-dir", type=str, help="Path to directory of PDFs to ingest")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Text chunk size (default: 1000)")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap (default: 200)")
    args = parser.parse_args()

    print("=" * 60)
    print("🏥 Medical RAG Chatbot — Document Ingestion Pipeline")
    print("=" * 60)

    pdf_files = []

    if args.pdf:
        # Single PDF
        if not os.path.exists(args.pdf):
            print(f"❌ File not found: {args.pdf}")
            sys.exit(1)
        pdf_files.append(args.pdf)

    elif args.pdf_dir:
        # Directory of PDFs
        pdf_dir = Path(args.pdf_dir)
        if not pdf_dir.is_dir():
            print(f"❌ Directory not found: {args.pdf_dir}")
            sys.exit(1)
        pdf_files = [str(p) for p in pdf_dir.glob("*.pdf")]
        if not pdf_files:
            print(f"❌ No PDF files found in {args.pdf_dir}")
            sys.exit(1)
        print(f"\n📂 Found {len(pdf_files)} PDFs in {args.pdf_dir}")

    else:
        # Default: download sample medical PDFs
        print("\n📥 Downloading sample medical textbooks (open access)...\n")
        for sample in SAMPLE_PDFS:
            print(f"  📚 {sample['name']}")
            try:
                filepath = download_pdf(sample["url"])
                pdf_files.append(filepath)
            except Exception as e:
                print(f"  ⚠️  Failed to download {sample['name']}: {e}")

    if not pdf_files:
        print("❌ No PDF files to process. Exiting.")
        sys.exit(1)

    # Process each PDF
    all_chunks = []
    for pdf_path in pdf_files:
        print(f"\n{'─' * 50}")
        chunks = load_and_chunk_pdf(pdf_path, args.chunk_size, args.chunk_overlap)
        all_chunks.extend(chunks)

    print(f"\n{'─' * 50}")
    print(f"📊 Total chunks to ingest: {len(all_chunks)}")
    print(f"{'─' * 50}")

    # Upsert to Pinecone
    print("\n⬆️  Upserting to Pinecone...")
    upsert_to_pinecone(all_chunks)

    print("\n" + "=" * 60)
    print("✅ Ingestion complete! Your chatbot can now answer medical questions.")
    print("=" * 60)


if __name__ == "__main__":
    main()
