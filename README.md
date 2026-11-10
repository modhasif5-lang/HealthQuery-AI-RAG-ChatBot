# 🏥 Medical RAG Chatbot

An advanced **Retrieval-Augmented Generation (RAG)** chatbot for medical question answering, powered by a multi-stage retrieval pipeline, Groq's LLaMA 3.3 70B, and Pinecone vector search.

---

## ✨ Features

- 🔍 **Multi-Stage Advanced RAG Pipeline** — Query Expansion → Hybrid Retrieval → RRF Fusion → Cross-Encoder Re-ranking → Contextual Compression
- 🧠 **Intent-Aware Routing** — Automatically classifies queries as `MEDICAL` (RAG pipeline) or `GENERAL` (direct LLM)
- 💬 **Conversational Memory** — PostgreSQL-backed persistent session history with in-memory fallback
- ⚡ **Streaming Responses** — Server-Sent Events (SSE) for real-time token streaming
- 🐳 **Dockerized** — Ready for deployment on AWS EC2 via GitHub Actions CI/CD

---

## 🏗️ Architecture Overview

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│         Intent Router (LLM)         │  ◄── Classifies: MEDICAL or GENERAL
└─────────────────────────────────────┘
    │                   │
  MEDICAL             GENERAL
    │                   │
    ▼                   ▼
┌──────────────┐   ┌──────────────┐
│  Query       │   │  Direct LLM  │
│  Rephraser   │   │  Response    │
└──────────────┘   └──────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                   RETRIEVAL PIPELINE                            │
│                                                                 │
│  [1] Query Expansion  →  4 semantically diverse query variants  │
│         │                                                       │
│  [2] Vector Search    →  Pinecone (top-10 per query variant)    │
│         │                                                       │
│  [3] Hybrid Merger    →  60% Vector + 40% BM25 keyword signals  │
│         │                                                       │
│  [4] RRF Fusion       →  Deduplicate & rank by position scores  │
│         │                                                       │
│  [5] Cross-Encoder    →  ms-marco-MiniLM-L-6-v2 re-ranking     │
│         │                                                       │
│  [6] Contextual       →  Batched LLM compression (1 call)      │
│       Compression                                               │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│      Answer Generation (LLM)        │  ◄── LLaMA 3.3 70B via Groq
└─────────────────────────────────────┘
    │
    ▼
Streaming Response (SSE)
```

---

## 🔬 RAG Pipeline — Deep Dive

### Stage 1 — Query Expansion
**File:** `src/retriever_utils.py` → `generate_queries`

The user's raw query is sent to the LLM, which generates **4 diverse search variants** with medical synonym substitution (e.g., *"heart attack"* → *"myocardial infarction"*). This bridges the vocabulary gap between lay patient language and formal textbook terminology.

```python
# Example expansion for "chest pain on exertion"
1. "Angina pectoris symptoms and treatment"
2. "Exertional chest pain differential diagnosis"
3. "Stable vs unstable angina distinction"
4. "Myocardial ischemia causes and management"
```

---

### Stage 2 — Vector Retrieval (Pinecone)
**File:** `src/config.py` → `vectorstore`, `src/retriever_utils.py` → `vector_retriever`

Each of the 4 expanded queries is independently run against **Pinecone**, retrieving the top-10 semantically similar document chunks using `BAAI/bge-large-en` embeddings (1024-dim, normalized).

| Component | Detail |
|-----------|--------|
| Embedding Model | `BAAI/bge-large-en` (HuggingFace BGE) |
| Index | Pinecone `medical-index` |
| Results per query | Top 10 chunks |
| Total candidates | Up to 40 raw chunks |

---

### Stage 3 — Hybrid Merger (Vector + BM25)
**File:** `src/retriever_utils.py` → `merger_retriever`

The retrieved documents from Pinecone are re-scored using **BM25 keyword matching** (via `rank-bm25`) and combined with the vector rank signal:

```
hybrid_score = 0.6 × vector_rank_score + 0.4 × BM25_normalized_score
```

This hybrid approach ensures that exact medical terms (drug names, ICD codes, rare terminology) that embedding models may miss are still surfaced by the keyword-based BM25 component.

---

### Stage 4 — Reciprocal Rank Fusion (RRF)
**File:** `src/doc_utils.py` → `reciprocal_rank_fusion`

Results from all 4 query variants (now hybrid-ranked) are **merged and deduplicated** via RRF. A document's score is cumulative across all lists where it appears:

```
score(doc) = Σ  1 / (k + rank_i)    [k=60 by default]
```

Documents that consistently rank near the top across multiple query expansions receive higher fused scores and rise above documents that appear in only one list. The top-5 unique documents are returned.

---

### Stage 5 — Cross-Encoder Re-ranking
**File:** `src/reranking_utils.py` → `rerank_with_cross_encoder`

The hybrid-ranked candidates are passed through `cross-encoder/ms-marco-MiniLM-L-6-v2`. Unlike bi-encoders, cross-encoders **jointly encode the query and each document together**, producing a much more accurate relevance score. The top-5 documents by cross-encoder score are selected.

```python
# Cross-encoder scores (query, passage) pairs jointly
pairs  = [[query, doc.page_content] for doc in docs]
scores = cross_encoder.predict(pairs)
```

---

### Stage 6 — Contextual Compression
**File:** `src/retriever_utils.py` → `contextual_compression`

Rather than sending full raw document chunks to the LLM, a single **batched LLM call** extracts only the sentences directly relevant to the user's question. All top-5 documents are compressed in one API call (instead of 5 separate calls), reducing latency by ~2–4 seconds.

The compressed, focused context is then injected into the **answer prompt**.

---

### Conversational Context: Query Rephraser
**File:** `src/chain_utils.py` → `query_contextualizer`

Before the retrieval pipeline, the user's message is enriched with **session chat history** to produce a standalone, self-contained question. This resolves coreferences like *"What are its side effects?"* → *"What are the side effects of Metformin?"*

---

### Intent Router (Parallel Execution)
**File:** `src/chain_utils.py` → `intent_router`

Intent classification and query rephrasing run **in parallel** via `ThreadPoolExecutor(max_workers=2)`, saving ~0.5–1s per request. Intent is either:

- **`MEDICAL`** → Full 6-stage RAG pipeline is triggered
- **`GENERAL`** → RAG is bypassed; LLM answers directly with chat history context

---

## 🗂️ Project Structure

```
MEDICAL-RAG-CHATBOT/
├── app.py                  # FastAPI entry point (REST + SSE streaming endpoints)
├── src/
│   ├── config.py           # Embeddings, Pinecone vectorstore, Groq LLM setup
│   ├── chain_utils.py      # Intent router, rephraser, pipeline orchestration
│   ├── retriever_utils.py  # Query expansion, RRF, BM25, cross-encoder, compression
│   ├── reranking_utils.py  # Cross-encoder model wrapper
│   ├── db_utils.py         # PostgreSQL chat history persistence
│   └── doc_utils.py        # RRF algorithm, document serialization, text cleaning
├── templates/
│   └── index.html          # Frontend chat UI
├── static/                 # CSS / JS assets
├── data/                   # Source medical PDFs (offline ingestion)
├── Dockerfile              # Production container definition
├── .github/workflows/      # GitHub Actions CI/CD → AWS EC2
└── requirements.txt
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | LLaMA 3.3 70B Versatile (via [Groq](https://groq.com)) |
| **Embeddings** | `BAAI/bge-large-en` (HuggingFace) |
| **Vector DB** | [Pinecone](https://pinecone.io) |
| **Cross-Encoder** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **BM25** | `rank-bm25` |
| **Framework** | LangChain + FastAPI |
| **Memory** | PostgreSQL (`langchain-postgres`) |
| **Frontend** | Jinja2 + HTML/CSS/JS (SSE streaming) |
| **Infra** | Docker + AWS EC2 + GitHub Actions |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- A [Pinecone](https://pinecone.io) account with a `medical-index` created
- A [Groq](https://console.groq.com) API key
- PostgreSQL instance (optional — falls back to in-memory if unavailable)

### 1. Clone & Install

```bash
git clone https://github.com/AjAyrAo43/MEDICAL-RAG-CHATBOT.git
cd MEDICAL-RAG-CHATBOT
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file:

```env
PINECONE_API_KEY=your_pinecone_api_key
GROQ_API_KEY=your_groq_api_key
DATABASE_URL=postgresql://user:password@host:5432/dbname   # optional
```

### 3. Run the Application

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at `http://localhost:8000`.

---

### 🐳 Docker

```bash
docker build -t medical-rag-chatbot .
docker run -p 8000:8000 --env-file .env medical-rag-chatbot
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Chat UI (HTML) |
| `POST` | `/chat` | Standard JSON response |
| `POST` | `/chat/stream` | Token-by-token SSE streaming |

### `/chat/stream` SSE Format
```json
data: {"token": "Hypertension", "intent": "MEDICAL", "done": false}
data: {"token": " is characterized by...", "intent": "MEDICAL", "done": false}
data: {"token": "", "intent": "", "done": true}
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.