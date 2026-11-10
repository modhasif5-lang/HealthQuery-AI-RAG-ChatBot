# ─────────────────────────────────────────────
# config.py — Global setup: env, device, LLM, embeddings, vectorstore
# ─────────────────────────────────────────────
import os
import torch
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

# ── Device & CPU Throttling ───────────────────
import torch
torch.set_num_threads(5)  # Set PyTorch to utilize exactly 5 CPU threads
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device} (configured with {torch.get_num_threads()} CPU threads)")

# ── API Keys ──────────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
GROQ_API_KEY     = os.getenv("GROQ_API_KEY", "")

if PINECONE_API_KEY:
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"]     = GROQ_API_KEY

# ── Zero-RAM Cloud Embeddings Client ─────────────────
from langchain_core.embeddings import Embeddings
import requests

class CloudInferenceEmbeddings(Embeddings):
    """
    Zero-RAM Cloud Embeddings client for BAAI/bge-large-en.
    Queries Hugging Face Router over HTTPS with multi-endpoint fallback.
    """
    def __init__(self, model_name="BAAI/bge-large-en", api_key=None):
        self.model_name = model_name
        self.api_key = api_key
        self.endpoints = [
            ("https://router.huggingface.co/hf-inference/v1/embeddings", "openai"),
            (f"https://router.huggingface.co/pipeline/feature-extraction/{model_name}", "raw"),
            (f"https://api-inference.huggingface.co/models/{model_name}", "raw")
        ]

    def _get_headers(self):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _fetch_single(self, text: str) -> list[float]:
        headers = self._get_headers()
        last_err = None

        for url, style in self.endpoints:
            try:
                if style == "openai":
                    payload = {"model": self.model_name, "input": text}
                else:
                    payload = {"inputs": text, "options": {"wait_for_model": True}}

                res = requests.post(url, headers=headers, json=payload, timeout=15)
                if res.status_code == 200:
                    data = res.json()
                    if style == "openai" and isinstance(data, dict) and "data" in data and len(data["data"]) > 0:
                        return data["data"][0]["embedding"]
                    elif isinstance(data, list):
                        if len(data) > 0 and isinstance(data[0], list):
                            return [sum(col) / len(col) for col in zip(*data)]
                        return data
            except Exception as e:
                last_err = e
                continue

        print(f"Warning: All embedding endpoints failed ({last_err}), using fallback vector.")
        return [0.0] * 1024

    def embed_query(self, text: str) -> list[float]:
        return self._fetch_single(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._fetch_single(t) for t in texts]

# ── Embeddings Selection ───────────────────────
hf_token = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
is_cloud = os.getenv("RENDER") is not None or os.getenv("PORT") is not None or os.getenv("USE_CLOUD_EMBEDDINGS", "").lower() == "true"

if is_cloud:
    print("Using zero-CPU CloudInferenceEmbeddings via Hugging Face API...")
    embeddings = CloudInferenceEmbeddings("BAAI/bge-large-en-v1.5", api_key=hf_token)
else:
    try:
        embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
            query_instruction="Represent this sentence for searching relevant passages:"
        )
    except Exception as e:
        print(f"Local BGE embeddings failed ({e}), using CloudInferenceEmbeddings...")
        embeddings = CloudInferenceEmbeddings("BAAI/bge-large-en-v1.5", api_key=hf_token)

# ── Pinecone Vectorstore ──────────────────────
pc = Pinecone(api_key=PINECONE_API_KEY)

vectorstore = PineconeVectorStore(
    index_name="medical-index",
    embedding=embeddings,
    pinecone_api_key=PINECONE_API_KEY
)

# ── LLM ──────────────────────────────────────
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=0
)