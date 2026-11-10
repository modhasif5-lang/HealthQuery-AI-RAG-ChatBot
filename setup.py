from setuptools import setup, find_packages

setup(
    name="medical-rag-chatbot",
    version="0.1.0",
    description="Medical RAG Chatbot using RAG Fusion, Hybrid Retrieval and Cross-Encoder Reranking",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "langchain-community",
        "langchain-core",
        "langchain-pinecone",
        "langchain-groq",
        "pinecone-client",
        "sentence-transformers",
        "torch",
        "pypdf",
        "rank-bm25",
        "fastapi",
        "uvicorn",
        "jinja2",
        "python-dotenv",
        "huggingface-hub",
    ],
)