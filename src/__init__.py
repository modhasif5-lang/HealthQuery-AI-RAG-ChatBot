# src/__init__.py
# Marks this directory as a Python package.
# Import shortcuts for convenience:

from src.chain_utils import run_elite_pipeline, get_session_history
from src.retriever_utils import full_retrieval_pipeline
from src.config import llm, vectorstore, embeddings
# ```

# ---

# ## Dependency Map
# ```
# config.py
#    └── embeddings, vectorstore, llm (shared globals)

# doc_utils.py          ← no internal deps
# reranking_utils.py    ← no internal deps

# retriever_utils.py    ← imports: config, doc_utils, reranking_utils
# chain_utils.py        ← imports: config, doc_utils, retriever_utils
# main.py               ← imports: chain_utils, config
# __init__.py           ← re-exports key symbols for clean external imports