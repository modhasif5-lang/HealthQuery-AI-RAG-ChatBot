# ─────────────────────────────────────────────
# doc_utils.py — Document helpers: serialize, RRF fusion, format context
# ─────────────────────────────────────────────
import json
import re
from langchain_core.documents import Document


def dumps(doc: Document) -> str:
    """Serialize a Document to a JSON string (used as a dedup key in RRF)."""
    return json.dumps(
        {"page_content": doc.page_content, "metadata": doc.metadata},
        sort_keys=True
    )


def loads(doc_str: str) -> Document:
    """Deserialize a JSON string back into a LangChain Document."""
    data = json.loads(doc_str)
    return Document(page_content=data["page_content"], metadata=data["metadata"])


def reciprocal_rank_fusion(results: list[list], k: int = 60, top_n: int = 15) -> list:
    """
    Fuses multiple ranked document lists using Reciprocal Rank Fusion (RRF).

    Docs appearing consistently at the top across multiple query results
    receive higher cumulative scores and bubble up after fusion.

    Args:
        results : List of ranked doc lists (one per expanded query)
        k       : RRF constant (default 60, controls rank sensitivity)
        top_n   : Number of top docs to return after fusion

    Returns:
        List of top_n fused & deduplicated Documents
    """
    fused_scores: dict[str, float] = {}

    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (k + rank + 1)

    reranked = [
        loads(doc_str)
        for doc_str, _ in sorted(
            fused_scores.items(), key=lambda x: x[1], reverse=True
        )
    ]
    return reranked[:top_n]


def format_docs(docs: list) -> str:
    """Format a list of Documents into a readable context string without metadata and with cleaned text."""
    cleaned_docs = []
    for doc in docs:
        content = doc.page_content
        # 1. Collapse multiple spaces/newlines
        content = re.sub(r'\s+', ' ', content)
        # 2. Fix common PDF artifacts like "T he" or "M edical" (Capital letter separated from word)
        content = re.sub(r'\b([A-Z])\s+([a-z])', r'\1\2', content)
        # 3. Strip out loose multiple choice / heading letters like "A. CLINICAL PRESENTATION" -> "CLINICAL PRESENTATION"
        content = re.sub(r'\b[A-Z]\.\s+', '', content)
        # 5. Strip out common medical citation patterns (e.g., "Am J Med 2009", "2009; 122(12): 6-13", "p. 123", "Vol. 4")
        content = re.sub(r'\b[A-Z][a-z]+\s[A-Z][a-z]+\s[A-Z]?\s?\d{4};\s?\d+', '', content) # e.g. Am J Med 2009
        content = re.sub(r'\d{4};\s\d+[\d\(\):, \-]*', '', content) # e.g. 2009; 122(12): 6-13
        content = re.sub(r'\b(p|pp|Vol|No|Edition|Fig|Table)\.?\s?\d+\b', '', content, flags=re.IGNORECASE)
        # 6. Strip out trailing random isolated characters (artifacts of scanning)
        content = re.sub(r'\s([A-Za-z0-9])\s(?=[A-Za-z0-9]\s)', ' ', content)
        cleaned_docs.append(content.strip())
    
    return "\n\n".join(cleaned_docs)