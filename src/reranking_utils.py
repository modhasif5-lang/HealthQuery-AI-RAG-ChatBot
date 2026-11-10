# ─────────────────────────────────────────────
# reranking_utils.py — Cross-Encoder re-ranking after RRF fusion
# ─────────────────────────────────────────────

_cross_encoder = None

def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder("BAAI/bge-reranker-v2-m3")
    return _cross_encoder


def rerank_with_cross_encoder(query: str, docs: list, top_n: int = 5, min_score: float = 0.1) -> list:
    """
    Re-scores RRF-fused documents using a cross-encoder for fine-grained ranking.
    Filters out documents below the relevance threshold.
    """
    if not docs:
        return []
    try:
        encoder = get_cross_encoder()
        pairs  = [[query, doc.page_content] for doc in docs]
        scores = encoder.predict(pairs)
        scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        # Filter out irrelevant documents below threshold
        filtered = [(s, doc) for s, doc in scored_docs if s >= min_score]
        if not filtered:
            print(f"DEBUG: All docs below relevance threshold ({min_score}). No context available.")
            return []
        return [doc for _, doc in filtered[:top_n]]
    except Exception as e:
        print(f"Cross-encoder reranking skipped due to error: {e}")
        return docs[:top_n]