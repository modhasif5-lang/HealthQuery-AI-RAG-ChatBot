# ─────────────────────────────────────────────
# retriever_utils.py — Query expansion, RAG Fusion, Hybrid (Vector+BM25),
#                       Merger Retriever, Cross-Encoder, Contextual Compression
# ─────────────────────────────────────────────
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from rank_bm25 import BM25Okapi

from src.config import llm, vectorstore
from src.doc_utils import reciprocal_rank_fusion
from src.reranking_utils import rerank_with_cross_encoder


# ── 1. Query Expansion (with Medical Synonyms) ──
expansion_prompt = ChatPromptTemplate(
    input_variables=["question"],
    messages=[
        SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[],
                template=(
                    "You are a medical search query optimizer. Generate multiple "
                    "search queries based on the input query. Include medical synonyms "
                    "and alternative terminology (e.g., 'heart attack' → 'myocardial "
                    "infarction', 'high blood pressure' → 'hypertension'). "
                    "This helps bridge vocabulary gaps between patient language and "
                    "medical textbook terminology."
                )
            )
        ),
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["question"],
                template=(
                    "Generate multiple search queries related to: {question}\n"
                    "OUTPUT (4 queries):"
                )
            )
        )
    ]
)

generate_queries = (
    expansion_prompt
    | llm
    | StrOutputParser()
    | (lambda x: [q.strip() for q in x.split("\n") if q.strip()])
)


# ── 2. Vector Retriever (Pinecone) ───────────
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 15})


# ── 3. RAG Fusion Chain ──────────────────────
# Query Expansion → Pinecone (per query) → RRF Merge
ragfusion_chain = (
    generate_queries
    | vector_retriever.map()
    | reciprocal_rank_fusion
)


# ── 4. BM25 Keyword Re-Scoring ───────────────
def bm25_rescore(query: str, docs: list) -> list:
    """
    Re-scores documents using BM25 keyword matching on top of vector results.
    This adds a keyword-based signal to complement the semantic vector search.
    
    BM25 excels at finding exact medical terms (drug names, disease codes)
    that embedding models sometimes miss.
    
    Args:
        query : The user's search query
        docs  : List of Documents from vector/RRF retrieval
    
    Returns:
        List of Documents sorted by combined BM25 keyword relevance
    """
    if not docs:
        return []

    # Tokenize documents for BM25
    tokenized_corpus = [doc.page_content.lower().split() for doc in docs]
    tokenized_query = query.lower().split()

    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)

    # Pair scores with documents and sort by descending relevance
    scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs]


# ── 5. Merger Retriever (Vector + BM25 Fusion) ──
def merger_retriever(query: str, fused_docs: list) -> list:
    """
    Merges Vector (semantic) and BM25 (keyword) signals to produce
    a hybrid-ranked list of documents.
    
    Strategy:
      - Vector rank comes from RRF fusion (already done)
      - BM25 rank is computed on the same document set
      - Final score = 0.6 * vector_rank_score + 0.4 * bm25_rank_score
    
    This ensures both semantic meaning AND exact keyword matches 
    contribute to the final ranking.
    
    Args:
        query      : The user's search query
        fused_docs : Documents already ranked by RRF vector fusion
    
    Returns:
        List of Documents sorted by hybrid (vector + BM25) score
    """
    if not fused_docs:
        return []

    # Get BM25 scores
    tokenized_corpus = [doc.page_content.lower().split() for doc in fused_docs]
    tokenized_query = query.lower().split()
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(tokenized_query)

    # Normalize BM25 scores to 0-1 range
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
    bm25_normalized = [s / max_bm25 for s in bm25_scores]

    # Vector rank score (position-based: top doc = 1.0, decreasing)
    n = len(fused_docs)
    vector_rank_scores = [(n - i) / n for i in range(n)]

    # Weighted hybrid combination
    VECTOR_WEIGHT = 0.6
    BM25_WEIGHT = 0.4
    hybrid_scores = [
        VECTOR_WEIGHT * v + BM25_WEIGHT * b
        for v, b in zip(vector_rank_scores, bm25_normalized)
    ]

    # Sort by combined hybrid score
    scored_docs = sorted(zip(hybrid_scores, fused_docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs]


# ── 6. Contextual Compression (BATCHED — Single LLM Call) ──
def contextual_compression(query: str, docs: list, top_n: int = 5) -> list:
    """
    Compresses retrieved documents by extracting ONLY the sentences
    relevant to the user's query using a SINGLE batched LLM call.
    
    Instead of calling the LLM once per document (5 calls), we batch
    all passages into one prompt, saving ~4 round-trips (~2-4 seconds).
    
    Args:
        query : The user's search query
        docs  : Re-ranked documents from cross-encoder
        top_n : Max documents to compress
    
    Returns:
        List of Documents with compressed page_content
    """
    if not docs:
        return []

    # Build a single batched prompt with all passages numbered
    passages_text = ""
    for i, doc in enumerate(docs[:top_n], 1):
        passages_text += f"\n--- PASSAGE {i} ---\n{doc.page_content}\n"

    compression_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a medical text extractor. Given a user question and multiple text passages, "
            "extract ONLY the sentences from each passage that are directly relevant to answering "
            "the question. Output them grouped by passage number. If a passage has nothing relevant, "
            "skip it entirely. Do NOT add commentary, just extract the relevant sentences."
        ),
        ("human", "Question: {question}\n{passages}")
    ])

    try:
        result = llm.invoke(
            compression_prompt.invoke({
                "question": query,
                "passages": passages_text
            })
        ).content.strip()

        if result:
            from langchain_core.documents import Document
            compressed_doc = Document(
                page_content=result,
                metadata=docs[0].metadata if docs else {}
            )
            return [compressed_doc]
    except Exception as e:
        print(f"DEBUG: Batched compression failed, using originals: {e}")

    return docs[:top_n]


# ── Full Retrieval Pipeline ───────────────────
def full_retrieval_pipeline(inputs: dict, fast_mode: bool = True) -> list:
    """
    End-to-end advanced retrieval pipeline:
      1. Query Expansion      -> 4 diverse variants (with medical synonyms)
      2. Vector Retrieval     -> Pinecone semantic search per variant
      3. RRF Fusion           -> Merge + deduplicate by rank score
      4. Hybrid Merger        -> Combine Vector + BM25 keyword signals
      5. Cross-Encoder        -> Fine-grained relevance re-ranking
      6. Contextual Compress  -> Extract relevant sentences (skipped in fast_mode for minimal TTFT)
      7. Return to LLM        -> Clean, focused context for answer generation

    Args:
        inputs    : dict with key "question" (the standalone medical question)
        fast_mode : bool, if True skips extra LLM compression round-trip for 3x faster response

    Returns:
        List of top-ranked Documents ready for LLM context injection
    """
    q = inputs["question"]
    print(f"DEBUG: [1/6] Query Expansion for: {q}")

    # Step 1-3: RAG Fusion (Expansion -> Vector -> RRF)
    fused_docs = ragfusion_chain.invoke({"question": q})
    print(f"DEBUG: [2/6] RRF Fusion returned {len(fused_docs)} docs")

    # Step 4: Hybrid Merger (Vector + BM25)
    hybrid_docs = merger_retriever(q, fused_docs)
    print(f"DEBUG: [3/6] Hybrid Merger ranked {len(hybrid_docs)} docs")

    # Step 5: Cross-Encoder Re-ranking
    reranked_docs = rerank_with_cross_encoder(q, hybrid_docs, top_n=5)
    print(f"DEBUG: [4/6] Cross-Encoder selected top {len(reranked_docs)} docs")

    if fast_mode:
        print(f"DEBUG: [5/6] Fast mode active -> using top cross-encoder reranked passages directly")
        return reranked_docs[:5]

    # Step 6: Contextual Compression
    compressed_docs = contextual_compression(q, reranked_docs, top_n=5)
    print(f"DEBUG: [5/6] Contextual Compression -> {len(compressed_docs)} compressed docs")

    print(f"DEBUG: [6/6] Pipeline complete. Returning context to LLM.")
    return compressed_docs