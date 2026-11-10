# ─────────────────────────────────────────────
# chain_utils.py — Prompts, memory, intent router, rephraser, final answer chain
# ─────────────────────────────────────────────
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.chat_history import InMemoryChatMessageHistory

from src.config import llm
from src.doc_utils import format_docs
from src.retriever_utils import full_retrieval_pipeline
from src.db_utils import get_postgres_history


# ── Session Memory Store ──────────────────────
def get_session_history(session_id: str):
    """
    Returns (or creates) a persistent chat history in PostgreSQL.
    Falls back to in-memory if DB connection fails.
    """
    history = get_postgres_history(session_id)
    if history:
        return history
    
    # Fallback
    from langchain_core.chat_history import InMemoryChatMessageHistory
    global history_store
    if 'history_store' not in globals():
        globals()['history_store'] = {}
    if session_id not in globals()['history_store']:
        globals()['history_store'][session_id] = InMemoryChatMessageHistory()
    return globals()['history_store'][session_id]


# ── Intent Router ─────────────────────────────
# Classifies query as MEDICAL (→ RAG pipeline) or GENERAL (→ direct LLM)
router_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an intent classifier. Classify the user query into ONLY one of two "
        "categories: 'MEDICAL' (asking about symptoms, drugs, treatments, or anatomy) "
        "or 'GENERAL' (greetings, simple chat, or non-medical info). Output only the word."
    ),
    ("human", "{question}")
])
intent_router = router_prompt | llm | StrOutputParser()


# ── Query Rephraser (Contextualizer) ──────────
# Converts follow-up questions into standalone questions using chat history
rephrase_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert Medical Query Optimizer. Your task is to take a user's medical question and the conversation history, and output a SINGLE standalone, clear medical question.\n\n"
        "RULES:\n"
        "1. DO NOT include any formatting like 'A.' or 'B.' in your output.\n"
        "2. DO NOT include any context from the history that is not directly related to clarifying the user's latest question.\n"
        "3. Output ONLY the rephrased question. No preamble."
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])
query_contextualizer = rephrase_prompt | llm | StrOutputParser()


# ── Answer Prompt ─────────────────────────────
answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an Elite Medical Information System. Provide a professional, direct, and synthesized medical answer based ONLY on the provided textbook context.\n\n"
        "### CORE DIRECTIVES:\n"
        "1. **NO PREAMBLE**: Start your answer immediately. Do NOT say 'Based on the context' or 'Here is the summary'.\n"
        "2. **NO VERBATIM HEADINGS**: Never output letters like 'A.', 'B.' or headings like 'CLINICAL PRESENTATION'. Use natural paragraphs.\n"
        "3. **DETECT & IGNORE NOISE**: Textbooks contain citations (e.g., 'Am J Med 2009') and review questions. **NEVER** include these in your output.\n"
        "4. **EDUCATIONAL USE ONLY**: End your response with exactly one sentence: 'This information is for educational purposes only.'\n\n"
        "### TEXTBOOK DATA:\n{context}"
    ),
    ("human", "{question}")
])


# ── Fast Intent Classifier ─────────────────────
MEDICAL_KEYWORDS = {
    "symptom", "symptoms", "treatment", "treatments", "cause", "causes", "disease", "diseases",
    "infection", "fever", "cough", "pain", "cancer", "diabetes", "heart", "blood", "doctor",
    "medicine", "medication", "dose", "dosage", "syndrome", "diagnosis", "therapy", "patient",
    "virus", "bacteria", "side effect", "side effects", "pressure", "hypertension", "asthma",
    "headache", "throat", "stomach", "rash", "drug", "drugs", "surgery", "pulse", "clinic",
    "pathology", "anatomy", "physiology", "hospital", "physician", "tablet", "pill", "cure"
}

def fast_classify_intent(query: str) -> str:
    q_lower = query.lower()
    words = set(q_lower.split())
    if words.intersection(MEDICAL_KEYWORDS) or any(k in q_lower for k in ["what is", "how to treat", "is it safe", "how do i know", "can i take"]):
        if not any(g in q_lower for g in ["hello", "hi ", "hey ", "who are you", "what is your name"]):
            return "MEDICAL"
    return intent_router.invoke({"question": query}).strip().upper()


# ── Final LangChain Answer Chain ──────────────
final_chain = (
    {
        "context":  RunnableLambda(lambda x: format_docs(full_retrieval_pipeline(x))),
        "question": RunnablePassthrough() | RunnableLambda(lambda x: x["question"])
    }
    | answer_prompt
    | llm
    | StrOutputParser()
)


# ── Elite Pipeline Entry Point ────────────────
def run_elite_pipeline(session_id: str, user_input: str) -> tuple[str, str]:
    """
    Full conversational RAG pipeline:
      1. Retrieve session memory
      2. Route intent (MEDICAL vs GENERAL)
      3. Rephrase follow-up into a standalone question
      4. Run RAG pipeline (medical) or direct LLM (general)
      5. Update memory with the exchange

    Args:
        session_id : Unique identifier for the conversation session
        user_input : Raw user query

    Returns:
        (response: str, intent: str)
    """
    history      = get_session_history(session_id)
    chat_history = history.messages

    if not chat_history:
        intent = fast_classify_intent(user_input)
        standalone_q = user_input
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as executor:
            intent_future = executor.submit(lambda: fast_classify_intent(user_input))
            rephrase_future = executor.submit(lambda: query_contextualizer.invoke({
                "chat_history": chat_history,
                "question": user_input
            }))
            intent = intent_future.result()
            standalone_q = rephrase_future.result()

    print(f"DEBUG: Intent detected  -> {intent}")

    if "MEDICAL" in intent:
        print(f"DEBUG: Rephrased query  -> {standalone_q}")
        docs     = full_retrieval_pipeline({"question": standalone_q}, fast_mode=False)
        
        if not docs:
            response = "I could not find relevant medical information for your question in my knowledge base. Could you rephrase or ask a different medical question?"
        else:
            context  = format_docs(docs)

            response = llm.invoke(
                answer_prompt.invoke({"context": context, "question": standalone_q})).content

    else:
        print("DEBUG: Skipping RAG for general query.")
        general_prompt = ChatPromptTemplate.from_messages([
            (
                "system", 
                "You are a friendly but STRICTLY MEDICAL AI assistant. "
                "You may exchange pleasantries, say hello, and acknowledge the user's name. "
                "However, you MUST politely refuse to answer ANY questions or chat about non-medical topics "
                "(e.g., video games, sports, politics, coding, general trivia). "
                "If the user asks a non-medical question, tell them you are a dedicated medical assistant "
                "and ask how you can help them with health-related queries today."
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        response = llm.invoke(
            general_prompt.invoke({"chat_history": chat_history, "question": user_input})
        ).content

    history.add_user_message(user_input)
    history.add_ai_message(response)

    return response, intent


# ── Streaming Pipeline Entry Point ────────────
def stream_elite_pipeline(session_id: str, user_input: str):
    """
    Streaming version of the RAG pipeline.
    Yields (token, intent, status) tuples as processing happens:
      - Initial yields send status updates to UI for zero-latency feedback
      - Subsequent yields stream token chunks as LLM generates them
    """
    yield "", "PROCESSING", "Analyzing query..."

    history      = get_session_history(session_id)
    chat_history = history.messages

    if not chat_history:
        intent = fast_classify_intent(user_input)
        standalone_q = user_input
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as executor:
            intent_future = executor.submit(lambda: fast_classify_intent(user_input))
            rephrase_future = executor.submit(lambda: query_contextualizer.invoke({
                "chat_history": chat_history,
                "question": user_input
            }))
            intent = intent_future.result()
            standalone_q = rephrase_future.result()

    print(f"DEBUG: [STREAM] Intent -> {intent}")

    if "MEDICAL" in intent:
        yield "", "MEDICAL", "Searching verified medical textbooks..."
        print(f"DEBUG: [STREAM] Rephrased -> {standalone_q}")
        docs = full_retrieval_pipeline({"question": standalone_q}, fast_mode=False)
        if not docs:
            yield "I couldn't find relevant medical information for your question in my knowledge base. Could you rephrase or ask a different medical question?", "MEDICAL", ""
            history.add_user_message(user_input)
            history.add_ai_message("I couldn't find relevant medical information for your question.")
            return
        context = format_docs(docs)
        prompt_value = answer_prompt.invoke({"context": context, "question": standalone_q})
    else:
        yield "", "GENERAL", "Preparing response..."
        print("DEBUG: [STREAM] General query")
        general_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a friendly but STRICTLY MEDICAL AI assistant. "
                "You may exchange pleasantries, say hello, and acknowledge the user's name. "
                "However, you MUST politely refuse to answer ANY questions or chat about non-medical topics "
                "(e.g., video games, sports, politics, coding, general trivia). "
                "If the user asks a non-medical question, tell them you are a dedicated medical assistant "
                "and ask how you can help them with health-related queries today."
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        prompt_value = general_prompt.invoke({"chat_history": chat_history, "question": user_input})

    # Stream tokens from the LLM
    full_response = ""
    for chunk in llm.stream(prompt_value):
        token = chunk.content
        if token:
            full_response += token
            yield token, intent, ""

    # Persist after streaming completes
    history.add_user_message(user_input)
    history.add_ai_message(full_response)
