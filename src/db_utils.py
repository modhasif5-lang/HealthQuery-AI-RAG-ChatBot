# ─────────────────────────────────────────────
# db_utils.py — PostgreSQL connection & chat history persistence
# ─────────────────────────────────────────────
import os
from dotenv import load_dotenv
from langchain_postgres import PostgresChatMessageHistory
import psycopg

load_dotenv()

# Load connection string from environment
DATABASE_URL = os.getenv("DATABASE_URL")

import uuid

def get_postgres_history(session_id: str, table_name: str = "chat_history"):
    """
    Returns a PostgresChatMessageHistory object for the given session.
    If DATABASE_URL is missing, it returns None.
    """
    if not DATABASE_URL:
        print("DEBUG: DATABASE_URL not found in environment. Persistent memory disabled.")
        return None

    try:
        # Normalize session_id to a valid UUID if needed
        try:
            uuid.UUID(session_id)
            valid_session_id = session_id
        except (ValueError, AttributeError):
            valid_session_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(session_id)))

        # Initialize a psycopg connection correctly for newer langchain-postgres
        conn = psycopg.connect(DATABASE_URL)
        
        # Initialize the history object using positional-only arguments for table_name and session_id
        # and passing the sync_connection object.
        history = PostgresChatMessageHistory(
            table_name,
            valid_session_id,
            sync_connection=conn
        )
        
        # Ensure the table exists
        PostgresChatMessageHistory.create_tables(conn, table_name)
        
        print(f"DEBUG: Successfully connected to PostgreSQL for session: {session_id} (UUID: {valid_session_id})")
        return history
    except Exception as e:
        print(f"ERROR: Failed to connect to PostgreSQL: {e}")
        return None

