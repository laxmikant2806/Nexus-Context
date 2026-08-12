"""
examples/03_code_analysis.py
============================
Parsing local codebase repository files, extracting import relationship edges into a graph,
and answering code dependency questions.
"""

from context_nexus import ContextNexus

def main():
    nexus = ContextNexus(token_budget=4096)

    # Ingest Python code files
    code_file1 = """
import psycopg2
import os

def connect_db():
    host = os.getenv("DB_HOST", "prod.db.internal")
    return psycopg2.connect(host=host)
"""

    code_file2 = """
from db_module import connect_db

def get_users():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")
    return cursor.fetchall()
"""

    nexus.ingest(code_file1, doc_id="db_module.py")
    nexus.ingest(code_file2, doc_id="user_service.py")

    context = nexus.get_context("Which module imports connect_db and queries users?")
    print("=== Code Dependency Context ===")
    print(context["context_text"])
    print("\n=== Citations ===")
    print(context["citations"])

if __name__ == "__main__":
    main()
