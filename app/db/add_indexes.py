"""
Migration script to add indexes to existing database.
Run this after updating models.py with new indexes.
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "transcription.db"

def add_indexes():
    """Add performance indexes to database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Check and create indexes if they don't exist
        indexes = [
            ("idx_tasks_status", "CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)"),
            ("idx_tasks_api_key", "CREATE INDEX IF NOT EXISTS idx_tasks_api_key ON tasks(api_key)"),
            ("idx_tasks_created_at", "CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON tasks(created_at)"),
            ("idx_apikey_active", "CREATE INDEX IF NOT EXISTS idx_apikey_active ON api_keys(active)"),
        ]
        
        for idx_name, sql in indexes:
            print(f"Creating index {idx_name}...")
            cursor.execute(sql)
            print(f"✅ Index {idx_name} created")
        
        conn.commit()
        print("\n✅ All indexes created successfully!")
        
    except Exception as e:
        print(f"❌ Error creating indexes: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    print("Adding performance indexes to database...")
    add_indexes()
