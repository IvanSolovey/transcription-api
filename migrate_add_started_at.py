#!/usr/bin/env python3
"""
Migration: Add started_at column to tasks table
Phase 3 Fix 7.13 - Track when processing actually started
"""
import sqlite3
import os

DB_PATH = "data/transcription.db"

def migrate():
    """Add started_at column to tasks table if it doesn't exist."""
    
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at {DB_PATH}")
        return False
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if column exists
        cursor.execute("PRAGMA table_info(tasks)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if 'started_at' in columns:
            print("✅ Column 'started_at' already exists in tasks table")
            conn.close()
            return True
        
        # Add the column
        print("Adding 'started_at' column to tasks table...")
        cursor.execute("""
            ALTER TABLE tasks 
            ADD COLUMN started_at TIMESTAMP
        """)
        
        conn.commit()
        conn.close()
        
        print("✅ Migration completed successfully!")
        print("   Column 'started_at' added to tasks table")
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Migration: Add started_at column (Phase 3 Fix 7.13)")
    print("=" * 60)
    print()
    
    success = migrate()
    
    if success:
        print()
        print("✅ Database migration completed!")
        print("   You can now restart the server.")
    else:
        print()
        print("❌ Migration failed. Please check the error above.")
        exit(1)
