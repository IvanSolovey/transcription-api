"""
Initialize database tables.
Run this once to create schema.
"""
from sqlmodel import SQLModel
from app.db.engine import engine, DB_PATH
from app.db.models import APIKey, Task, MasterToken


def init_db():
    """Create all tables in database."""
    print(f"Creating database at: {DB_PATH}")
    SQLModel.metadata.create_all(engine)
    print("✅ Database tables created successfully")


def drop_all():
    """Drop all tables (use with caution!)."""
    print(f"Dropping all tables from: {DB_PATH}")
    SQLModel.metadata.drop_all(engine)
    print("✅ All tables dropped")


def reset_db():
    """Drop and recreate all tables."""
    drop_all()
    init_db()


if __name__ == "__main__":
    # Run this script to initialize database
    init_db()
