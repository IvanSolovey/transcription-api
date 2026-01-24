"""
SQLite engine configuration.
Single-threaded SQLite with check_same_thread=False for FastAPI.
"""
from sqlmodel import create_engine
from sqlalchemy import event
from pathlib import Path

# Database file location
DB_PATH = Path(__file__).parent.parent.parent / "data" / "transcription.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# SQLite connection string
DATABASE_URL = f"sqlite:///{DB_PATH}"

# Engine configuration
# check_same_thread=False: allows multiple threads to use same connection
# connect_args for SQLite-specific settings
engine = create_engine(
    DATABASE_URL,
    echo=False,  # Set to True for SQL query logging
    connect_args={"check_same_thread": False}
)

# SQLite PRAGMA settings for production
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """
    Configure SQLite for production use:
    - WAL mode: allows concurrent reads with writes
    - NORMAL sync: balance between speed and safety
    - Foreign keys: enforce referential integrity
    """
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA synchronous=NORMAL;")
    cursor.execute("PRAGMA foreign_keys=ON;")
    cursor.close()
