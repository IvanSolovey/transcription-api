"""
Database session management.
Simple sync session without async complications.
"""
from sqlmodel import Session
from app.db.engine import engine


def get_session():
    """
    FastAPI dependency для отримання DB session.
    
    Usage:
        @app.get("/endpoint")
        def endpoint(session: Session = Depends(get_session)):
            # use session
    """
    with Session(engine) as session:
        yield session


def get_db_session():
    """
    Direct session getter для використання поза FastAPI endpoints.
    
    Usage:
        with get_db_session() as session:
            # use session
    """
    return Session(engine)
