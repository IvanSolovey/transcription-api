"""
SQLModel database models.
Simple, explicit, no magic.
"""
from datetime import datetime
from typing import Optional
from enum import Enum
from sqlmodel import SQLModel, Field, Index


class TaskStatus(str, Enum):
    """Task status enum to prevent typos."""
    queued = "queued"  # Waiting in queue
    processing = "processing"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class APIKey(SQLModel, table=True):
    """API key with usage statistics."""
    
    __tablename__ = "api_keys"
    __table_args__ = (
        Index("idx_apikey_active", "active"),
    )
    
    key: str = Field(primary_key=True, max_length=255)
    client_name: str = Field(max_length=255)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    active: bool = Field(default=True)
    last_used: Optional[datetime] = Field(default=None)
    
    # Statistics
    usage_count: int = Field(default=0)  # Deprecated but kept for compatibility
    total_requests: int = Field(default=0)
    successful_requests: int = Field(default=0)
    failed_requests: int = Field(default=0)
    total_processing_time: float = Field(default=0.0)
    
    # Optional metadata
    notes: Optional[str] = Field(default=None, max_length=1000)


class Task(SQLModel, table=True):
    """Transcription task."""
    
    __tablename__ = "tasks"
    __table_args__ = (
        Index("idx_tasks_status", "status"),
        Index("idx_tasks_api_key", "api_key"),
        Index("idx_tasks_created_at", "created_at"),
    )
    
    id: str = Field(primary_key=True, max_length=255)
    api_key: str = Field(foreign_key="api_keys.key", max_length=255)
    
    # Task info
    status: str = Field(default=TaskStatus.queued.value, max_length=50)  # Use enum values
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = Field(default=None)  # When processing began (Fix 7.13)
    completed_at: Optional[datetime] = Field(default=None)
    
    # File info
    filename: str = Field(max_length=500)
    duration_sec: Optional[float] = Field(default=None)
    
    # Model config
    model_size: str = Field(max_length=50)
    has_diarization: bool = Field(default=False)
    
    # Transcription result (stored as JSON string)
    result_json: Optional[str] = Field(default=None)
    
    # Error handling
    error_message: Optional[str] = Field(default=None, max_length=2000)


class MasterToken(SQLModel, table=True):
    """Master authentication token."""
    
    __tablename__ = "master_tokens"
    
    token: str = Field(primary_key=True, max_length=255)
    created_at: datetime = Field(default_factory=datetime.utcnow)
