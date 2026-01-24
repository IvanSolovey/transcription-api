"""
Task repository.
All database operations for transcription tasks.
"""
from typing import Optional, List
from datetime import datetime
from sqlmodel import Session, select, or_
from app.db.models import Task, TaskStatus


class TaskRepository:
    """Repository pattern для роботи з tasks."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def create(
        self,
        task_id: str,
        api_key: str,
        filename: str,
        model_size: str,
        has_diarization: bool = False,
        status: str = TaskStatus.queued.value
    ) -> Task:
        """Create new task."""
        task = Task(
            id=task_id,
            api_key=api_key,
            filename=filename,
            model_size=model_size,
            has_diarization=has_diarization,
            status=status
        )
        self.session.add(task)
        self.session.commit()
        self.session.refresh(task)
        return task
    
    def get_by_id(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        statement = select(Task).where(Task.id == task_id)
        return self.session.exec(statement).first()
    
    def get_by_api_key(self, api_key: str, limit: int = 100) -> List[Task]:
        """Get all tasks for specific API key."""
        statement = (
            select(Task)
            .where(Task.api_key == api_key)
            .order_by(Task.created_at.desc())
            .limit(limit)
        )
        return list(self.session.exec(statement).all())
    
    def get_by_api_key_paginated(
        self,
        api_key: str,
        limit: int = 50,
        offset: int = 0,
        status: Optional[str] = None
    ) -> tuple[List[Task], int]:
        """Get paginated tasks for API key with total count."""
        from sqlalchemy import func
        
        # Build base query
        statement = select(Task).where(Task.api_key == api_key)
        
        # Add status filter if provided
        if status:
            statement = statement.where(Task.status == status)
        
        # Get total count (efficient count query)
        count_statement = select(func.count()).select_from(Task).where(Task.api_key == api_key)
        if status:
            count_statement = count_statement.where(Task.status == status)
        
        total = self.session.exec(count_statement).one()
        
        # Get paginated results
        statement = statement.order_by(Task.created_at.desc()).offset(offset).limit(limit)
        tasks = list(self.session.exec(statement).all())
        
        return tasks, total
    
    def get_by_status(self, status: str, limit: int = 100) -> List[Task]:
        """Get tasks by status."""
        statement = (
            select(Task)
            .where(Task.status == status)
            .order_by(Task.created_at.desc())
            .limit(limit)
        )
        return list(self.session.exec(statement).all())
    
    def get_pending_tasks(self, limit: int = 100) -> List[Task]:
        """Get queued or processing tasks."""
        statement = (
            select(Task)
            .where(or_(Task.status == "queued", Task.status == "processing"))
            .order_by(Task.created_at)
            .limit(limit)
        )
        return list(self.session.exec(statement).all())
    
    def update_status(
        self,
        task_id: str,
        status: str,
        error_message: Optional[str] = None
    ) -> Optional[Task]:
        """Update task status."""
        task = self.get_by_id(task_id)
        if not task:
            return None
        
        task.status = status
        
        if error_message:
            task.error_message = error_message
        
        # Fix 7.13: Track when processing actually started
        if status == TaskStatus.processing.value and not task.started_at:
            task.started_at = datetime.utcnow()
        
        if status in [TaskStatus.completed.value, TaskStatus.failed.value, TaskStatus.cancelled.value]:
            task.completed_at = datetime.utcnow()
        
        self.session.add(task)
        self.session.commit()
        self.session.refresh(task)
        return task
    
    def claim_for_processing(self, task_id: str) -> bool:
        """
        Atomically claim task for processing using database-level locking.
        
        ⚠️ RESERVED FOR FUTURE USE - Not currently called in production.
        
        Purpose:
            Prevents double-processing in multi-process deployments by using
            atomic UPDATE WHERE status='queued' with SQLite row-level locking.
        
        Current State:
            - Single-process deployment makes this unnecessary
            - asyncio.Queue provides in-process task distribution
            - Method retained for future horizontal scaling
        
        Integration Requirements (when needed):
            1. Call this BEFORE calling process_transcription_task_sync()
            2. If returns False, skip processing (task already claimed)
            3. Enables safe multi-worker deployment across processes
        
        Returns:
            True if successfully claimed (status was 'queued', now 'processing')
            False if already processing (race condition detected)
        
        Example Future Usage:
            if not repo.claim_for_processing(task_id):
                logger.warning(f"Task {task_id} already claimed by another worker")
                return  # Skip processing
        """
        from sqlmodel import update
        
        stmt = (
            update(Task)
            .where(Task.id == task_id)
            .where(Task.status == TaskStatus.queued.value)
            .values(
                status=TaskStatus.processing.value,
                started_at=datetime.utcnow()  # Fix 7.13: Track processing start time
            )
        )
        
        result = self.session.exec(stmt)
        self.session.commit()
        
        # If rowcount is 0, task was already claimed
        return result.rowcount > 0
    
    def update_duration(self, task_id: str, duration_sec: float) -> Optional[Task]:
        """Update task duration."""
        task = self.get_by_id(task_id)
        if not task:
            return None
        
        task.duration_sec = duration_sec
        self.session.add(task)
        self.session.commit()
        self.session.refresh(task)
        return task
    
    def mark_completed(
        self,
        task_id: str,
        duration_sec: Optional[float] = None,
        result_json: Optional[str] = None
    ) -> Optional[Task]:
        """Mark task as completed."""
        task = self.get_by_id(task_id)
        if not task:
            return None
        
        task.status = "completed"
        task.completed_at = datetime.utcnow()
        if duration_sec:
            task.duration_sec = duration_sec
        if result_json:
            task.result_json = result_json
        
        self.session.add(task)
        self.session.commit()
        self.session.refresh(task)
        return task
    
    def mark_failed(self, task_id: str, error_message: str) -> Optional[Task]:
        """Mark task as failed."""
        task = self.get_by_id(task_id)
        if not task:
            return None
        
        task.status = "failed"
        task.completed_at = datetime.utcnow()
        task.error_message = error_message
        
        self.session.add(task)
        self.session.commit()
        self.session.refresh(task)
        return task
    
    def delete(self, task_id: str) -> bool:
        """Delete task."""
        task = self.get_by_id(task_id)
        if not task:
            return False
        
        self.session.delete(task)
        self.session.commit()
        return True
    
    def get_statistics(self, api_key: Optional[str] = None) -> dict:
        """Get task statistics."""
        statement = select(Task)
        if api_key:
            statement = statement.where(Task.api_key == api_key)
        
        tasks = list(self.session.exec(statement).all())
        
        total = len(tasks)
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        queued = sum(1 for t in tasks if t.status == "queued")
        processing = sum(1 for t in tasks if t.status == "processing")
        
        total_duration = sum(
            t.duration_sec for t in tasks 
            if t.duration_sec is not None
        )
        
        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "queued": queued,
            "processing": processing,
            "total_duration_sec": total_duration,
            "avg_duration_sec": total_duration / completed if completed > 0 else 0
        }
