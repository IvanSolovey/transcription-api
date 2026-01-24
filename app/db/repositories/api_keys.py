"""
API Key repository.
All database operations for API keys.
"""
from typing import Optional, List
from datetime import datetime
from sqlmodel import Session, select
from app.db.models import APIKey


class APIKeyRepository:
    """Repository pattern для роботи з API ключами."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def create(self, key: str, client_name: str, notes: Optional[str] = None) -> APIKey:
        """Create new API key."""
        api_key = APIKey(
            key=key,
            client_name=client_name,
            notes=notes
        )
        self.session.add(api_key)
        self.session.commit()
        self.session.refresh(api_key)
        return api_key
    
    def get_by_key(self, key: str) -> Optional[APIKey]:
        """Get API key by key value."""
        statement = select(APIKey).where(APIKey.key == key)
        return self.session.exec(statement).first()
    
    def get_all(self, active_only: bool = False) -> List[APIKey]:
        """Get all API keys."""
        statement = select(APIKey)
        if active_only:
            statement = statement.where(APIKey.active == True)
        return list(self.session.exec(statement).all())
    
    def update(self, key: str, **kwargs) -> Optional[APIKey]:
        """Update API key fields."""
        api_key = self.get_by_key(key)
        if not api_key:
            return None
        
        for field, value in kwargs.items():
            if hasattr(api_key, field):
                setattr(api_key, field, value)
        
        self.session.add(api_key)
        self.session.commit()
        self.session.refresh(api_key)
        return api_key
    
    def deactivate(self, key: str) -> bool:
        """Deactivate API key."""
        api_key = self.get_by_key(key)
        if not api_key:
            return False
        
        api_key.active = False
        self.session.add(api_key)
        self.session.commit()
        return True
    
    def delete(self, key: str) -> bool:
        """Delete API key."""
        api_key = self.get_by_key(key)
        if not api_key:
            return False
        
        self.session.delete(api_key)
        self.session.commit()
        return True
    
    def verify_key(self, key: str) -> bool:
        """Verify if API key exists and is active."""
        api_key = self.get_by_key(key)
        return api_key is not None and api_key.active
    
    def log_request(
        self, 
        key: str, 
        success: bool, 
        processing_time: float = 0.0
    ) -> None:
        """
        Log API request usage with atomic counters.
        Uses SQL UPDATE to avoid race conditions.
        """
        from sqlmodel import update
        from app.db.models import APIKey
        
        # Atomic counter update - no race conditions
        stmt = (
            update(APIKey)
            .where(APIKey.key == key)
            .values(
                usage_count=APIKey.usage_count + 1,
                last_used=datetime.utcnow(),
                total_requests=APIKey.total_requests + 1,
                successful_requests=APIKey.successful_requests + (1 if success else 0),
                failed_requests=APIKey.failed_requests + (0 if success else 1),
                total_processing_time=APIKey.total_processing_time + processing_time
            )
        )
        
        self.session.exec(stmt)
        self.session.commit()
    
    def get_all_statistics(self) -> dict:
        """Get global statistics for all API keys."""
        all_keys = self.get_all()
        
        total_keys = len(all_keys)
        active_keys = sum(1 for k in all_keys if k.active)
        total_requests = sum(k.total_requests for k in all_keys)
        total_processing_time = sum(k.total_processing_time for k in all_keys)
        
        return {
            "total_keys": total_keys,
            "active_keys": active_keys,
            "total_requests": total_requests,
            "total_processing_time": total_processing_time
        }
    
    def get_statistics(self, key: str) -> Optional[dict]:
        """Get usage statistics for API key."""
        api_key = self.get_by_key(key)
        if not api_key:
            return None
        
        return {
            "key": api_key.key,
            "client_name": api_key.client_name,
            "active": api_key.active,
            "created_at": api_key.created_at.isoformat(),
            "total_requests": api_key.total_requests,
            "successful_requests": api_key.successful_requests,
            "failed_requests": api_key.failed_requests,
            "total_processing_time": api_key.total_processing_time,
            "avg_processing_time": (
                api_key.total_processing_time / api_key.total_requests
                if api_key.total_requests > 0 else 0
            )
        }
