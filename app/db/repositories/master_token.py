"""
Master Token repository.
Simple authentication token management.
"""
from typing import Optional
from sqlmodel import Session, select
from app.db.models import MasterToken


class MasterTokenRepository:
    """Repository для роботи з master tokens."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def create(self, token: str) -> MasterToken:
        """Create new master token."""
        master_token = MasterToken(token=token)
        self.session.add(master_token)
        self.session.commit()
        self.session.refresh(master_token)
        return master_token
    
    def get(self, token: str) -> Optional[MasterToken]:
        """Get master token."""
        statement = select(MasterToken).where(MasterToken.token == token)
        return self.session.exec(statement).first()
    
    def get_latest(self) -> Optional[MasterToken]:
        """Get latest master token."""
        statement = select(MasterToken).order_by(MasterToken.created_at.desc())
        return self.session.exec(statement).first()
    
    def verify(self, token: str) -> bool:
        """Verify if master token exists."""
        return self.get(token) is not None
    
    def delete(self, token: str) -> bool:
        """Delete master token."""
        master_token = self.get(token)
        if not master_token:
            return False
        
        self.session.delete(master_token)
        self.session.commit()
        return True
