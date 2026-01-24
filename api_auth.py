"""
Система управління API токенами (SQLite)
"""
import secrets
from datetime import datetime
from typing import Dict, List, Optional
import logging
from app.db.session import get_db_session
from app.db.repositories.api_keys import APIKeyRepository
from app.db.repositories.master_token import MasterTokenRepository

logger = logging.getLogger(__name__)

class APIKeyManager:
    """Менеджер для управління API токенами через SQLite"""
    
    def __init__(self, data_dir: str = "data"):
        # data_dir більше не використовується, але залишаємо для зворотньої сумісності
        self._ensure_master_token()
    
    def _ensure_master_token(self):
        """Створює master токен якщо не існує"""
        with get_db_session() as session:
            repo = MasterTokenRepository(session)
            existing = repo.get_latest()
            
            if not existing:
                master_token = secrets.token_urlsafe(32)
                repo.create(master_token)
                logger.info(f"🔑 Створено master токен: {master_token}")
                logger.info("📋 Збережіть цей токен! Він потрібен для доступу до адмін панелі")
    
    def get_master_token(self) -> str:
        """Отримує master токен"""
        with get_db_session() as session:
            repo = MasterTokenRepository(session)
            token_obj = repo.get_latest()
            if not token_obj:
                raise RuntimeError("Master токен не знайдено в БД")
            return token_obj.token
    
    def verify_master_token(self, token: str) -> bool:
        """Перевіряє master токен"""
        with get_db_session() as session:
            repo = MasterTokenRepository(session)
            return repo.verify(token)
    
    def generate_api_key(self, client_name: str) -> str:
        """Генерує новий API ключ"""
        api_key = secrets.token_urlsafe(32)
        
        with get_db_session() as session:
            repo = APIKeyRepository(session)
            repo.create(
                key=api_key,
                client_name=client_name,
                notes=""
            )
        
        logger.info(f"Створено новий API ключ для клієнта: {client_name}")
        return api_key
    
    def delete_api_key(self, api_key: str) -> bool:
        """Видаляє API ключ"""
        with get_db_session() as session:
            repo = APIKeyRepository(session)
            api_key_obj = repo.get_by_key(api_key)
            
            if api_key_obj:
                client_name = api_key_obj.client_name
                repo.delete(api_key)
                logger.info(f"Видалено API ключ для клієнта: {client_name}")
                return True
            return False
    
    def verify_api_key(self, api_key: str) -> bool:
        """Перевіряє API ключ"""
        with get_db_session() as session:
            repo = APIKeyRepository(session)
            return repo.verify_key(api_key)
    
    def get_api_key_info(self, api_key: str) -> Optional[Dict]:
        """Отримує інформацію про API ключ"""
        with get_db_session() as session:
            repo = APIKeyRepository(session)
            api_key_obj = repo.get_by_key(api_key)
            
            if not api_key_obj:
                return None
            
            return {
                "client_name": api_key_obj.client_name,
                "created_at": api_key_obj.created_at.isoformat(),
                "active": api_key_obj.active,
                "usage_count": api_key_obj.usage_count,
                "last_used": api_key_obj.last_used.isoformat() if api_key_obj.last_used else None,
                "total_requests": api_key_obj.total_requests,
                "successful_requests": api_key_obj.successful_requests,
                "failed_requests": api_key_obj.failed_requests,
                "total_processing_time": api_key_obj.total_processing_time,
                "notes": api_key_obj.notes or ""
            }
    
    def list_api_keys(self) -> List[Dict]:
        """Отримує список всіх API ключів"""
        with get_db_session() as session:
            repo = APIKeyRepository(session)
            all_keys = repo.get_all()
            
            result = []
            for api_key_obj in all_keys:
                avg_time = (api_key_obj.total_processing_time / max(api_key_obj.total_requests, 1) 
                           if api_key_obj.total_requests > 0 else 0)
                
                result.append({
                    "key": api_key_obj.key,
                    "client_name": api_key_obj.client_name,
                    "created_at": api_key_obj.created_at.isoformat(),
                    "active": api_key_obj.active,
                    "usage_count": api_key_obj.usage_count,
                    "last_used": api_key_obj.last_used.isoformat() if api_key_obj.last_used else None,
                    "total_requests": api_key_obj.total_requests,
                    "successful_requests": api_key_obj.successful_requests,
                    "failed_requests": api_key_obj.failed_requests,
                    "total_processing_time": round(api_key_obj.total_processing_time, 2),
                    "average_processing_time": round(avg_time, 2),
                    "notes": api_key_obj.notes or ""
                })
            return result
    
    def get_stats(self) -> Dict:
        """Отримує статистику API ключів"""
        with get_db_session() as session:
            repo = APIKeyRepository(session)
            stats = repo.get_all_statistics()
            
            avg_time = (stats["total_processing_time"] / max(stats["total_requests"], 1) 
                       if stats["total_requests"] > 0 else 0)
            
            return {
                "total_keys": stats["total_keys"],
                "active_keys": stats["active_keys"],
                "inactive_keys": stats["total_keys"] - stats["active_keys"],
                "total_requests": stats["total_requests"],
                "total_processing_time": round(stats["total_processing_time"], 2),
                "average_processing_time": round(avg_time, 2)
            }
    
    def log_api_usage(self, api_key: str, success: bool = True, processing_time: float = 0.0):
        """Логує використання API ключа"""
        try:
            with get_db_session() as session:
                repo = APIKeyRepository(session)
                repo.log_request(api_key, success, processing_time)
        except Exception as e:
            # Не критична помилка - логування може провалитися
            logger.warning(f"Не вдалося залогувати використання API: {e}")
    
    def update_api_key_notes(self, api_key: str, notes: str) -> bool:
        """Оновлює нотатки для API ключа"""
        with get_db_session() as session:
            repo = APIKeyRepository(session)
            api_key_obj = repo.get_by_key(api_key)
            
            if api_key_obj:
                repo.update(api_key, notes=notes)
                return True
            return False
    
    def toggle_api_key_status(self, api_key: str) -> bool:
        """Перемикає статус API ключа (активний/неактивний)"""
        with get_db_session() as session:
            repo = APIKeyRepository(session)
            api_key_obj = repo.get_by_key(api_key)
            
            if api_key_obj:
                new_status = not api_key_obj.active
                if new_status:
                    repo.update(api_key, active=True)
                else:
                    repo.deactivate(api_key)
                return True
            return False
    
    def print_startup_info(self):
        """Виводить інформацію про master токен при запуску"""
        master_token = self.get_master_token()
        logger.info("=" * 60)
        logger.info("🔑 MASTER TOKEN для адмін панелі:")
        logger.info(f"   {master_token}")
        logger.info("=" * 60)
        logger.info("📋 Доступ до адмін панелі:")
        logger.info("   • Статична: http://localhost:8000/admin-panel")
        logger.info("   • Динамічна: http://localhost:8000/admin?master_token=TOKEN")
        logger.info("=" * 60)

# Глобальний екземпляр менеджера
api_key_manager = APIKeyManager()
