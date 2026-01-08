"""
Система управління API токенами
"""
import json
import secrets
import os
import fcntl
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

class FileLockTimeout(Exception):
    """Виняток при неможливості отримати блокування файлу"""
    pass

class APIKeyManager:
    """Менеджер для управління API токенами"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.api_keys_file = self.data_dir / "api_keys.json"
        self.master_token_file = self.data_dir / "master_token.txt"
        
        # Ініціалізуємо файли якщо не існують
        self._init_files()
    
    @contextmanager
    def _file_lock(self, lock_type=fcntl.LOCK_EX, timeout=10.0):
        """
        Context manager для блокування файлу
        lock_type: fcntl.LOCK_EX (exclusive) або fcntl.LOCK_SH (shared)
        """
        lock_file = self.data_dir / ".api_keys.lock"
        lock_fd = None
        
        try:
            # Створюємо lock файл якщо не існує
            lock_fd = os.open(str(lock_file), os.O_CREAT | os.O_RDWR)
            
            # Спроба отримати блокування з таймаутом
            start_time = time.time()
            while True:
                try:
                    fcntl.flock(lock_fd, lock_type | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.time() - start_time > timeout:
                        raise FileLockTimeout(f"Не вдалося отримати блокування файлу за {timeout}s")
                    time.sleep(0.01)
            
            yield lock_fd
            
        finally:
            if lock_fd is not None:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    os.close(lock_fd)
                except Exception as e:
                    logger.warning(f"Помилка звільнення блокування: {e}")
    
    def _init_files(self):
        """Ініціалізує файли якщо вони не існують (thread-safe)"""
        # Створюємо master токен якщо не існує
        if not self.master_token_file.exists():
            # Використовуємо atomic write для master токена
            temp_fd, temp_path = tempfile.mkstemp(dir=self.data_dir, prefix='.master_token_', suffix='.tmp')
            try:
                master_token = secrets.token_urlsafe(32)
                os.write(temp_fd, master_token.encode('utf-8'))
                os.close(temp_fd)
                temp_fd = None
                
                # Atomic rename - якщо файл вже існує (інший worker створив), цей запис буде проігноровано
                try:
                    os.rename(temp_path, self.master_token_file)
                    logger.info(f"🔑 Створено master токен: {master_token}")
                    logger.info("📋 Збережіть цей токен! Він потрібен для доступу до адмін панелі")
                except FileExistsError:
                    # Інший процес вже створив файл - видаляємо temp файл
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
            finally:
                if temp_fd is not None:
                    os.close(temp_fd)
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        
        # Створюємо файл API ключів якщо не існує (з блокуванням)
        if not self.api_keys_file.exists():
            try:
                with self._file_lock(fcntl.LOCK_EX, timeout=5.0):
                    # Double-check після отримання блокування
                    if not self.api_keys_file.exists():
                        self._save_api_keys_unlocked({})
                        logger.info("📄 Створено файл API ключів")
            except FileLockTimeout:
                # Якщо не вдалося отримати блокування, файл вже створено іншим процесом
                pass
    
    def _load_api_keys_unlocked(self) -> Dict[str, Dict]:
        """Завантажує API ключі БЕЗ блокування (використовується всередині locked context)"""
        if not self.api_keys_file.exists():
            return {}
        
        try:
            with open(self.api_keys_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"JSON файл пошкоджено: {e}")
            backup_path = f"{self.api_keys_file}.corrupted.{int(time.time())}".replace('\\', '/')
            
            try:
                with open(self.api_keys_file, 'rb') as src:
                    with open(backup_path, 'wb') as dst:
                        dst.write(src.read())
                logger.error(f"Пошкоджений файл збережено як: {backup_path}")
            except Exception as backup_error:
                logger.error(f"Не вдалося створити backup: {backup_error}")
            
            raise RuntimeError(f"API keys file corrupted. Backup saved to: {backup_path}")
    
    def _load_api_keys(self) -> Dict[str, Dict]:
        """Завантажує API ключі з файлу з shared блокуванням"""
        if not self.api_keys_file.exists():
            return {}
        
        try:
            with self._file_lock(fcntl.LOCK_SH, timeout=10.0):
                with open(self.api_keys_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except json.JSONDecodeError as e:
            # JSON файл пошкоджено - створюємо backup та повертаємо помилку
            logger.error(f"JSON файл пошкоджено: {e}")
            backup_path = f"{self.api_keys_file}.corrupted.{int(time.time())}".replace('\\', '/')
            
            try:
                # Створюємо backup пошкодженого файлу
                with open(self.api_keys_file, 'rb') as src:
                    with open(backup_path, 'wb') as dst:
                        dst.write(src.read())
                logger.error(f"Пошкоджений файл збережено як: {backup_path}")
                logger.error("❌ КРИТИЧНА ПОМИЛКА: Файл API ключів пошкоджено!")
                logger.error("Відновіть дані з backup або видаліть пошкоджений файл для створення нового.")
            except Exception as backup_error:
                logger.error(f"Не вдалося створити backup: {backup_error}")
            
            raise RuntimeError(f"API keys file corrupted. Backup saved to: {backup_path}")
        except FileLockTimeout as e:
            logger.error(f"Не вдалося отримати блокування для читання: {e}")
            raise
        except Exception as e:
            logger.error(f"Помилка завантаження API ключів: {e}")
            raise
    
    def _save_api_keys_unlocked(self, api_keys: Dict[str, Dict]):
        """
        Зберігає API ключі у файл з атомарним записом (БЕЗ блокування).
        Використовується тільки коли блокування вже отримано ззовні.
        """
        temp_fd = None
        temp_path = None
        
        try:
            # Створюємо тимчасовий файл у тій же директорії (важливо для atomic rename)
            temp_fd, temp_path = tempfile.mkstemp(
                dir=self.data_dir,
                prefix='.api_keys_',
                suffix='.json.tmp'
            )
            
            # Записуємо дані у тимчасовий файл
            with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
                temp_fd = None  # fdopen взяв ownership
                json.dump(api_keys, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())  # Примусовий запис на диск
            
            # Атомарна заміна старого файлу новим
            os.replace(temp_path, self.api_keys_file)
            temp_path = None
            
            logger.info(f"Збережено {len(api_keys)} API ключів (atomic write)")
            
        except Exception as e:
            logger.error(f"Помилка збереження API ключів: {e}")
            raise
        finally:
            # Очищення тимчасових файлів у разі помилки
            if temp_fd is not None:
                try:
                    os.close(temp_fd)
                except Exception:
                    pass
            if temp_path is not None and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception as cleanup_error:
                    logger.warning(f"Не вдалося видалити тимчасовий файл {temp_path}: {cleanup_error}")
    
    def _save_api_keys(self, api_keys: Dict[str, Dict]):
        """Зберігає API ключі у файл з exclusive блокуванням та атомарним записом"""
        try:
            with self._file_lock(fcntl.LOCK_EX, timeout=10.0):
                self._save_api_keys_unlocked(api_keys)
        except FileLockTimeout as e:
            logger.error(f"Не вдалося отримати блокування для запису: {e}")
            raise
    
    def get_master_token(self) -> str:
        """Отримує master токен"""
        return self.master_token_file.read_text().strip()
    
    def verify_master_token(self, token: str) -> bool:
        """Перевіряє master токен"""
        return token == self.get_master_token()
    
    def generate_api_key(self, client_name: str) -> str:
        """Генерує новий API ключ (thread-safe з блокуванням)"""
        api_key = secrets.token_urlsafe(32)
        
        try:
            with self._file_lock(fcntl.LOCK_EX, timeout=10.0):
                # Завантажуємо під блокуванням
                api_keys = self._load_api_keys_unlocked()
                
                # Модифікуємо
                api_keys[api_key] = {
                    "client_name": client_name,
                    "created_at": datetime.now().isoformat(),
                    "active": True,
                    "usage_count": 0,
                    "last_used": None,
                    "total_requests": 0,
                    "successful_requests": 0,
                    "failed_requests": 0,
                    "total_processing_time": 0.0,
                    "notes": ""
                }
                
                # Зберігаємо під тим же блокуванням
                self._save_api_keys_unlocked(api_keys)
                
            logger.info(f"Створено новий API ключ для клієнта: {client_name}")
            return api_key
        except Exception as e:
            logger.error(f"Помилка генерації API ключа: {e}")
            raise
    
    def delete_api_key(self, api_key: str) -> bool:
        """Видаляє API ключ (thread-safe з блокуванням)"""
        try:
            with self._file_lock(fcntl.LOCK_EX, timeout=10.0):
                api_keys = self._load_api_keys_unlocked()
                
                if api_key in api_keys:
                    client_name = api_keys[api_key]["client_name"]
                    del api_keys[api_key]
                    self._save_api_keys_unlocked(api_keys)
                    logger.info(f"Видалено API ключ для клієнта: {client_name}")
                    return True
                return False
        except Exception as e:
            logger.error(f"Помилка видалення API ключа: {e}")
            raise
    
    def verify_api_key(self, api_key: str) -> bool:
        """Перевіряє API ключ"""
        api_keys = self._load_api_keys()
        return api_key in api_keys and api_keys[api_key].get("active", True)
    
    def get_api_key_info(self, api_key: str) -> Optional[Dict]:
        """Отримує інформацію про API ключ"""
        api_keys = self._load_api_keys()
        return api_keys.get(api_key)
    
    def list_api_keys(self) -> List[Dict]:
        """Отримує список всіх API ключів"""
        api_keys = self._load_api_keys()
        result = []
        for key, info in api_keys.items():
            result.append({
                "key": key,
                "client_name": info["client_name"],
                "created_at": info["created_at"],
                "active": info.get("active", True),
                "usage_count": info.get("usage_count", 0),
                "last_used": info.get("last_used"),
                "total_requests": info.get("total_requests", 0),
                "successful_requests": info.get("successful_requests", 0),
                "failed_requests": info.get("failed_requests", 0),
                "total_processing_time": round(info.get("total_processing_time", 0), 2),
                "average_processing_time": round(info.get("total_processing_time", 0) / max(info.get("total_requests", 1), 1), 2),
                "notes": info.get("notes", "")
            })
        return result
    
    def get_stats(self) -> Dict:
        """Отримує статистику API ключів"""
        api_keys = self._load_api_keys()
        active_count = sum(1 for info in api_keys.values() if info.get("active", True))
        total_requests = sum(info.get("total_requests", 0) for info in api_keys.values())
        total_processing_time = sum(info.get("total_processing_time", 0) for info in api_keys.values())
        
        return {
            "total_keys": len(api_keys),
            "active_keys": active_count,
            "inactive_keys": len(api_keys) - active_count,
            "total_requests": total_requests,
            "total_processing_time": round(total_processing_time, 2),
            "average_processing_time": round(total_processing_time / max(total_requests, 1), 2)
        }
    
    def log_api_usage(self, api_key: str, success: bool = True, processing_time: float = 0.0):
        """Логує використання API ключа (thread-safe з блокуванням)"""
        try:
            with self._file_lock(fcntl.LOCK_EX, timeout=10.0):
                api_keys = self._load_api_keys_unlocked()
                
                if api_key in api_keys:
                    api_keys[api_key]["usage_count"] = api_keys[api_key].get("usage_count", 0) + 1
                    api_keys[api_key]["last_used"] = datetime.now().isoformat()
                    api_keys[api_key]["total_requests"] = api_keys[api_key].get("total_requests", 0) + 1
                    
                    if success:
                        api_keys[api_key]["successful_requests"] = api_keys[api_key].get("successful_requests", 0) + 1
                    else:
                        api_keys[api_key]["failed_requests"] = api_keys[api_key].get("failed_requests", 0) + 1
                    
                    api_keys[api_key]["total_processing_time"] = api_keys[api_key].get("total_processing_time", 0) + processing_time
                    
                    self._save_api_keys_unlocked(api_keys)
        except Exception as e:
            # Не критична помилка - логування може провалитися
            logger.warning(f"Не вдалося залогувати використання API: {e}")
    
    def update_api_key_notes(self, api_key: str, notes: str) -> bool:
        """Оновлює нотатки для API ключа (thread-safe з блокуванням)"""
        try:
            with self._file_lock(fcntl.LOCK_EX, timeout=10.0):
                api_keys = self._load_api_keys_unlocked()
                
                if api_key in api_keys:
                    api_keys[api_key]["notes"] = notes
                    self._save_api_keys_unlocked(api_keys)
                    return True
                return False
        except Exception as e:
            logger.error(f"Помилка оновлення нотаток: {e}")
            raise
    
    def toggle_api_key_status(self, api_key: str) -> bool:
        """Перемикає статус API ключа (активний/неактивний) (thread-safe з блокуванням)"""
        try:
            with self._file_lock(fcntl.LOCK_EX, timeout=10.0):
                api_keys = self._load_api_keys_unlocked()
                
                if api_key in api_keys:
                    api_keys[api_key]["active"] = not api_keys[api_key].get("active", True)
                    self._save_api_keys_unlocked(api_keys)
                    return True
                return False
        except Exception as e:
            logger.error(f"Помилка зміни статусу ключа: {e}")
            raise
    
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
