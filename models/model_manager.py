"""
Глобальний менеджер моделей Whisper з контролем пам'яті.

Забезпечує:
- Єдиний інстанс моделі в пам'яті (singleton)
- Безпечне перемикання між розмірами моделей
- Блокування під час завантаження/вивантаження
- Перевірку доступної пам'яті перед завантаженням
"""

import os
import gc
import logging
import threading
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Приблизний розмір моделей у RAM (INT8 quantized, GB)
MODEL_MEMORY_REQUIREMENTS = {
    "tiny": 0.5,
    "base": 0.8,
    "small": 1.2,
    "medium": 2.5,
    "large": 4.5,
    "large-v2": 4.5,
    "large-v3": 4.5,
}

# Мінімальний запас пам'яті для безпечної роботи (GB)
# Зменшено для підтримки машин з обмеженою RAM
MEMORY_SAFETY_MARGIN_GB = 0.5

# Перевірка пам'яті: True = строга (відхиляти якщо недостатньо), False = м'яка (тільки warning)
STRICT_MEMORY_CHECK = os.environ.get("STRICT_MEMORY_CHECK", "false").lower() == "true"


@dataclass
class ModelInfo:
    """Інформація про завантажену модель"""
    model_size: str
    device: str
    compute_type: str
    loaded_at: float
    memory_usage_gb: float


class ModelManager:
    """
    Singleton менеджер для Whisper моделей.
    
    Гарантує, що в пам'яті знаходиться лише одна модель,
    і безпечно перемикає між розмірами.
    """
    
    _instance = None
    _lock = threading.RLock()  # Reentrant lock для вкладених викликів
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._model = None
        self._model_info: Optional[ModelInfo] = None
        self._loading = False
        self._initialized = True
        logger.info("🔧 ModelManager ініціалізовано (singleton)")
    
    @property
    def current_model(self):
        """Поточна завантажена модель"""
        return self._model
    
    @property
    def current_model_size(self) -> Optional[str]:
        """Розмір поточної моделі"""
        return self._model_info.model_size if self._model_info else None
    
    @property
    def is_loading(self) -> bool:
        """Чи відбувається зараз завантаження"""
        return self._loading
    
    def get_available_memory_gb(self) -> float:
        """Повертає доступну пам'ять у GB"""
        try:
            import psutil
            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024 ** 3)
            return available_gb
        except ImportError:
            logger.warning("psutil недоступний, припускаємо 8GB вільної пам'яті")
            return 8.0
    
    def get_total_memory_gb(self) -> float:
        """Повертає загальну пам'ять у GB"""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return mem.total / (1024 ** 3)
        except ImportError:
            return 16.0
    
    def can_load_model(self, model_size: str, strict: bool = None) -> tuple[bool, str]:
        """
        Перевіряє чи можна безпечно завантажити модель.
        
        Args:
            model_size: Розмір моделі
            strict: True = блокувати якщо недостатньо, False = тільки warning, None = за налаштуванням
        
        Returns:
            (can_load, reason)
        """
        if strict is None:
            strict = STRICT_MEMORY_CHECK
            
        required_memory = MODEL_MEMORY_REQUIREMENTS.get(model_size, 2.0)
        available_memory = self.get_available_memory_gb()
        total_memory = self.get_total_memory_gb()
        
        # Якщо модель того ж розміру вже завантажена - OK
        if self._model and self._model_info and self._model_info.model_size == model_size:
            return True, "Model already loaded"
        
        # Враховуємо, що стара модель буде вивантажена
        current_model_memory = 0
        if self._model_info:
            current_model_memory = MODEL_MEMORY_REQUIREMENTS.get(
                self._model_info.model_size, 0
            )
        
        # Доступна пам'ять після вивантаження старої моделі
        effective_available = available_memory + current_model_memory
        needed = required_memory + MEMORY_SAFETY_MARGIN_GB
        
        if effective_available < needed:
            reason = (
                f"Insufficient memory: need {needed:.1f}GB, "
                f"available {effective_available:.1f}GB (total {total_memory:.1f}GB)"
            )
            if strict:
                return False, reason
            else:
                # М'яка перевірка - тільки попередження, але дозволяємо спробу
                logger.warning(f"⚠️ {reason} - attempting anyway (STRICT_MEMORY_CHECK=false)")
                return True, f"Warning: {reason}"
        
        return True, "OK"
    
    def unload_model(self) -> bool:
        """
        Вивантажує поточну модель з пам'яті.
        
        Returns:
            True якщо модель була вивантажена, False якщо моделі не було
        """
        with self._lock:
            if self._model is None:
                logger.debug("Немає моделі для вивантаження")
                return False
            
            old_size = self._model_info.model_size if self._model_info else "unknown"
            logger.info(f"🗑️ Вивантаження моделі {old_size}...")
            
            # Видаляємо посилання на модель
            self._model = None
            self._model_info = None
            
            # Агресивне очищення пам'яті
            for _ in range(5):
                gc.collect()
            
            # Спроба очистити CUDA кеш якщо використовується
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception:
                pass
            
            logger.info(f"✅ Модель {old_size} вивантажена, RAM: {self.get_available_memory_gb():.1f}GB вільно")
            return True
    
    def load_model(self, model_size: str, device: str = "cpu", force: bool = False) -> Optional[Any]:
        """
        Завантажує модель Whisper, вивантажуючи попередню якщо потрібно.
        
        Args:
            model_size: Розмір моделі (tiny, base, small, medium, large)
            device: Пристрій (cpu/cuda)
            force: Примусово перезавантажити навіть якщо та сама модель
        
        Returns:
            Завантажена модель або None при помилці
        """
        with self._lock:
            # Якщо модель того ж розміру вже завантажена - повертаємо її
            if not force and self._model and self._model_info:
                if self._model_info.model_size == model_size:
                    logger.debug(f"Модель {model_size} вже завантажена")
                    return self._model
            
            # Перевіряємо чи можна завантажити
            can_load, reason = self.can_load_model(model_size)
            if not can_load:
                logger.error(f"❌ Неможливо завантажити модель {model_size}: {reason}")
                raise MemoryError(f"Cannot load model {model_size}: {reason}")
            
            self._loading = True
            
            try:
                # Вивантажуємо стару модель
                if self._model is not None:
                    old_size = self._model_info.model_size if self._model_info else "unknown"
                    logger.info(f"🔄 Перемикання з {old_size} на {model_size}")
                    self.unload_model()
                
                # Завантажуємо нову модель
                logger.info(f"📥 Завантаження моделі {model_size}...")
                
                from faster_whisper import WhisperModel
                from .config import MODELS_DIR, CPU_COMPUTE_TYPE, GPU_COMPUTE_TYPE
                
                compute_type = CPU_COMPUTE_TYPE if device == "cpu" else GPU_COMPUTE_TYPE
                cpu_threads = min(8, os.cpu_count() or 4)
                
                import time
                start_time = time.time()
                
                self._model = WhisperModel(
                    model_size,
                    device=device,
                    compute_type=compute_type,
                    cpu_threads=cpu_threads,
                    num_workers=2 if device == "cpu" else 1,
                    download_root=str(MODELS_DIR)
                )
                
                load_time = time.time() - start_time
                memory_used = MODEL_MEMORY_REQUIREMENTS.get(model_size, 2.0)
                
                self._model_info = ModelInfo(
                    model_size=model_size,
                    device=device,
                    compute_type=compute_type,
                    loaded_at=time.time(),
                    memory_usage_gb=memory_used
                )
                
                logger.info(
                    f"✅ Модель {model_size} завантажена за {load_time:.1f}с "
                    f"(~{memory_used:.1f}GB RAM, вільно: {self.get_available_memory_gb():.1f}GB)"
                )
                
                return self._model
                
            except MemoryError:
                raise
            except Exception as e:
                logger.error(f"❌ Помилка завантаження моделі {model_size}: {e}")
                self._model = None
                self._model_info = None
                raise
            finally:
                self._loading = False
    
    def get_status(self) -> Dict[str, Any]:
        """Повертає статус менеджера моделей"""
        return {
            "model_loaded": self._model is not None,
            "current_model_size": self._model_info.model_size if self._model_info else None,
            "current_device": self._model_info.device if self._model_info else None,
            "is_loading": self._loading,
            "available_memory_gb": round(self.get_available_memory_gb(), 2),
            "total_memory_gb": round(self.get_total_memory_gb(), 2),
            "model_memory_requirements": MODEL_MEMORY_REQUIREMENTS,
        }


# Глобальний singleton інстанс
model_manager = ModelManager()
