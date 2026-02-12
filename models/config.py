"""
Конфігурація та константи для моделей
"""

import os
from pathlib import Path

# Директорія для локальних моделей
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Налаштування логування
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Імпорт LanguageTool для орфографічної корекції
try:
    import language_tool_python
    LANGUAGE_TOOL_AVAILABLE = True
    logger.info("LanguageTool доступний для орфографічної корекції")
except ImportError:
    LANGUAGE_TOOL_AVAILABLE = False
    logger.warning("LanguageTool недоступний - орфографічна корекція вимкнена")

# Константи для оптимізації (налаштовано для CPU-only сервера 14GB RAM + 8 CPU)
DEFAULT_CHUNK_SIZE = 45  # секунд (ще більше зменшено для економії пам'яті)
MAX_WORKERS = 3  # Ще більше зменшено для стабільності (37.5% ядер)
CACHE_MAX_SIZE = 2  # Мінімальний кеш для економії пам'яті

# CPU-only оптимізації (ви завжди використовуєте CPU)
CPU_CHUNK_SIZE = 45  # секунд (ще більше зменшено для економії пам'яті)
CPU_MAX_WORKERS = 3  # Ще більше зменшено для стабільності
CPU_CACHE_MAX_SIZE = 2  # Мінімальний кеш для економії пам'яті

# Додаткові оптимізації для сервера 14GB RAM
MAX_FILE_SIZE_MB = 500  # Максимальний розмір файлу (500MB) - зменшено для безпеки
MAX_UPLOAD_SIZE_MB = 300  # Максимальний розмір для завантаження (300MB)
MAX_AUDIO_DURATION_MINUTES = 80  # Максимальна тривалість аудіо (60 хвилин)
MEMORY_PRESSURE_THRESHOLD = 75.0  # Поріг тиску на пам'ять (75%)

# GPU оптимізації (не використовуються, але залишені для сумісності)
GPU_CHUNK_SIZE = 150  # секунд (більші чанки для GPU)
GPU_MAX_WORKERS = 6  # Більше процесів для GPU
GPU_CACHE_MAX_SIZE = 10  # Більший кеш для GPU

# Налаштування діаризації (оптимізовано для CPU-only)
ENABLE_DIARIZATION = True  # Можна відключити для економії ресурсів
DIARIZATION_MAX_WORKERS = 4  # Зменшено для економії пам'яті (threads замість processes)

# Оптимізовані параметри для швидкості
SPEED_OPTIMIZED_BEAM_SIZE = 1  # Завжди 1 для максимальної швидкості
SPEED_OPTIMIZED_VAD = True     # Увімкнено для кращого виявлення початку мовлення
SPEED_OPTIMIZED_CHUNK_SIZES = {
    'short': 60,    # < 5 хв (оптимізовано для 8 CPU)
    'medium': 120,  # 5-30 хв (оптимізовано для 8 CPU)
    'long': 180     # > 30 хв (оптимізовано для 8 CPU)
}

# Доступні моделі для різних мов
# Для української мови: tiny, base, small, medium, large
# Для англійської мови: також distil-small.en, distil-medium.en
SUPPORTED_MODELS = ['tiny', 'base', 'small', 'medium', 'large']

# Quantized моделі (рекомендовані ChatGPT для кращої продуктивності)
# faster-whisper автоматично використовує quantized версії при compute_type="int8"
QUANTIZED_MODELS = {
    'tiny': 'tiny',    # tiny + int8 = quantized
    'base': 'base',    # base + int8 = quantized  
    'small': 'small',  # small + int8 = quantized 
    'medium': 'medium', # medium + int8 = quantized
    'large': 'large'   # large + int8 = quantized
}

# Налаштування compute_type для різних пристроїв
CPU_COMPUTE_TYPE = "int8"  # Quantized для CPU (швидше та економніше)
GPU_COMPUTE_TYPE = "float16"  # Float16 для GPU
