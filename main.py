from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends, Header, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl
import tempfile
import os
import httpx
import time
import uuid
import json
import asyncio
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Any
import logging
from models import LocalTranscriptionService
from models.config import MAX_UPLOAD_SIZE_MB
from middleware import verify_api_key, verify_master_token, verify_master_token_from_query

# Ініціалізуємо БД перед імпортом api_key_manager
from app.db.init_db import init_db
init_db()

from api_auth import api_key_manager
from app.db.session import get_db_session
from app.db.repositories.tasks import TaskRepository
from app.db.models import TaskStatus as TaskStatusEnum

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Ukrainian Audio Transcription API (Local Models)",
    description="API for transcribing Ukrainian audio/video with local speaker-aware models",
    version="1.0.0"
)

# Додаємо CORS middleware для правильного кодування
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Глобальні змінні для сервісу транскрипції та черги
transcription_service = None
task_queue = None
tasks = {}  # task_id -> TaskStatus
executor = None
worker_tasks = []  # Зберігаємо посилання на воркер-таски



class TranscriptionRequest(BaseModel):
    url: Optional[HttpUrl] = None
    language: str = "uk"  # Українська мова за замовчуванням
    model_size: str = "large"  # Розмір моделі Whisper
    enhance_audio: bool = True  # Попередня обробка аудіо
    use_diarization: bool = False  # Використовувати діаризацію

class TranscriptionResponse(BaseModel):
    text: str
    segments: List[Dict[str, Any]]
    speakers: Optional[List[Dict[str, Any]]] = None
    duration: float
    language: str
    diarization_type: Optional[str] = None

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

class GenerateKeyRequest(BaseModel):
    client_name: str

class GenerateKeyResponse(BaseModel):
    api_key: str
    client_name: str
    created_at: str

class DeleteKeyRequest(BaseModel):
    api_key: str

class UpdateKeyNotesRequest(BaseModel):
    api_key: str
    notes: str

class ToggleKeyStatusRequest(BaseModel):
    api_key: str

class APIKeyInfo(BaseModel):
    key: str
    client_name: str
    created_at: str
    active: bool
    usage_count: int
    last_used: Optional[str]
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_processing_time: float
    average_processing_time: float
    notes: str

class TaskStatus(BaseModel):
    task_id: str
    status: str  # queued, processing, completed, failed
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    progress: int  # 0-100
    result: Optional[Dict[str, Any]]
    error: Optional[str]
    file_name: str
    language: str
    model_size: str
    use_diarization: bool
    api_key: Optional[str] = None

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str

@app.on_event("startup")
async def load_models():
    """Завантаження локальних моделей при запуску сервера"""
    global transcription_service, task_queue, executor, worker_tasks
    
    try:
        logger.info("Ініціалізація локального сервісу транскрипції...")
        transcription_service = LocalTranscriptionService()
        
        if transcription_service.load_models():
            logger.info("Локальні моделі завантажені успішно")
        else:
            logger.error("Не вдалося завантажити моделі")
            raise RuntimeError("Моделі не завантажені")
        
        # Виводимо інформацію про master токен
        api_key_manager.print_startup_info()
        
        # Ініціалізуємо чергу та executor (оптимізовано для 8 CPU + 14GB RAM)
        task_queue = asyncio.Queue(maxsize=25)  # Збільшено розмір черги до 25
        executor = ThreadPoolExecutor(max_workers=3)  # Ще більше зменшено для стабільності
        
        # Запускаємо воркери для обробки черги
        logger.info("Запуск воркерів для обробки черги транскрипції...")
        for i in range(3):  # Запускаємо 3 воркери (мінімізовано для стабільності)
            worker_task = asyncio.create_task(worker())
            worker_tasks.append(worker_task)
            logger.info(f"Воркер {i+1} запущено")
        
    except Exception as e:
        logger.error(f"Помилка при завантаженні моделей: {e}")
        raise RuntimeError(f"Не вдалося ініціалізувати сервіс: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Очищення ресурсів при завершенні сервера"""
    global executor, worker_tasks
    
    logger.info("Завершення роботи сервера...")
    
    # Скасовуємо всі воркер-таски
    for worker_task in worker_tasks:
        worker_task.cancel()
    
    # Чекаємо завершення воркерів
    if worker_tasks:
        await asyncio.gather(*worker_tasks, return_exceptions=True)
    
    # Закриваємо executor
    if executor:
        executor.shutdown(wait=True)
    
    logger.info("Сервер завершив роботу")

async def download_file_from_url(url: str) -> str:
    """Завантаження файлу з URL"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(str(url))
            response.raise_for_status()
            
            # Створюємо тимчасовий файл з context manager
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as temp_file:
                temp_file.write(response.content)
                temp_file.flush()
                return temp_file.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File download failed: {str(e)}")

def save_task_status(task_id: str, task_status: TaskStatus, raise_on_error: bool = False):
    """
    Збереження статусу задачі в SQLite.
    
    Args:
        task_id: ID задачі
        task_status: Об'єкт статусу задачі
        raise_on_error: If True, re-raise exceptions after logging (Fix 7.14)
                        If False, only log errors (backward compatible)
    
    Raises:
        Exception: If raise_on_error=True and database operation fails
    """
    try:
        with get_db_session() as session:
            repo = TaskRepository(session)
            existing = repo.get_by_id(task_id)
            
            if existing:
                # Оновлюємо існуючу задачу
                logger.info(f"Оновлення існуючої задачі {task_id}: status={task_status.status}")
                repo.update_status(
                    task_id=task_id,
                    status=task_status.status,
                    error_message=task_status.error
                )
                
                if task_status.status == "completed" and task_status.result:
                    duration = task_status.result.get('duration', 0)
                    # Зберігаємо result як JSON
                    import json
                    result_json = json.dumps(task_status.result, ensure_ascii=False)
                    repo.mark_completed(task_id, duration_sec=duration, result_json=result_json)
                elif task_status.status == "failed" and task_status.error:
                    repo.mark_failed(task_id, task_status.error)
            else:
                # Створюємо нову задачу
                api_key = task_status.api_key if task_status.api_key else 'unknown'
                logger.info(f"Створення нової задачі {task_id}: api_key={api_key}, file={task_status.file_name}")
                repo.create(
                    task_id=task_id,
                    api_key=api_key,
                    filename=task_status.file_name,
                    model_size=task_status.model_size,
                    has_diarization=task_status.use_diarization,
                    status=task_status.status
                )
                logger.info(f"Задача {task_id} успішно створена в БД")
    except Exception as e:
        logger.error(f"Помилка збереження статусу задачі {task_id}: {e}", exc_info=True)
        # Fix 7.14: Re-raise if caller needs to handle the failure
        if raise_on_error:
            raise

def clean_old_tasks(all_tasks: dict, max_age_days: int = 7) -> dict:
    """Очищення старих задач з файлу"""
    from datetime import datetime, timedelta
    
    try:
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        cleaned_tasks = {}
        removed_count = 0
        
        for task_id, task_data in all_tasks.items():
            try:
                # Парсимо дату створення задачі
                created_at_str = task_data.get('created_at', '')
                if created_at_str:
                    created_at = datetime.strptime(created_at_str, "%Y-%m-%d %H:%M:%S")
                    
                    # Якщо задача старша за cutoff_date, видаляємо її
                    if created_at < cutoff_date:
                        removed_count += 1
                        continue
                
                # Зберігаємо задачу
                cleaned_tasks[task_id] = task_data
                
            except Exception as e:
                logger.warning(f"Помилка обробки задачі {task_id}: {e}")
                # Якщо не можемо розпарсити дату, зберігаємо задачу
                cleaned_tasks[task_id] = task_data
        
        if removed_count > 0:
            logger.info(f"Автоматично очищено {removed_count} старих задач (старші {max_age_days} днів)")
        
        return cleaned_tasks
        
    except Exception as e:
        logger.error(f"Помилка очищення старих задач: {e}")
        return all_tasks  # Повертаємо оригінальний словник при помилці

def load_task_status(task_id: str) -> Optional[TaskStatus]:
    """Завантаження статусу задачі з SQLite"""
    try:
        import json
        with get_db_session() as session:
            repo = TaskRepository(session)
            task = repo.get_by_id(task_id)
            
            if not task:
                return None
            
            # Парсимо result_json якщо є
            result = None
            if task.result_json:
                try:
                    result = json.loads(task.result_json)
                except:
                    pass
            
            # Конвертуємо Task model в TaskStatus
            return TaskStatus(
                task_id=task.id,
                status=task.status,
                created_at=task.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                started_at=task.started_at.strftime("%Y-%m-%d %H:%M:%S") if task.started_at else None,  # Fix 7.13
                completed_at=task.completed_at.strftime("%Y-%m-%d %H:%M:%S") if task.completed_at else None,
                progress=100 if task.status == "completed" else 0,
                result=result,
                error=task.error_message,
                file_name=task.filename,
                language="uk",  # TODO: додати в модель
                model_size=task.model_size,
                use_diarization=task.has_diarization,
                api_key=task.api_key
            )
    except Exception as e:
        logger.error(f"Помилка завантаження статусу задачі {task_id}: {e}")
    return None

def process_transcription_task_sync(task_id: str, file_path: str, language: str, model_size: str, use_diarization: bool, api_key: str):
    """Синхронна обробка задачі транскрипції"""
    try:
        # Оновлюємо статус на "processing"
        task_status = tasks[task_id]
        task_status.status = "processing"
        task_status.started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        task_status.progress = 10
        save_task_status(task_id, task_status)
        
        logger.info(f"Початок обробки задачі {task_id}")
        
        # Виконуємо транскрипцію синхронно з обробкою помилок пам'яті
        try:
            if use_diarization:
                result = transcription_service.transcribe_with_diarization(file_path, language, model_size)
            else:
                result = transcription_service.transcribe_simple(file_path, language, model_size)
        except MemoryError as mem_err:
            logger.error(f"⚠️ Недостатньо пам'яті для обробки задачі {task_id}: {mem_err}")
            raise MemoryError(f"Недостатньо оперативної пам'яті для обробки файлу. Спробуйте менший файл або меншу модель (tiny/base/small).")
        
        # Оновлюємо статус на "completed"
        task_status.status = "completed"
        task_status.completed_at = time.strftime("%Y-%m-%d %H:%M:%S")
        task_status.progress = 100
        task_status.result = result
        save_task_status(task_id, task_status, raise_on_error=True)  # Fix 7.14: Must succeed before file cleanup
        
        # Логуємо успішне використання API
        processing_time = time.time() - time.mktime(time.strptime(task_status.started_at, "%Y-%m-%d %H:%M:%S"))
        api_key_manager.log_api_usage(api_key, success=True, processing_time=processing_time)
        
        logger.info(f"Задача {task_id} завершена успішно")
        
        # CRITICAL: Видаляємо файл ТІЛЬКИ після успішного збереження в БД
        if os.path.exists(file_path):
            try:
                os.unlink(file_path)
                logger.info(f"Тимчасовий файл видалено: {file_path}")
            except Exception as e:
                logger.warning(f"Не вдалося видалити тимчасовий файл: {e}")
        
        # Очищуємо кеш аудіо після успішної обробки
        try:
            transcription_service.clear_audio_cache()
            import gc
            gc.collect()
            logger.debug(f"Пам'ять очищена після задачі {task_id}")
        except Exception as e:
            logger.warning(f"Помилка очищення пам'яті: {e}")
        
    except Exception as e:
        # Оновлюємо статус на "failed"
        task_status = tasks[task_id]
        task_status.status = "failed"
        task_status.completed_at = time.strftime("%Y-%m-%d %H:%M:%S")
        task_status.error = str(e)
        
        # Спроба зберегти failed статус (можуть бути повторні помилки БД)
        try:
            save_task_status(task_id, task_status)
        except Exception as db_error:
            logger.error(f"Критична помилка збереження failed статусу для {task_id}: {db_error}")
        
        # Логуємо невдале використання API
        try:
            processing_time = time.time() - time.mktime(time.strptime(task_status.started_at, "%Y-%m-%d %H:%M:%S"))
            api_key_manager.log_api_usage(api_key, success=False, processing_time=processing_time)
        except:
            pass  # Не блокуємо cleanup якщо started_at відсутній
        
        logger.error(f"Помилка обробки задачі {task_id}: {e}")
        
        # Спеціальна обробка MemoryError
        if isinstance(e, MemoryError):
            logger.error(f"⚠️ КРИТИЧНО: Недостатньо пам'яті для обробки {task_id}")
            task_status.error = "Недостатньо пам'яті для обробки файлу. Спробуйте менший файл або меншу модель."
        
        # Видаляємо файл навіть при помилці (файл вже непотрібний)
        if os.path.exists(file_path):
            try:
                os.unlink(file_path)
                logger.info(f"Тимчасовий файл видалено після помилки: {file_path}")
            except Exception as cleanup_error:
                logger.warning(f"Не вдалося видалити тимчасовий файл: {cleanup_error}")
        
        # Очищуємо кеш навіть при помилці
        try:
            transcription_service.clear_audio_cache()
            import gc
            gc.collect()
            logger.debug(f"Пам'ять очищена після помилки в задачі {task_id}")
        except Exception as cleanup_e:
            logger.warning(f"Помилка очищення пам'яті після помилки: {cleanup_e}")

async def worker():
    """Воркер для обробки задач з черги (оптимізований для CPU)"""
    worker_id = id(asyncio.current_task())
    logger.info(f"Воркер {worker_id} запущено")
    
    while True:
        try:
            # Очікуємо задачу з черги з таймаутом
            try:
                task_data = await asyncio.wait_for(task_queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                # Якщо немає задач, очищуємо пам'ять та чекаємо далі
                import gc
                gc.collect()
                continue
            
            logger.info(f"Воркер {worker_id} отримав задачу {task_data['task_id']}")
            
            # Обробляємо задачу в окремому потоці з обмеженням часу
            try:
                await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        executor,
                        process_transcription_task_sync,
                        task_data['task_id'],
                        task_data['file_path'],
                        task_data['language'],
                        task_data['model_size'],
                        task_data['use_diarization'],
                        task_data['api_key']
                    ),
                    timeout=7200.0  # 2 години максимум
                )
                
                # Позначаємо задачу як виконану
                task_queue.task_done()
                logger.info(f"Воркер {worker_id} завершив задачу {task_data['task_id']}")
                
                # Очищення пам'яті після задачі
                import gc
                gc.collect()
                
                # Очищення пам'яті після задачі
                import gc
                gc.collect()
                
            except asyncio.TimeoutError:
                task_id = task_data['task_id']
                logger.error(f"Воркер {worker_id}: задача {task_id} перевищила час виконання (2 години)")
                
                # CRITICAL: Оновлюємо статус в БД - задача failed через timeout
                try:
                    with get_db_session() as session:
                        repo = TaskRepository(session)
                        repo.mark_failed(task_id, "Перевищено час обробки (2 години)")
                    logger.info(f"Задача {task_id} позначена як failed через timeout в БД")
                except Exception as db_error:
                    logger.error(f"Помилка оновлення БД для timeout задачі {task_id}: {db_error}")
                
                # Оновлюємо memory cache якщо задача там є
                if task_id in tasks:
                    tasks[task_id].status = "failed"
                    tasks[task_id].error = "Перевищено час обробки (2 години)"
                    tasks[task_id].completed_at = time.strftime("%Y-%m-%d %H:%M:%S")
                
                task_queue.task_done()
            
            # Очищуємо пам'ять після кожної задачі
            import gc
            gc.collect()
            logger.debug(f"Воркер {worker_id} очистив пам'ять")
            
        except asyncio.CancelledError:
            logger.info(f"Воркер {worker_id} отримав сигнал завершення")
            break
        except Exception as e:
            logger.error(f"Помилка воркера {worker_id}: {e}")
            await asyncio.sleep(2)  # Більша пауза перед наступною спробою

# Функції транскрипції тепер використовують локальний сервіс

@app.post("/transcribe", response_model=TaskResponse)
async def transcribe_audio_file(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    language: str = Form("uk"),
    model_size: str = Form("large"),
    use_diarization: bool = Form(False),
    api_key: str = Depends(verify_api_key)
):
    """
    Транскрипція аудіо/відео файлу з визначенням дикторів (асинхронна через чергу)
    
    Параметри:
    - file: Завантажений файл (аудіо або відео)
    - url: URL посилання на файл
    - language: Мова транскрипції (за замовчуванням 'uk' для української)
    - model_size: Розмір моделі Whisper (tiny, base, small, medium, large, auto)
    - use_diarization: Використовувати діаризацію Оператор/Клієнт (True/False)
    
    Повертає task_id для відстеження статусу через /task/{task_id}
    """
    
    if not file and not url:
        raise HTTPException(status_code=400, detail="Either a file or URL must be provided")
    
    if file and url:
        raise HTTPException(status_code=400, detail="Provide either a file or a URL, not both")
    
    # Валідація розміру моделі
    if model_size not in ["tiny", "base", "small", "medium", "large", "auto"]:
        raise HTTPException(status_code=400, detail="Model size must be one of: tiny, base, small, medium, large, auto")
    
    # Перевірка доступності пам'яті для запитаної моделі
    if model_size != "auto":
        try:
            from models.model_manager import model_manager
            can_load, reason = model_manager.can_load_model(model_size)
            if not can_load:
                raise HTTPException(
                    status_code=507,
                    detail=f"Insufficient memory for model '{model_size}': {reason}. Try a smaller model or wait for current tasks to complete."
                )
        except ImportError:
            pass  # ModelManager недоступний, продовжуємо без перевірки
    
    # Генеруємо унікальний ID задачі
    task_id = str(uuid.uuid4())
    
    temp_file_path = None
    
    try:
        # Обробка файлу або URL
        if file:
            # Перевірка розміру файлу ПЕРЕД завантаженням
            file_size_mb = 0
            if hasattr(file, 'size') and file.size:
                file_size_mb = file.size / (1024 * 1024)
            elif hasattr(file, 'spool_max_size'):
                # Отримуємо розмір через content-length
                file_size_mb = file.spool_max_size / (1024 * 1024) if file.spool_max_size else 0
            
            if file_size_mb > MAX_UPLOAD_SIZE_MB:
                raise HTTPException(
                    status_code=413,
                    detail=f"Файл завеликий: {file_size_mb:.1f}MB. Максимум {MAX_UPLOAD_SIZE_MB}MB"
                )
            
            # Збереження завантаженого файлу з context manager та обробкою помилок
            try:
                content = await file.read()
            except MemoryError:
                raise HTTPException(
                    status_code=507,
                    detail=f"Недостатньо пам'яті для завантаження файлу {file_size_mb:.1f}MB"
                )
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as temp_file:
                temp_file.write(content)
                temp_file.flush()
                temp_file_path = temp_file.name
            
            # Очищуємо content з пам'яті
            del content
            import gc
            gc.collect()
            
            file_name = file.filename
            
        elif url:
            # Завантаження файлу з URL
            temp_file_path = await download_file_from_url(url)
            file_name = url.split('/')[-1] if '/' in url else "downloaded_file"
        
        # Створюємо статус задачі
        task_status = TaskStatus(
            task_id=task_id,
            status="queued",
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            started_at=None,
            completed_at=None,
            progress=0,
            result=None,
            error=None,
            file_name=file_name,
            language=language,
            model_size=model_size,
            use_diarization=use_diarization,
            api_key=api_key
        )
        
        # Зберігаємо статус задачі
        tasks[task_id] = task_status
        save_task_status(task_id, task_status, raise_on_error=True)  # Fix 7.14: Must succeed or return error
        
        # Перевіряємо розмір черги перед додаванням задачі
        if task_queue.qsize() >= 20:  # Якщо черга майже повна (залишаємо 5 місць)
            raise HTTPException(
                status_code=503, 
                detail="Server overloaded. Please try again later."
            )
        
        # Додаємо задачу в чергу
        await task_queue.put({
            'task_id': task_id,
            'file_path': temp_file_path,
            'language': language,
            'model_size': model_size,
            'use_diarization': use_diarization,
            'api_key': api_key
        })
        
        logger.info(f"Задача {task_id} додана в чергу для файлу {file_name}")
        
        return TaskResponse(
            task_id=task_id,
            status="queued",
            message=f"File {file_name} queued for processing. Use /task/{task_id} to track progress."
        )
        
    except HTTPException as http_exc:
        # CRITICAL FIX 7.5: Cleanup temp file if task creation failed
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.info(f"Очищено temp файл після помилки: {temp_file_path}")
            except Exception as cleanup_error:
                logger.warning(f"Не вдалося видалити temp файл: {cleanup_error}")
        raise http_exc
    except Exception as e:
        # CRITICAL FIX 7.5: Cleanup temp file on unexpected error
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.info(f"Очищено temp файл після помилки: {temp_file_path}")
            except Exception as cleanup_error:
                logger.warning(f"Не вдалося видалити temp файл: {cleanup_error}")
        logger.error(f"Неочікувана помилка: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/transcribe-with-diarization", response_model=TranscriptionResponse)
async def transcribe_with_diarization(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    language: str = Form("uk"),
    model_size: str = Form("large"),
    api_key: str = Depends(verify_api_key)
):
    """
    Транскрипція аудіо/відео файлу з діаризацією Оператор/Клієнт
    
    Параметри:
    - file: Завантажений файл (аудіо або відео)
    - url: URL посилання на файл
    - language: Мова транскрипції (за замовчуванням 'uk' для української)
    - model_size: Розмір моделі Whisper (tiny, base, small, medium, large, auto)
    """
    
    if not file and not url:
        raise HTTPException(status_code=400, detail="Either a file or URL must be provided")
    
    if file and url:
        raise HTTPException(status_code=400, detail="Provide either a file or a URL, not both")
    
    # Валідація розміру моделі
    if model_size not in ["tiny", "base", "small", "medium", "large", "auto"]:
        raise HTTPException(status_code=400, detail="Model size must be one of: tiny, base, small, medium, large, auto")
    
    temp_file_path = None
    
    try:
        # Обробка файлу або URL
        if file:
            # Перевірка розміру файлу ПЕРЕД завантаженням
            file_size_mb = 0
            if hasattr(file, 'size') and file.size:
                file_size_mb = file.size / (1024 * 1024)
            
            if file_size_mb > MAX_UPLOAD_SIZE_MB:
                raise HTTPException(
                    status_code=413,
                    detail=f"Файл завеликий: {file_size_mb:.1f}MB. Максимум {MAX_UPLOAD_SIZE_MB}MB"
                )
            
            # Збереження завантаженого файлу з обробкою помилок пам'яті
            try:
                content = await file.read()
            except MemoryError:
                raise HTTPException(
                    status_code=507,
                    detail=f"Недостатньо пам'яті для завантаження файлу {file_size_mb:.1f}MB"
                )
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as temp_file:
                temp_file.write(content)
                temp_file.flush()
                temp_file_path = temp_file.name
            
            # Очищуємо content з пам'яті
            del content
            import gc
            gc.collect()
            
        elif url:
            # Завантаження файлу з URL
            temp_file_path = await download_file_from_url(url)
        
        # Транскрипція з діаризацією
        logger.info(f"📝 Параметри запиту: model_size={model_size}, language={language}")
        logger.info(f"Початок транскрипції з діаризацією файлу: {temp_file_path}")
        start_time = time.time()
        
        try:
            processed_result = transcription_service.transcribe_with_diarization(temp_file_path, language, model_size)
            
            # Логуємо успішне використання
            processing_time = time.time() - start_time
            api_key_manager.log_api_usage(api_key, success=True, processing_time=processing_time)
            
            logger.info("Транскрипція з діаризацією завершена успішно")
            return TranscriptionResponse(**processed_result)
            
        except Exception as e:
            # Логуємо невдале використання
            processing_time = time.time() - start_time
            api_key_manager.log_api_usage(api_key, success=False, processing_time=processing_time)
            raise e
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Неочікувана помилка: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    finally:
        # Очищення тимчасових файлів
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.info(f"Тимчасовий файл видалено: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Не вдалося видалити тимчасовий файл: {e}")

@app.get("/health")
async def health_check():
    """Перевірка стану сервісу"""
    queue_size = task_queue.qsize() if task_queue else 0
    active_tasks = len([t for t in tasks.values() if t.status == "processing"])
    max_workers = executor._max_workers if executor else 0
    
    return {
        "status": "healthy",
        "models_loaded": transcription_service is not None and transcription_service.models_loaded,
        "whisper_loaded": transcription_service is not None and transcription_service.whisper_model.model is not None,
        "queue_size": queue_size,
        "active_tasks": active_tasks,
        "max_workers": max_workers,
        "worker_tasks": len(worker_tasks)
    }

@app.get("/task/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """Отримання статусу задачі транскрипції"""
    # Спочатку перевіряємо в пам'яті
    if task_id in tasks:
        return tasks[task_id]
    
    # Якщо не знайдено в пам'яті, завантажуємо з файлу
    task_status = load_task_status(task_id)
    if task_status:
        return task_status
    
    raise HTTPException(status_code=404, detail="Task not found")

@app.get("/tasks")
async def list_tasks(limit: int = 50, status: Optional[str] = None):
    """Отримання списку задач з фільтрацією"""
    try:
        with get_db_session() as session:
            repo = TaskRepository(session)
            
            # Фільтруємо за статусом якщо вказано
            if status:
                db_tasks = repo.get_by_status(status, limit=limit)
            else:
                # Отримуємо всі задачі з сортуванням за датою
                from sqlmodel import select
                from app.db.models import Task
                statement = select(Task).order_by(Task.created_at.desc()).limit(limit)
                db_tasks = list(session.exec(statement).all())
            
            # Конвертуємо в TaskStatus
            tasks_list = []
            for task in db_tasks:
                result = None
                if task.result_json:
                    try:
                        result = json.loads(task.result_json)
                    except:
                        pass
                
                tasks_list.append(TaskStatus(
                    task_id=task.id,
                    status=task.status,
                    created_at=task.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    started_at=task.started_at.strftime("%Y-%m-%d %H:%M:%S") if task.started_at else None,  # Fix 7.13
                    completed_at=task.completed_at.strftime("%Y-%m-%d %H:%M:%S") if task.completed_at else None,
                    progress=100 if task.status == "completed" else 0,
                    result=result,
                    error=task.error_message,
                    file_name=task.filename,
                    language="uk",
                    model_size=task.model_size,
                    use_diarization=task.has_diarization,
                    api_key=task.api_key
                ))
            
            return {
                "tasks": tasks_list,
                "total": len(db_tasks),
                "limit": limit,
                "status_filter": status
            }
        
    except Exception as e:
        logger.error(f"Помилка отримання списку задач: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch task list: {str(e)}")

@app.get("/my-tasks")
async def get_my_tasks(
    api_key: str = Depends(verify_api_key),
    limit: int = 50,
    offset: int = 0,
    status: Optional[str] = None
):
    """
    Отримання історії транскрипцій поточного користувача.
    
    Повертає всі задачі, створені з поточним API ключем,
    відсортовані за датою створення (нові спочатку).
    
    Args:
        limit: Кількість задач на сторінці (макс. 200)
        offset: Зміщення для пагінації (default 0)
        status: Фільтр за статусом (queued/processing/completed/failed/cancelled)
    
    Returns:
        {
            "tasks": [...],
            "total": int,
            "limit": int,
            "offset": int,
            "has_more": bool
        }
    """
    # Валідація параметрів
    if limit > 200:
        raise HTTPException(status_code=400, detail="Maximum limit is 200")
    
    if offset < 0:
        raise HTTPException(status_code=400, detail="Offset must be >= 0")
    
    try:
        with get_db_session() as session:
            repo = TaskRepository(session)
            
            # Використовуємо новий метод для пагінації
            db_tasks, total_count = repo.get_by_api_key_paginated(
                api_key=api_key,
                limit=limit + 1,  # Запитуємо +1 щоб визначити has_more
                offset=offset,
                status=status
            )
            
            # Визначаємо чи є ще задачі
            has_more = len(db_tasks) > limit
            if has_more:
                db_tasks = db_tasks[:limit]
            
            # Конвертуємо в TaskStatus
            tasks_list = []
            for task in db_tasks:
                result = None
                if task.result_json:
                    try:
                        result = json.loads(task.result_json)
                    except:
                        pass
                
                tasks_list.append(TaskStatus(
                    task_id=task.id,
                    status=task.status,
                    created_at=task.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    started_at=task.started_at.strftime("%Y-%m-%d %H:%M:%S") if task.started_at else None,
                    completed_at=task.completed_at.strftime("%Y-%m-%d %H:%M:%S") if task.completed_at else None,
                    progress=100 if task.status == "completed" else 0,
                    result=result,
                    error=task.error_message,
                    file_name=task.filename,
                    language="uk",
                    model_size=task.model_size,
                    use_diarization=task.has_diarization,
                    api_key=task.api_key
                ))
            
            return {
                "tasks": tasks_list,
                "total": total_count,
                "limit": limit,
                "offset": offset,
                "has_more": has_more,
                "status_filter": status
            }
        
    except Exception as e:
        logger.error(f"Помилка отримання історії задач: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")

@app.delete("/task/{task_id}")
async def cancel_task(task_id: str, api_key: str = Depends(verify_api_key)):
    """Скасування задачі (тільки якщо вона ще в черзі)"""
    if task_id not in tasks:
        task_status = load_task_status(task_id)
        if not task_status:
            raise HTTPException(status_code=404, detail="Task not found")
        tasks[task_id] = task_status
    
    task_status = tasks[task_id]
    
    if task_status.status == "completed":
        raise HTTPException(status_code=400, detail="Task already completed")
    
    if task_status.status == "processing":
        raise HTTPException(status_code=400, detail="Task already processing and cannot be cancelled")
    
    if task_status.status == "failed":
        raise HTTPException(status_code=400, detail="Task already failed")
    
    # Видаляємо задачу з черги (якщо вона там є)
    # Примітка: це спрощена реалізація, в реальному проекті потрібно більш складну логіку
    task_status.status = "cancelled"
    task_status.completed_at = time.strftime("%Y-%m-%d %H:%M:%S")
    save_task_status(task_id, task_status)
    
    return {"message": f"Task {task_id} was cancelled"}


# Адмін endpoints
@app.post("/admin/generate-key", response_model=GenerateKeyResponse)
async def generate_api_key(
    request: GenerateKeyRequest,
    master_token: str = Depends(verify_master_token)
):
    """Генерує новий API ключ (потребує master токен)"""
    try:
        api_key = api_key_manager.generate_api_key(request.client_name)
        key_info = api_key_manager.get_api_key_info(api_key)
        
        return GenerateKeyResponse(
            api_key=api_key,
            client_name=key_info["client_name"],
            created_at=key_info["created_at"]
        )
    except Exception as e:
        logger.error(f"Помилка генерації API ключа: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate API key: {str(e)}")

@app.post("/admin/delete-key")
async def delete_api_key(
    request: DeleteKeyRequest,
    master_token: str = Depends(verify_master_token)
):
    """Видаляє API ключ (потребує master токен)"""
    try:
        success = api_key_manager.delete_api_key(request.api_key)
        if success:
            return {"message": "API key deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="API key not found")
    except Exception as e:
        logger.error(f"Помилка видалення API ключа: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete API key: {str(e)}")

@app.get("/admin/list-keys")
async def list_api_keys(master_token: str = Depends(verify_master_token)):
    """Отримує список всіх API ключів (потребує master токен)"""
    try:
        keys = api_key_manager.list_api_keys()
        stats = api_key_manager.get_stats()
        
        return {
            "keys": keys,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Помилка отримання списку ключів: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch API key list: {str(e)}")

@app.post("/admin/update-key-notes")
async def update_key_notes(
    request: UpdateKeyNotesRequest,
    master_token: str = Depends(verify_master_token)
):
    """Оновлює нотатки для API ключа (потребує master токен)"""
    try:
        success = api_key_manager.update_api_key_notes(request.api_key, request.notes)
        if success:
            return {"message": "Notes updated successfully"}
        else:
            raise HTTPException(status_code=404, detail="API key not found")
    except Exception as e:
        logger.error(f"Помилка оновлення нотаток: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update notes: {str(e)}")

@app.post("/admin/toggle-key-status")
async def toggle_key_status(
    request: ToggleKeyStatusRequest,
    master_token: str = Depends(verify_master_token)
):
    """Перемикає статус API ключа (потребує master токен)"""
    try:
        success = api_key_manager.toggle_api_key_status(request.api_key)
        if success:
            key_info = api_key_manager.get_api_key_info(request.api_key)
            status = "active" if key_info.get("active", True) else "inactive"
            return {"message": f"API key is now {status}"}
        else:
            raise HTTPException(status_code=404, detail="API key not found")
    except Exception as e:
        logger.error(f"Помилка зміни статусу ключа: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to change status: {str(e)}")

@app.get("/admin/key-details/{api_key}")
async def get_key_details(
    api_key: str,
    master_token: str = Depends(verify_master_token)
):
    """Отримує детальну інформацію про API ключ (потребує master токен)"""
    try:
        key_info = api_key_manager.get_api_key_info(api_key)
        if key_info:
            return {
                "key": api_key,
                "client_name": key_info["client_name"],
                "created_at": key_info["created_at"],
                "active": key_info.get("active", True),
                "usage_count": key_info.get("usage_count", 0),
                "last_used": key_info.get("last_used"),
                "total_requests": key_info.get("total_requests", 0),
                "successful_requests": key_info.get("successful_requests", 0),
                "failed_requests": key_info.get("failed_requests", 0),
                "total_processing_time": round(key_info.get("total_processing_time", 0), 2),
                "average_processing_time": round(key_info.get("total_processing_time", 0) / max(key_info.get("total_requests", 1), 1), 2),
                "notes": key_info.get("notes", "")
            }
        else:
            raise HTTPException(status_code=404, detail="API key not found")
    except Exception as e:
        logger.error(f"Помилка отримання деталей ключа: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch key details: {str(e)}")


# ============== MODEL MANAGEMENT ENDPOINTS ==============

@app.get("/admin/model-status")
async def get_model_status(master_token: str = Depends(verify_master_token)):
    """
    Отримує статус завантаженої моделі та пам'яті.
    
    Повертає інформацію про:
    - Поточну завантажену модель
    - Доступну та загальну пам'ять
    - Вимоги до пам'яті для різних моделей
    """
    try:
        from models.model_manager import model_manager
        
        status = model_manager.get_status()
        
        # Додаємо інформацію про чергу та активні задачі
        queue_size = task_queue.qsize() if task_queue else 0
        active_tasks = len([t for t in tasks.values() if getattr(t, 'status', None) == "processing"])
        return {
            **status,
            "queue_size": queue_size,
            "queue_max_size": 25,
            "active_tasks": active_tasks,
        }
    except ImportError:
        # Fallback якщо model_manager недоступний
        return {
            "model_loaded": transcription_service.whisper_model is not None if transcription_service else False,
            "current_model_size": transcription_service.whisper_model.model_size if transcription_service and transcription_service.whisper_model else None,
            "error": "ModelManager not available"
        }
    except Exception as e:
        logger.error(f"Помилка отримання статусу моделі: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model status: {str(e)}")


@app.post("/admin/unload-model")
async def unload_model(master_token: str = Depends(verify_master_token)):
    """
    Вивантажує поточну модель з пам'яті.
    
    Використовуйте для звільнення RAM без перезапуску сервера.
    Наступний запит на транскрипцію автоматично завантажить модель.
    """
    try:
        from models.model_manager import model_manager
        
        if model_manager.is_loading:
            raise HTTPException(status_code=409, detail="Model is currently loading, cannot unload")
        
        # Перевіряємо чи є активні задачі
        queue_size = task_queue.qsize() if task_queue else 0
        if queue_size > 0:
            raise HTTPException(
                status_code=409, 
                detail=f"Cannot unload model: {queue_size} tasks in queue. Wait for completion or cancel tasks."
            )
        
        old_size = model_manager.current_model_size
        success = model_manager.unload_model()
        
        if success:
            # Оновлюємо посилання в transcription_service
            if transcription_service and transcription_service.whisper_model:
                transcription_service.whisper_model.model = None
            
            return {
                "message": f"Model {old_size} unloaded successfully",
                "available_memory_gb": round(model_manager.get_available_memory_gb(), 2)
            }
        else:
            return {"message": "No model was loaded"}
            
    except ImportError:
        raise HTTPException(status_code=501, detail="ModelManager not available")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Помилка вивантаження моделі: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to unload model: {str(e)}")


@app.post("/admin/switch-model/{model_size}")
async def switch_model(
    model_size: str,
    master_token: str = Depends(verify_master_token)
):
    """
    Перемикає на іншу модель, вивантажуючи поточну.
    
    Параметри:
    - model_size: tiny, base, small, medium, large
    
    Перевіряє доступну пам'ять перед завантаженням.
    """
    if model_size not in ["tiny", "base", "small", "medium", "large"]:
        raise HTTPException(status_code=400, detail="Invalid model size. Use: tiny, base, small, medium, large")
    
    try:
        from models.model_manager import model_manager
        
        if model_manager.is_loading:
            raise HTTPException(status_code=409, detail="Another model is currently loading")
        
        # Перевіряємо чи можна завантажити
        can_load, reason = model_manager.can_load_model(model_size)
        if not can_load:
            raise HTTPException(status_code=507, detail=f"Insufficient memory: {reason}")
        
        old_size = model_manager.current_model_size
        
        # Завантажуємо нову модель (стара буде автоматично вивантажена)
        device = "cuda" if transcription_service and hasattr(transcription_service, 'whisper_model') and transcription_service.whisper_model.device == "cuda" else "cpu"
        
        model = model_manager.load_model(model_size, device)
        
        # Оновлюємо посилання в transcription_service
        if transcription_service and transcription_service.whisper_model:
            transcription_service.whisper_model.model = model
            transcription_service.whisper_model.model_size = model_size
        
        return {
            "message": f"Switched from {old_size or 'none'} to {model_size}",
            "current_model": model_size,
            "available_memory_gb": round(model_manager.get_available_memory_gb(), 2)
        }
        
    except ImportError:
        raise HTTPException(status_code=501, detail="ModelManager not available")
    except MemoryError as e:
        raise HTTPException(status_code=507, detail=f"Insufficient memory: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Помилка перемикання моделі: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to switch model: {str(e)}")


@app.get("/admin", response_class=HTMLResponse)
async def admin_panel(request: Request):
    """Адмін панель для управління API ключами"""
    # Перевіряємо master токен з query параметра
    master_token = request.query_params.get("master_token")
    if not master_token or not api_key_manager.verify_master_token(master_token):
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>API Admin Panel - Access Denied</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 50px; text-align: center; }
                .error { color: #d32f2f; background: #ffebee; padding: 20px; border-radius: 8px; }
            </style>
        </head>
        <body>
            <div class="error">
                <h1>🔒 Access Denied</h1>
                <p>Missing or invalid master token</p>
                <p>Use: <code>/admin?master_token=YOUR_MASTER_TOKEN</code></p>
            </div>
        </body>
        </html>
        """, status_code=401)
    
    # Отримуємо список ключів
    try:
        keys = api_key_manager.list_api_keys()
        stats = api_key_manager.get_stats()
    except Exception as e:
        keys = []
        stats = {"total_keys": 0, "active_keys": 0, "inactive_keys": 0}
    
    # Генеруємо HTML
    keys_html = ""
    for key in keys:
        status_class = "active" if key["active"] else "inactive"
        keys_html += f"""
        <tr class="{status_class}">
            <td><code>{key["key"][:20]}...</code></td>
            <td>{key["client_name"]}</td>
            <td>{key["created_at"][:19]}</td>
            <td>
                <button onclick="deleteKey('{key["key"]}')" class="delete-btn">Delete</button>
            </td>
        </tr>
        """
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>API Admin Panel</title>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #1976d2; border-bottom: 2px solid #1976d2; padding-bottom: 10px; }}
            .stats {{ display: flex; gap: 20px; margin: 20px 0; flex-wrap: wrap; }}
            .stat-card {{ background: #e3f2fd; padding: 15px; border-radius: 8px; text-align: center; flex: 1; min-width: 120px; }}
            .stat-number {{ font-size: 24px; font-weight: bold; color: #1976d2; }}
            .form-section {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }}
            .model-section {{ background: #fff3e0; padding: 20px; border-radius: 8px; margin: 20px 0; border: 1px solid #ffcc80; }}
            .model-status {{ display: flex; gap: 15px; align-items: center; flex-wrap: wrap; margin: 10px 0; }}
            .model-info {{ background: white; padding: 10px 15px; border-radius: 6px; border: 1px solid #ddd; }}
            .model-info strong {{ color: #e65100; }}
            .memory-bar {{ width: 200px; height: 20px; background: #e0e0e0; border-radius: 10px; overflow: hidden; }}
            .memory-fill {{ height: 100%; background: linear-gradient(90deg, #4caf50, #ff9800, #f44336); transition: width 0.3s; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background: #f8f9fa; font-weight: bold; }}
            .active {{ background: #e8f5e8; }}
            .inactive {{ background: #ffe8e8; }}
            input[type="text"] {{ width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }}
            select {{ padding: 8px 12px; border: 1px solid #ddd; border-radius: 4px; }}
            button {{ padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; margin: 2px; }}
            .generate-btn {{ background: #4caf50; color: white; }}
            .delete-btn {{ background: #f44336; color: white; }}
            .unload-btn {{ background: #ff9800; color: white; }}
            .switch-btn {{ background: #2196f3; color: white; }}
            .refresh-btn {{ background: #9e9e9e; color: white; }}
            .generate-btn:hover {{ background: #45a049; }}
            .delete-btn:hover {{ background: #da190b; }}
            .unload-btn:hover {{ background: #f57c00; }}
            .switch-btn:hover {{ background: #1976d2; }}
            .refresh-btn:hover {{ background: #757575; }}
            button:disabled {{ background: #ccc; cursor: not-allowed; }}
            .new-key {{ background: #e8f5e8; padding: 15px; border-radius: 8px; margin: 10px 0; display: none; }}
            .new-key code {{ background: #f0f0f0; padding: 5px; border-radius: 3px; }}
            .loading {{ opacity: 0.6; pointer-events: none; }}
            .status-badge {{ padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }}
            .status-loaded {{ background: #c8e6c9; color: #2e7d32; }}
            .status-unloaded {{ background: #ffcdd2; color: #c62828; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Admin Panel</h1>
            
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-number">{stats["total_keys"]}</div>
                    <div>Total keys</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{stats["active_keys"]}</div>
                    <div>Active</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{stats["inactive_keys"]}</div>
                    <div>Inactive</div>
                </div>
            </div>
            
            <!-- Model Management Section -->
            <div class="model-section">
                <h3>🧠 Model Management</h3>
                <div class="model-status" id="modelStatus">
                    <div class="model-info">
                        <strong>Model:</strong> <span id="currentModel">Loading...</span>
                        <span id="modelBadge" class="status-badge status-unloaded">—</span>
                    </div>
                    <div class="model-info">
                        <strong>RAM:</strong> <span id="memoryInfo">—</span>
                        <div class="memory-bar">
                            <div class="memory-fill" id="memoryBar" style="width: 0%"></div>
                        </div>
                    </div>
                    <div class="model-info">
                        <strong>Queue:</strong> <span id="queueInfo">—</span>
                    </div>
                </div>
                <div style="margin-top: 15px;">
                    <button class="refresh-btn" onclick="refreshModelStatus()">🔄 Refresh</button>
                    <button class="unload-btn" id="unloadBtn" onclick="unloadModel()">📤 Unload Model</button>
                    <select id="modelSelect">
                        <option value="tiny">tiny (~0.5GB)</option>
                        <option value="base">base (~0.8GB)</option>
                        <option value="small">small (~1.2GB)</option>
                        <option value="medium">medium (~2.5GB)</option>
                        <option value="large">large (~4.5GB)</option>
                    </select>
                    <button class="switch-btn" onclick="switchModel()">🔄 Switch Model</button>
                </div>
            </div>
            
            <div class="form-section">
                <h3>➕ Create a new API key</h3>
                <input type="text" id="clientName" placeholder="Client name" />
                <button class="generate-btn" onclick="generateKey()">Generate key</button>
                <div id="newKey" class="new-key"></div>
            </div>
            
            <div class="form-section">
                <h3>📋 API key list</h3>
                <table>
                    <thead>
                        <tr>
                            <th>API key</th>
                            <th>Client</th>
                            <th>Created</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {keys_html}
                    </tbody>
                </table>
            </div>
        </div>
        
        <script>
            const masterToken = '{master_token}';
            
            // Load model status on page load
            document.addEventListener('DOMContentLoaded', refreshModelStatus);
            
            async function refreshModelStatus() {{
                try {{
                    const response = await fetch('/admin/model-status', {{
                        headers: {{ 'Authorization': 'Bearer ' + masterToken }}
                    }});
                    
                    if (response.ok) {{
                        const data = await response.json();
                        
                        // Update model info
                        const modelSpan = document.getElementById('currentModel');
                        const modelBadge = document.getElementById('modelBadge');
                        const unloadBtn = document.getElementById('unloadBtn');
                        
                        if (data.model_loaded) {{
                            modelSpan.textContent = data.current_model_size || 'unknown';
                            modelBadge.textContent = 'LOADED';
                            modelBadge.className = 'status-badge status-loaded';
                            unloadBtn.disabled = false;
                        }} else {{
                            modelSpan.textContent = 'None';
                            modelBadge.textContent = 'UNLOADED';
                            modelBadge.className = 'status-badge status-unloaded';
                            unloadBtn.disabled = true;
                        }}
                        
                        // Update memory info
                        const memInfo = document.getElementById('memoryInfo');
                        const memBar = document.getElementById('memoryBar');
                        const usedMem = data.total_memory_gb - data.available_memory_gb;
                        const memPercent = (usedMem / data.total_memory_gb * 100).toFixed(0);
                        memInfo.textContent = `${{data.available_memory_gb.toFixed(1)}}GB free / ${{data.total_memory_gb.toFixed(1)}}GB`;
                        memBar.style.width = memPercent + '%';
                        
                        // Update queue info
                        document.getElementById('queueInfo').textContent = 
                            `${{data.queue_size || 0}} / ${{data.queue_max_size || 25}}`;
                    }}
                }} catch (error) {{
                    console.error('Failed to fetch model status:', error);
                }}
            }}
            
            async function unloadModel() {{
                if (!confirm('Unload the current model? New transcription requests will reload it automatically.')) {{
                    return;
                }}
                
                const btn = document.getElementById('unloadBtn');
                btn.disabled = true;
                btn.textContent = '⏳ Unloading...';
                
                try {{
                    const response = await fetch('/admin/unload-model', {{
                        method: 'POST',
                        headers: {{ 'Authorization': 'Bearer ' + masterToken }}
                    }});
                    
                    const data = await response.json();
                    
                    if (response.ok) {{
                        alert('✅ ' + data.message);
                        refreshModelStatus();
                    }} else {{
                        alert('❌ ' + (data.detail || 'Failed to unload model'));
                    }}
                }} catch (error) {{
                    alert('Error: ' + error.message);
                }} finally {{
                    btn.textContent = '📤 Unload Model';
                    refreshModelStatus();
                }}
            }}
            
            async function switchModel() {{
                const modelSize = document.getElementById('modelSelect').value;
                
                if (!confirm(`Switch to ${{modelSize}} model? This will unload the current model.`)) {{
                    return;
                }}
                
                const btn = document.querySelector('.switch-btn');
                btn.disabled = true;
                btn.textContent = '⏳ Loading...';
                
                try {{
                    const response = await fetch('/admin/switch-model/' + modelSize, {{
                        method: 'POST',
                        headers: {{ 'Authorization': 'Bearer ' + masterToken }}
                    }});
                    
                    const data = await response.json();
                    
                    if (response.ok) {{
                        alert('✅ ' + data.message);
                    }} else {{
                        alert('❌ ' + (data.detail || 'Failed to switch model'));
                    }}
                }} catch (error) {{
                    alert('Error: ' + error.message);
                }} finally {{
                    btn.disabled = false;
                    btn.textContent = '🔄 Switch Model';
                    refreshModelStatus();
                }}
            }}
            
            async function generateKey() {{
                const clientName = document.getElementById('clientName').value;
                if (!clientName) {{
                    alert('Enter a client name');
                    return;
                }}
                
                try {{
                    const response = await fetch('/admin/generate-key', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json',
                            'Authorization': 'Bearer ' + masterToken
                        }},
                        body: JSON.stringify({{ client_name: clientName }})
                    }});
                    
                    if (response.ok) {{
                        const data = await response.json();
                        const newKeyDiv = document.getElementById('newKey');
                        newKeyDiv.innerHTML = `
                            <h4>✅ New API key created!</h4>
                            <p><strong>Client:</strong> ${{data.client_name}}</p>
                            <p><strong>API key:</strong> <code>${{data.api_key}}</code></p>
                            <p><strong>Created:</strong> ${{data.created_at}}</p>
                            <p style="color: #d32f2f;"><strong>⚠️ Save this key! It will not be shown again.</strong></p>
                        `;
                        newKeyDiv.style.display = 'block';
                        document.getElementById('clientName').value = '';
                        setTimeout(() => location.reload(), 2000);
                    }} else {{
                        alert('Failed to create API key');
                    }}
                }} catch (error) {{
                    alert('Error: ' + error.message);
                }}
            }}
            
            async function deleteKey(apiKey) {{
                if (!confirm('Are you sure you want to delete this API key?')) {{
                    return;
                }}
                
                try {{
                    const response = await fetch('/admin/delete-key', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json',
                            'Authorization': 'Bearer ' + masterToken
                        }},
                        body: JSON.stringify({{ api_key: apiKey }})
                    }});
                    
                    if (response.ok) {{
                        alert('API key deleted');
                        location.reload();
                    }} else {{
                        alert('Failed to delete API key');
                    }}
                }} catch (error) {{
                    alert('Error: ' + error.message);
                }}
            }}
            
            // Auto-refresh model status every 30 seconds
            setInterval(refreshModelStatus, 30000);
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(html_content)

@app.get("/admin-panel")
async def admin_panel_static():
    """Об'єднана адмін панель з розширеними функціями"""
    return FileResponse("static/admin.html")

@app.get("/transcription")
async def transcription_page():
    """Веб-сторінка для транскрипції аудіо/відео"""
    return FileResponse("static/transcription.html")

@app.get("/api")
async def api_info():
    """Інформація про API"""
    return {
        "message": "Ukrainian Audio Transcription API (Local Models)",
        "version": "1.0.0",
        "description": "API for Ukrainian audio/video transcription with local speaker-aware models",
        "endpoints": {
            "transcribe": "/transcribe (POST, requires API key, returns task_id)",
            "transcribe_with_diarization": "/transcribe-with-diarization (POST, requires API key)",
            "task_status": "/task/{task_id} (GET, public, check task status)",
            "list_tasks": "/tasks (GET, public, list all tasks with filtering)",
            "cancel_task": "/task/{task_id} (DELETE, requires API key, cancel queued task)",
            "health": "/health (GET, public, includes queue status)",
            "docs": "/docs (GET, public)",
            "api_info": "/api (GET, public)",
            "admin": "/admin (GET, requires master token)",
            "admin_panel": "/admin-panel (GET, unified admin panel with advanced features)",
            "transcription": "/transcription (GET, web interface for audio/video transcription)",
            "admin_generate_key": "/admin/generate-key (POST, requires master token)",
            "admin_delete_key": "/admin/delete-key (POST, requires master token)",
            "admin_list_keys": "/admin/list-keys (GET, requires master token)",
            "admin_update_notes": "/admin/update-key-notes (POST, requires master token)",
            "admin_toggle_status": "/admin/toggle-key-status (POST, requires master token)",
            "admin_key_details": "/admin/key-details/{api_key} (GET, requires master token)"
        },
        "features": [
            "Local transcription via faster-whisper",
            "Quantized models optimized for CPU",
            "Simple operator/customer diarization (WebRTC VAD)",
            "Supports file uploads and remote URLs",
            "Ukrainian-first language support",
            "Optimized for CPU and GPU nodes",
            "API token management"
        ],
        "supported_formats": [
            "Audio: WAV, MP3, M4A, FLAC, OGG",
            "Video: MP4, AVI, MOV, MKV"
        ],
        "model_sizes": ["tiny", "base", "small", "medium", "large", "auto"],
        "languages": ["uk", "en", "ru", "pl", "de", "fr", "es", "it"],
        "note": "An API key is required. Contact the administrator to obtain one."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
