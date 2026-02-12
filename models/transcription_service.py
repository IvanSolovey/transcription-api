"""
Основний сервіс для локальної транскрипції аудіо з орфографічною корекцією
"""

import os
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import torch
import librosa
import soundfile as sf
import time
import hashlib

from .config import logger, LANGUAGE_TOOL_AVAILABLE, SUPPORTED_MODELS, QUANTIZED_MODELS, ENABLE_DIARIZATION, DIARIZATION_MAX_WORKERS, MAX_FILE_SIZE_MB, MAX_AUDIO_DURATION_MINUTES, MEMORY_PRESSURE_THRESHOLD
from .whisper_model import LocalWhisperModel
from .diarization import SimpleDiarizationService

# Імпорт моніторингу пам'яті
try:
    from memory_monitor import memory_monitor
    MEMORY_MONITOR_AVAILABLE = True
except ImportError:
    MEMORY_MONITOR_AVAILABLE = False
    logger.warning("Моніторинг пам'яті недоступний")

class LocalTranscriptionService:
    """Сервіс для локальної транскрипції аудіо з орфографічною корекцією"""
    
    def __init__(self, beam_size: int = 1, best_of: int = 1, word_timestamps: bool = True):
        self.models_loaded = False
        self.whisper_model = None
        self.diarization_service = None
        self.language_tool = None
        
        # Параметри продуктивності
        self.beam_size = beam_size
        self.best_of = best_of
        self.word_timestamps = word_timestamps
        
        # Кешування аудіо для оптимізації (мінімізовано для економії пам'яті)
        self._audio_cache = {}
        self._cache_max_size = 2  # Максимум 2 файли в кеші (мінімізовано для економії пам'яті)
        
        # Кешування результатів LanguageTool для уникнення повторних перевірок
        self._language_tool_cache = {}
        self._lt_cache_max_size = 25  # Максимум 25 текстів в кеші LanguageTool (мінімізовано)
        
        # Ініціалізуємо LanguageTool для української мови (опціонально)
        if LANGUAGE_TOOL_AVAILABLE:
            try:
                import language_tool_python
                # Оптимізовані налаштування для сервера 8 CPU + 14GB RAM
                self.language_tool = language_tool_python.LanguageTool(
                    'uk-UA',
                    config={
                        'maxSpellingSuggestions': 3,  # Менше пропозицій = швидше
                        'maxErrorsPerWordRate': 0.3,  # Обмежуємо помилки
                        'maxLength': 10000,  # Обмежуємо довжину тексту
                    }
                )
                logger.info("LanguageTool ініціалізовано для української мови (оптимізовано)")
            except Exception as e:
                logger.warning(f"LanguageTool недоступний (потрібен Java): {e}")
                self.language_tool = None
        else:
            logger.info("LanguageTool не встановлений - орфографічна корекція вимкнена")
        
    def load_models(self, model_size: str = "auto") -> bool:
        """Завантажує всі необхідні моделі"""
        try:
            logger.info("Завантаження локальних моделей...")
            
            # Визначаємо пристрій
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Використовується пристрій: {device}")
            
            # Визначаємо розмір моделі (оптимізовано для сервера 14GB RAM + 8 CPU)
            if model_size == "auto":
                # Для української мови використовуємо звичайні моделі (distil тільки для англійської)
                try:
                    import psutil
                    memory_gb = psutil.virtual_memory().total / (1024**3)
                    cpu_count = psutil.cpu_count()
                    
                    if device == "cuda":
                        # Для GPU можна використовувати більші моделі
                        try:
                            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                            
                            if gpu_memory >= 8:  # 8GB+ GPU
                                model_size = "medium"  # Оптимально для GPU
                                logger.info(f"🚀 GPU {gpu_memory:.1f}GB + RAM {memory_gb:.1f}GB - використовується medium модель")
                            elif gpu_memory >= 4:  # 4GB+ GPU
                                model_size = "small"  # Безпечно для GPU
                                logger.info(f"🚀 GPU {gpu_memory:.1f}GB + RAM {memory_gb:.1f}GB - використовується small модель")
                            else:
                                model_size = "base"  # Для малих GPU
                                logger.info(f"🚀 GPU {gpu_memory:.1f}GB + RAM {memory_gb:.1f}GB - використовується base модель")
                        except:
                            model_size = "small"  # Fallback для GPU
                    else:
                        # Для CPU використовуємо quantized моделі (рекомендація ChatGPT)
                        if memory_gb >= 12 and cpu_count >= 8:
                            model_size = "medium"  # medium + int8 = quantized - оптимально для потужного сервера
                            logger.info(f"🚀 Потужний сервер {memory_gb:.1f}GB RAM + {cpu_count} CPU - використовується medium модель (quantized)")
                        elif memory_gb >= 8 and cpu_count >= 6:
                            model_size = "small"  # small + int8 = quantized - оптимально для CPU
                            logger.info(f"🚀 Сервер {memory_gb:.1f}GB RAM + {cpu_count} CPU - використовується small модель (quantized)")
                        elif memory_gb >= 6:
                            model_size = "base"  # base + int8 = quantized
                            logger.info(f"💾 Сервер {memory_gb:.1f}GB RAM - використовується base модель (quantized)")
                        else:
                            model_size = "tiny"  # tiny + int8 = quantized для малих серверів
                            logger.info(f"💾 Сервер {memory_gb:.1f}GB RAM - використовується tiny модель (quantized)")
                except:
                    model_size = "base"
            elif model_size not in SUPPORTED_MODELS:
                # Якщо вказано некоректний розмір, використовуємо base
                logger.warning(f"Невідомий розмір моделі: {model_size}, використовується base")
                model_size = "base"
            # Показуємо що використовується quantized модель
            logger.info(f"Обрана модель для української мови: {model_size} (quantized з int8)")
            
            # Завантажуємо Whisper з посиланням на сервіс для кешування
            self.whisper_model = LocalWhisperModel(model_size=model_size, device=device, transcription_service=self)
            if not self.whisper_model.load_model():
                return False
            
            # Діаризація буде ініціалізована тільки при потребі (lazy loading)
            # self.diarization_service = SimpleDiarizationService(self)  # Видалено для оптимізації
            
            self.models_loaded = True
            logger.info("Всі моделі завантажені успішно")
            return True
            
        except Exception as e:
            logger.error(f"Помилка завантаження моделей: {e}")
            return False
    
    def _load_audio_cached(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Завантажує аудіо з кешу або з диску (оптимізовано для економії пам'яті)"""
        try:
            # Створюємо хеш файлу для кешування
            file_stat = os.stat(audio_path)
            file_hash = f"{audio_path}_{file_stat.st_size}_{file_stat.st_mtime}"
            
            if file_hash in self._audio_cache:
                logger.debug(f"Аудіо завантажено з кешу: {audio_path}")
                return self._audio_cache[file_hash]
            
            # Завантажуємо з диску та кешуємо (з економією пам'яті)
            logger.debug(f"Завантаження аудіо з диску: {audio_path}")
            audio, sr = librosa.load(audio_path, sr=16000, mono=True, dtype=np.float32)
            
            # Додаємо в кеш (з обмеженням розміру та автоматичним очищенням)
            if len(self._audio_cache) >= self._cache_max_size:
                # Видаляємо найстаріший елемент
                oldest_key = next(iter(self._audio_cache))
                del self._audio_cache[oldest_key]
                logger.debug(f"Очищено старий аудіо кеш: {oldest_key}")
            
            self._audio_cache[file_hash] = (audio, sr)
            return audio, sr
            
        except Exception as e:
            logger.warning(f"Помилка кешування аудіо: {e}, завантажуємо з диску")
            audio, sr = librosa.load(audio_path, sr=16000, mono=True, dtype=np.float32)
            return audio, sr
    
    def clear_audio_cache(self):
        """Очищує кеш аудіо для економії пам'яті"""
        cache_size = len(self._audio_cache)
        self._audio_cache.clear()
        logger.info(f"Очищено аудіо кеш: {cache_size} файлів")
    
    def clear_language_tool_cache(self):
        """Очищує кеш LanguageTool для економії пам'яті"""
        cache_size = len(self._language_tool_cache)
        self._language_tool_cache.clear()
        logger.info(f"Очищено LanguageTool кеш: {cache_size} текстів")
    
    def clear_all_caches(self):
        """Очищує всі кеші для економії пам'яті"""
        self.clear_audio_cache()
        self.clear_language_tool_cache()
        logger.info("Всі кеші очищено")
    
    def _correct_text(self, text: str, language: str) -> str:
        """Орфографічна корекція тексту для української мови з кешуванням"""
        if not text or language != "uk" or not self.language_tool:
            return text
        
        try:
            # Перевіряємо кеш (використовуємо детермінований hash)
            import hashlib
            text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
            if text_hash in self._language_tool_cache:
                logger.debug("Орфографічна корекція завантажена з кешу")
                return self._language_tool_cache[text_hash]
            
            # Виправляємо помилки через LanguageTool (безпечний спосіб)
            matches = self.language_tool.check(text)
            corrected_text = text
            for match in reversed(matches):  # Обробляємо з кінця щоб не змінювати індекси
                if match.replacements:
                    corrected_text = corrected_text[:match.offset] + match.replacements[0] + corrected_text[match.offset + match.errorLength:]
            
            # Кешуємо результат
            if len(self._language_tool_cache) >= self._lt_cache_max_size:
                # Видаляємо найстаріший елемент
                oldest_key = next(iter(self._language_tool_cache))
                del self._language_tool_cache[oldest_key]
            
            self._language_tool_cache[text_hash] = corrected_text
            
            if corrected_text != text:
                logger.debug("Застосовано орфографічну корекцію")
            
            return corrected_text
            
        except Exception as e:
            logger.warning(f"Помилка орфографічної корекції: {e}")
            return text
    
    def _correct_text_batch(self, texts: List[str], language: str) -> List[str]:
        """Пакетна орфографічна корекція по реченнях для максимальної ефективності"""
        if not texts or language != "uk" or not self.language_tool:
            return texts
        
        try:
            # Спочатку перевіряємо кеш для кожного тексту
            cached_results = {}
            texts_to_process = []
            text_indices = []
            
            for i, text in enumerate(texts):
                if not text.strip():
                    cached_results[i] = text
                    continue
                
                import hashlib
                text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
                if text_hash in self._language_tool_cache:
                    cached_results[i] = self._language_tool_cache[text_hash]
                else:
                    texts_to_process.append(text)
                    text_indices.append(i)
            
            # Якщо всі тексти в кеші, повертаємо результати
            if not texts_to_process:
                logger.debug("Всі тексти знайдені в кеші LanguageTool")
                return [cached_results[i] for i in range(len(texts))]
            
            # Розбиваємо тексти на речення та зберігаємо мапінг
            sentence_mapping = []  # [(original_text_index, sentence_start, sentence_end), ...]
            all_sentences = []
            
            for text_idx, text in enumerate(texts_to_process):
                # Розбиваємо на речення (простий алгоритм)
                sentences = self._split_into_sentences(text)
                
                for sentence in sentences:
                    if sentence.strip():
                        sentence_mapping.append((text_idx, len(all_sentences)))
                        all_sentences.append(sentence.strip())
            
            if not all_sentences:
                return texts
            
            # Обмежуємо розмір тексту для LanguageTool (максимум 5000 символів)
            max_chunk_size = 5000
            if len(" ".join(all_sentences)) > max_chunk_size:
                logger.debug(f"Текст занадто великий ({len(' '.join(all_sentences))} символів), обробляємо частинами")
                return self._correct_text_batch_chunked(texts_to_process, language, text_indices, cached_results)
            
            # Об'єднуємо всі речення в один текст для корекції
            combined_text = " ".join(all_sentences)
            
            # Виконуємо корекцію (безпечний спосіб без utils.correct)
            matches = self.language_tool.check(combined_text)
            corrected_text = combined_text
            for match in reversed(matches):  # Обробляємо з кінця щоб не змінювати індекси
                if match.replacements:
                    corrected_text = corrected_text[:match.offset] + match.replacements[0] + corrected_text[match.offset + match.errorLength:]
            
            # Кешуємо результат (детермінований hash)
            import hashlib
            combined_hash = hashlib.md5(combined_text.encode("utf-8")).hexdigest()
            self._language_tool_cache[combined_hash] = corrected_text
            
            # Розбиваємо назад на речення
            corrected_sentences = self._split_into_sentences(corrected_text)
            
            # Відновлюємо оригінальну структуру текстів
            corrected_texts = [""] * len(texts)
            
            # Спочатку додаємо кешовані результати
            for i, result in cached_results.items():
                corrected_texts[i] = result
            
            # Потім обробляємо нові тексти
            for text_idx, text in enumerate(texts_to_process):
                original_idx = text_indices[text_idx]
                
                # Знаходимо речення для цього тексту
                text_sentences = []
                for mapping_text_idx, sentence_idx in sentence_mapping:
                    if mapping_text_idx == text_idx:
                        if sentence_idx < len(corrected_sentences):
                            text_sentences.append(corrected_sentences[sentence_idx])
                        else:
                            # Fallback на оригінальне речення
                            original_sentences = self._split_into_sentences(text)
                            if len(text_sentences) < len(original_sentences):
                                text_sentences.append(original_sentences[len(text_sentences)])
                
                # Об'єднуємо речення назад в текст
                corrected_texts[original_idx] = " ".join(text_sentences) if text_sentences else text
                
                # Кешуємо індивідуальний результат (детермінований hash)
                import hashlib
                text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
                self._language_tool_cache[text_hash] = corrected_texts[original_idx]
            
            logger.debug(f"Пакетна корекція: {len(all_sentences)} речень оброблено, {len(cached_results)} з кешу")
            return corrected_texts
            
        except Exception as e:
            logger.warning(f"Помилка пакетної корекції по реченнях: {e}")
            # Fallback на індивідуальну корекцію
            return [self._correct_text(text, language) for text in texts]
    
    def _correct_text_batch_chunked(self, texts: List[str], language: str, text_indices: List[int], cached_results: dict) -> List[str]:
        """Обробка великих текстів частинами"""
        corrected_texts = [""] * (len(texts) + len(cached_results))
        
        # Додаємо кешовані результати
        for i, result in cached_results.items():
            corrected_texts[i] = result
        
        # Обробляємо тексти частинами
        chunk_size = 3  # Обробляємо по 3 тексти за раз
        for i in range(0, len(texts), chunk_size):
            chunk_texts = texts[i:i+chunk_size]
            chunk_indices = text_indices[i:i+chunk_size]
            
            try:
                chunk_results = self._correct_text_batch(chunk_texts, language)
                for j, result in enumerate(chunk_results):
                    corrected_texts[chunk_indices[j]] = result
            except Exception as e:
                logger.warning(f"Помилка обробки частини тексту: {e}")
                # Fallback на індивідуальну корекцію
                for j, text in enumerate(chunk_texts):
                    corrected_texts[chunk_indices[j]] = self._correct_text(text, language)
        
        return corrected_texts
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Розбиває текст на речення (простий алгоритм для української мови)"""
        if not text.strip():
            return []
        
        # Простий алгоритм розбиття на речення
        import re
        
        # Додаємо пробіли перед знаками пунктуації
        text = re.sub(r'([.!?])([А-ЯЄІЇҐ])', r'\1 \2', text)
        
        # Розбиваємо по знаках завершення речень
        sentences = re.split(r'[.!?]+', text)
        
        # Очищаємо та фільтруємо
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 2:  # Ігноруємо дуже короткі фрагменти
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _validate_audio_file(self, audio_path: str) -> None:
        """Перевіряє розмір файлу та тривалість аудіо"""
        try:
            # Перевіряємо розмір файлу
            file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
            if file_size_mb > MAX_FILE_SIZE_MB:
                raise ValueError(f"Файл занадто великий: {file_size_mb:.1f}MB > {MAX_FILE_SIZE_MB}MB")
            
            # Перевіряємо тривалість аудіо (використовуємо кеш якщо можливо)
            try:
                # Спочатку спробуємо отримати з кешу
                audio, sr = self._load_audio_cached(audio_path)
                # Якщо в кеші було 16kHz, перезавантажуємо з оригінальною частотою
                if sr == 16000:
                    audio, sr = librosa.load(audio_path, sr=None, mono=True, dtype=np.float32)
            except:
                # Fallback: завантажуємо напряму
                audio, sr = librosa.load(audio_path, sr=None, mono=True, dtype=np.float32)
            
            duration_minutes = len(audio) / sr / 60
            if duration_minutes > MAX_AUDIO_DURATION_MINUTES:
                raise ValueError(f"Аудіо занадто довге: {duration_minutes:.1f}хв > {MAX_AUDIO_DURATION_MINUTES}хв")
            
            logger.info(f"✅ Файл валідний: {file_size_mb:.1f}MB, {duration_minutes:.1f}хв")
            
        except Exception as e:
            logger.error(f"Помилка валідації файлу: {e}")
            raise

    def _safe_switch_model(self, target_model_size: str) -> bool:
        """
        Безпечно перемикає модель з перевіркою пам'яті.
        
        Returns:
            True якщо модель успішно змінена або вже потрібного розміру
        """
        if target_model_size == "auto":
            return True  # auto = використовуємо поточну
        
        if target_model_size == self.whisper_model.model_size:
            return True  # Вже потрібна модель
        
        # Спробуємо безпечно перемкнути через новий метод
        if hasattr(self.whisper_model, 'switch_model'):
            try:
                if self.whisper_model.switch_model(target_model_size):
                    logger.info(f"✅ Модель безпечно змінена на {target_model_size}")
                    return True
                else:
                    logger.warning(f"⚠️ Не вдалося змінити модель на {target_model_size}, використовується {self.whisper_model.model_size}")
                    return False
            except MemoryError as e:
                logger.error(f"❌ Недостатньо пам'яті для {target_model_size}: {e}")
                raise  # Пробрасуємо помилку пам'яті наверх
        else:
            # Fallback на старий метод (небезпечний)
            logger.warning("⚠️ Використовується старий метод зміни моделі")
            return self.load_models(target_model_size)

    def transcribe_simple(self, audio_path: str, language: str = "uk", model_size: str = "auto", use_parallel: bool = False, force_no_chunks: bool = True) -> Dict[str, Any]:
        """Швидка транскрипція з опціональним паралельним обробленням"""
        if not self.models_loaded:
            raise RuntimeError("Моделі не завантажені")
        
        # Перевіряємо файл перед обробкою
        self._validate_audio_file(audio_path)
        
        # Перевіряємо тиск на пам'ять
        if MEMORY_MONITOR_AVAILABLE and memory_monitor.check_memory_pressure():
            logger.warning("⚠️ Високий тиск на пам'ять, примусове очищення")
            memory_monitor.force_garbage_collection()
        
        # Безпечна зміна моделі з перевіркою пам'яті
        if model_size != "auto" and model_size != self.whisper_model.model_size:
            logger.info(f"🔄 Запит зміни моделі з {self.whisper_model.model_size} на {model_size}")
            self._safe_switch_model(model_size)
        
        start_time = time.time()
        
        # Використовуємо контекст моніторингу пам'яті
        if MEMORY_MONITOR_AVAILABLE:
            with memory_monitor.memory_context("транскрипція"):
                try:
                    logger.info(f"Початок швидкої транскрипції з faster-whisper (модель: {self.whisper_model.model_size})...")
                    
                    # Завжди використовуємо послідовну обробку (чанки вимкнено)
                    logger.info("🚫 Чанки вимкнено - використовується послідовна обробка")
                    transcription_result = self.whisper_model.transcribe(audio_path, language)
                    
                    # Обробка результатів з орфографічною корекцією
                    processed_result = self._process_simple_results(transcription_result, language)
                    
                    elapsed_time = time.time() - start_time
                    logger.info(f"Транскрипція завершена успішно за {elapsed_time:.2f} секунд")
                    
                    # Очищуємо кеші для економії пам'яті
                    self.clear_all_caches()
                    
                    # Примусове очищення пам'яті
                    import gc
                    for _ in range(3):
                        gc.collect()
                    
                    # Очищення кешу Python
                    import sys
                    if hasattr(sys, '_clear_type_cache'):
                        sys._clear_type_cache()
                    
                    logger.info("🧹 Примусове очищення пам'яті завершено")
                    
                    return processed_result
                    
                except Exception as e:
                    logger.error(f"Помилка транскрипції: {e}")
                    raise
        else:
            # Fallback без моніторингу пам'яті
            try:
                logger.info(f"Початок швидкої транскрипції з faster-whisper (модель: {self.whisper_model.model_size})...")
                
                # Завжди використовуємо послідовну обробку (чанки вимкнено)
                logger.info("🚫 Чанки вимкнено - використовується послідовна обробка")
                transcription_result = self.whisper_model.transcribe(audio_path, language)
                
                # Обробка результатів з орфографічною корекцією
                processed_result = self._process_simple_results(transcription_result, language)
                
                elapsed_time = time.time() - start_time
                logger.info(f"Транскрипція завершена успішно за {elapsed_time:.2f} секунд")
                
                # Очищуємо кеші для економії пам'яті
                self.clear_all_caches()
                
                # Примусове очищення пам'яті
                import gc
                for _ in range(3):
                    gc.collect()
                
                # Очищення кешу Python
                import sys
                if hasattr(sys, '_clear_type_cache'):
                    sys._clear_type_cache()
                
                logger.info("🧹 Примусове очищення пам'яті завершено")
                
                return processed_result
                
            except Exception as e:
                logger.error(f"Помилка транскрипції: {e}")
                raise
    
    def transcribe_with_diarization(self, audio_path: str, language: str = "uk", model_size: str = "auto", use_parallel: bool = False, force_no_chunks: bool = True) -> Dict[str, Any]:
        """Транскрипція з діаризацією (Оператор/Клієнт) з паралельною обробкою"""
        if not self.models_loaded:
            raise RuntimeError("Моделі не завантажені")
        
        # Безпечна зміна моделі з перевіркою пам'яті
        if model_size != "auto" and model_size != self.whisper_model.model_size:
            logger.info(f"🔄 Запит зміни моделі з {self.whisper_model.model_size} на {model_size}")
            self._safe_switch_model(model_size)
        
        # Lazy loading діаризації - ініціалізуємо тільки при потребі
        if not ENABLE_DIARIZATION:
            logger.warning("⚠️ Діаризація відключена в конфігурації")
            raise RuntimeError("Діаризація відключена в конфігурації")
        
        if self.diarization_service is None:
            logger.info("🔧 Ініціалізація діаризації (lazy loading)...")
            self.diarization_service = SimpleDiarizationService(self)
        
        start_time = time.time()
        try:
            logger.info("Початок транскрипції з діаризацією...")
            
            # Конвертуємо файл та отримуємо аудіо масив напряму
            audio, sr = self.whisper_model._convert_to_optimal_format(audio_path)
            logger.info(f"🔄 Конвертовано аудіо: {len(audio)} зразків, {sr}Hz")
            
            # Діагностика аудіо
            audio_duration = len(audio) / sr
            logger.info(f"📊 Аудіо діагностика: тривалість={audio_duration:.2f}с, частота={sr}Hz, зразків={len(audio)}")
            logger.info(f"📊 Перші 0.5с аудіо: min={audio[:int(0.5*sr)].min():.4f}, max={audio[:int(0.5*sr)].max():.4f}, rms={np.sqrt(np.mean(audio[:int(0.5*sr)]**2)):.4f}")
            
            # Робимо діаризацію з аудіо масивом
            speaker_segments = self.diarization_service.process_audio_array(audio, sr)
            
            if not speaker_segments:
                logger.warning("Діаризація не знайшла сегменти, використовуємо просту транскрипцію")
                return self.transcribe_simple(audio_path, language, use_parallel)
            
            if use_parallel and len(speaker_segments) > 1:
                # Паралельна обробка сегментів з ThreadPoolExecutor
                logger.info(f"Паралельна обробка {len(speaker_segments)} сегментів діаризації...")
                processed_segments = self._process_diarization_segments_parallel(
                    audio, sr, speaker_segments, language
                )
            else:
                # Послідовна обробка (fallback)
                processed_segments = self._process_diarization_segments_sequential(
                    audio, sr, speaker_segments, language
                )
            
            # Підраховуємо статистику по дикторах
            speakers_stats = {}
            full_text = ""
            
            # Сортуємо сегменти за часом для правильного порядку
            processed_segments.sort(key=lambda x: x["start"])
            
            for segment in processed_segments:
                speaker = segment["speaker"]
                if speaker not in speakers_stats:
                    speakers_stats[speaker] = {
                        "speaker": speaker,
                        "segments_count": 0,
                        "total_duration": 0,
                        "first_segment": segment["start"],
                        "last_segment": segment["end"]
                    }
                
                speakers_stats[speaker]["segments_count"] += 1
                speakers_stats[speaker]["total_duration"] += segment["duration"]
                speakers_stats[speaker]["last_segment"] = max(speakers_stats[speaker]["last_segment"], segment["end"])
                
                # Додаємо час до тексту для кращого відстеження
                time_info = f"[{segment['start']:.1f}s-{segment['end']:.1f}s]"
                full_text += f"{time_info} [{speaker}]: {segment['text']}\n"
            
            result = {
                "text": full_text.strip(),
                "segments": processed_segments,
                "speakers": list(speakers_stats.values()),
                "duration": max([s["end"] for s in processed_segments]) if processed_segments else 0,
                "language": language,
                "diarization_type": "simple_alternating_parallel" if use_parallel else "simple_alternating"
            }
            
            elapsed_time = time.time() - start_time
            logger.info(f"Транскрипція з діаризацією завершена: {len(processed_segments)} сегментів за {elapsed_time:.2f} секунд")
            
            # Очищуємо великий аудіо масив з пам'яті
            del audio
            
            # Очищуємо кеші для економії пам'яті
            self.clear_all_caches()
            
            # Примусове очищення пам'яті
            import gc
            gc.collect()
            
            return result
            
        except Exception as e:
            logger.error(f"Помилка транскрипції з діаризацією: {e}")
            # Очищуємо пам'ять навіть при помилці
            try:
                del audio
            except:
                pass
            import gc
            gc.collect()
            raise
    
    def _process_diarization_segments_parallel(self, audio: np.ndarray, sr: int, 
                                             speaker_segments: List[Dict[str, Any]], 
                                             language: str) -> List[Dict[str, Any]]:
        """Паралельна обробка сегментів діаризації з threads (оптимізовано для CPU)"""
        try:
            from concurrent.futures import ThreadPoolExecutor
            import os
            
            # Динамічне визначення кількості потоків (оптимізовано для сервера 14GB RAM + 8 CPU)
            cpu_count = os.cpu_count()
            max_workers = min(DIARIZATION_MAX_WORKERS, len(speaker_segments), cpu_count - 1)  # Залишаємо 1 CPU для системи
            logger.info(f"🚀 Сервер {cpu_count} CPU + 14GB RAM - використовується {max_workers} потоків для паралельної діаризації (threads)")
            
            # Якщо сегментів мало, використовуємо послідовну обробку
            if len(speaker_segments) <= 2 or max_workers <= 1:
                logger.info("Використовується послідовна обробка (мало сегментів)")
                return self._process_diarization_segments_sequential(audio, sr, speaker_segments, language)
            
            # Паралельна обробка з ThreadPoolExecutor (використовуємо одну модель)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Створюємо завдання для кожного сегменту
                futures = [
                    executor.submit(
                        self._process_single_diarization_segment_threaded,
                        audio, sr, segment, language
                    )
                    for segment in speaker_segments
                ]
                
                # Чекаємо завершення всіх завдань
                processed_segments = []
                for i, future in enumerate(futures):
                    try:
                        result = future.result(timeout=300)  # 5 хвилин timeout
                        if result:
                            processed_segments.append(result)
                        
                        # Логуємо прогрес кожні 3 сегменти
                        if (i + 1) % 3 == 0:
                            logger.info(f"Паралельно оброблено {i + 1}/{len(speaker_segments)} сегментів")
                            
                    except Exception as e:
                        logger.warning(f"Помилка обробки сегменту {speaker_segments[i].get('speaker', 'unknown')}: {e}")
                        continue
            
            logger.info(f"Паралельна обробка завершена: {len(processed_segments)}/{len(speaker_segments)} сегментів")
            return processed_segments
            
        except Exception as e:
            logger.error(f"Помилка паралельної обробки діаризації: {e}")
            # Fallback до послідовної обробки
            return self._process_diarization_segments_sequential(audio, sr, speaker_segments, language)
    
    def _process_single_diarization_segment_threaded(self, audio: np.ndarray, sr: int,
                                                    speaker_info: Dict[str, Any], 
                                                    language: str) -> Optional[Dict[str, Any]]:
        """Threaded обробка одного сегменту діаризації (використовує існуючу модель)"""
        try:
            start_time = speaker_info["start"]
            end_time = speaker_info["end"]
            speaker = speaker_info["speaker"]
            
            # Витягуємо сегмент аудіо з попередньо завантаженого масиву
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment_audio = audio[start_sample:end_sample]
            
            # Конвертуємо segment_audio в BytesIO для безпечної передачі
            from io import BytesIO
            import soundfile as sf
            
            buf = BytesIO()
            sf.write(buf, segment_audio, sr, format="WAV", subtype='PCM_16')
            buf.seek(0)
            
            # Використовуємо існуючу модель (thread-safe для CPU)
            segments, info = self.whisper_model.model.transcribe(
                buf,  # Передаємо BytesIO замість масиву
                language=language,
                beam_size=self.beam_size,
                word_timestamps=False,  # Швидше для коротких сегментів
                vad_filter=False,  # Швидше для коротких сегментів
                temperature=0.0,
                best_of=1,
            )
            
            # Обробляємо результат
            segment_text = ""
            for segment in segments:
                segment_text += segment.text + " "
            
            if segment_text.strip():
                # Орфографічна корекція
                if language == "uk":
                    segment_text = self._correct_text(segment_text.strip(), language)
                
                return {
                    "start": start_time,
                    "end": end_time,
                    "text": segment_text.strip(),
                    "speaker": speaker,
                    "duration": end_time - start_time
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Помилка threaded обробки сегменту {speaker}: {e}")
            return None
    
    @staticmethod
    def _process_single_diarization_segment_worker_optimized(audio: np.ndarray, sr: int,
                                                           speaker_info: Dict[str, Any], 
                                                           language: str) -> Optional[Dict[str, Any]]:
        """Оптимізований worker для multiprocessing (статичний метод)"""
        try:
            # Імпортуємо тут, щоб уникнути проблем з multiprocessing
            from faster_whisper import WhisperModel
            import torch
            
            start_time = speaker_info["start"]
            end_time = speaker_info["end"]
            speaker = speaker_info["speaker"]
            
            # Витягуємо сегмент аудіо з попередньо завантаженого масиву
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment_audio = audio[start_sample:end_sample]
            
            # Визначаємо пристрій та compute_type (оптимізовано для сервера 14GB RAM + 8 CPU)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                try:
                    import psutil
                    memory_gb = psutil.virtual_memory().total / (1024**3)
                    cpu_count = psutil.cpu_count()
                    
                    if memory_gb >= 14 and cpu_count >= 8:
                        compute_type = "int8_float16"  # Оптимально для вашого сервера
                    elif memory_gb >= 8:
                        compute_type = "int8_float16"  # Швидше ніж int8
                    else:
                        compute_type = "int8"  # Економніше по пам'яті
                except:
                    compute_type = "int8"
            else:
                compute_type = "float16"
            
            # Завантажуємо модель в worker процесі (оптимізовано для 8 CPU)
            model = WhisperModel("base", device=device, compute_type=compute_type, cpu_threads=4)
            
            # Транскрибуємо сегмент з оптимізованими параметрами
            segments, info = model.transcribe(
                segment_audio,
                language=language,
                beam_size=1,  # Мінімальний для швидкості
                word_timestamps=False,  # Швидше для коротких сегментів
                vad_filter=False,  # Швидше для коротких сегментів
                temperature=0.0,
                best_of=1,
            )
            
            # Обробляємо результат
            segment_text = ""
            for segment in segments:
                segment_text += segment.text + " "
            
            if segment_text.strip():
                return {
                    "start": start_time,
                    "end": end_time,
                    "text": segment_text.strip(),
                    "speaker": speaker,
                    "duration": end_time - start_time
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Помилка worker обробки сегменту {speaker}: {e}")
            return None
    
    def _process_single_diarization_segment_optimized(self, audio: np.ndarray, sr: int,
                                                     speaker_info: Dict[str, Any], 
                                                     language: str) -> Optional[Dict[str, Any]]:
        """Оптимізована обробка одного сегменту діаризації (використовує існуючу модель)"""
        try:
            start_time = speaker_info["start"]
            end_time = speaker_info["end"]
            speaker = speaker_info["speaker"]
            
            # Витягуємо сегмент аудіо з попередньо завантаженого масиву
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment_audio = audio[start_sample:end_sample]
            
            # Конвертуємо segment_audio в BytesIO для безпечної передачі
            from io import BytesIO
            import soundfile as sf
            
            buf = BytesIO()
            sf.write(buf, segment_audio, sr, format="WAV", subtype='PCM_16')
            buf.seek(0)
            
            # Використовуємо існуючу модель замість створення нової
            segments, info = self.whisper_model.model.transcribe(
                buf,  # Передаємо BytesIO замість масиву
                language=language,
                beam_size=self.beam_size,
                word_timestamps=False,  # Швидше для коротких сегментів
                vad_filter=False,  # Швидше для коротких сегментів
                temperature=0.0,
                best_of=1,
            )
            
            # Обробляємо результат
            segment_text = ""
            for segment in segments:
                segment_text += segment.text + " "
            
            if segment_text.strip():
                # Орфографічна корекція
                if language == "uk":
                    segment_text = self._correct_text(segment_text.strip(), language)
                
                return {
                    "start": start_time,
                    "end": end_time,
                    "text": segment_text.strip(),
                    "speaker": speaker,
                    "duration": end_time - start_time
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Помилка обробки сегменту {speaker}: {e}")
            return None
    
    def _process_diarization_segments_sequential(self, audio: np.ndarray, sr: int,
                                               speaker_segments: List[Dict[str, Any]], 
                                               language: str) -> List[Dict[str, Any]]:
        """Послідовна обробка сегментів діаризації (fallback)"""
        processed_segments = []
        
        for speaker_info in speaker_segments:
            try:
                result = self._process_single_diarization_segment_optimized(audio, sr, speaker_info, language)
                if result:
                    processed_segments.append(result)
            except Exception as e:
                logger.warning(f"Помилка обробки сегменту {speaker_info.get('speaker', 'unknown')}: {e}")
                continue
        
        return processed_segments
    
    def _process_single_diarization_segment(self, audio: np.ndarray, sr: int,
                                          speaker_info: Dict[str, Any], 
                                          language: str) -> Optional[Dict[str, Any]]:
        """Обробка одного сегменту діаризації (fallback з тимчасовими файлами)"""
        try:
            start_time = speaker_info["start"]
            end_time = speaker_info["end"]
            speaker = speaker_info["speaker"]
            
            # Витягуємо сегмент аудіо з попередньо завантаженого масиву
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment_audio = audio[start_sample:end_sample]
            
            # Зберігаємо тимчасовий файл для сегменту (fallback метод)
            segment_path = f"temp_segment_{start_time:.1f}_{end_time:.1f}.wav"
            sf.write(segment_path, segment_audio, sr, format='WAV', subtype='PCM_16')
            
            try:
                # Транскрибуємо сегмент з оптимізованими параметрами
                segments, info = self.whisper_model.model.transcribe(
                    segment_path,
                    language=language,
                    beam_size=self.beam_size,
                    word_timestamps=False,  # Швидше
                    vad_filter=False,  # Швидше для коротких сегментів
                    temperature=0.0,
                    best_of=1,
                )
                
                # Обробляємо результат
                segment_text = ""
                for segment in segments:
                    segment_text += segment.text + " "
                
                if segment_text.strip():
                    # Орфографічна корекція
                    if language == "uk":
                        segment_text = self._correct_text(segment_text.strip(), language)
                    
                    return {
                        "start": start_time,
                        "end": end_time,
                        "text": segment_text.strip(),
                        "speaker": speaker,
                        "duration": end_time - start_time
                    }
                
                return None
                
            finally:
                # Видаляємо тимчасовий файл
                try:
                    os.unlink(segment_path)
                except:
                    pass
                    
        except Exception as e:
            logger.warning(f"Помилка обробки сегменту {speaker}: {e}")
            return None
    
    def _process_simple_results(self, transcription_result: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Обробка результатів простої транскрипції з орфографічною корекцією"""
        try:
            if not transcription_result or not transcription_result.get("text"):
                return {
                    "text": "Текст не знайдено - аудіо може бути без мовлення або занадто тихим",
                    "segments": [],
                    "duration": 0,
                    "language": language
                }
            
            # Отримуємо текст
            text = transcription_result.get("text", "").strip()
            
            # Переконуємося, що текст правильно кодується в UTF-8
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='ignore')
            
            # Орфографічна корекція для української мови
            if language == "uk":
                text = self._correct_text(text, language)
            
            # Отримуємо сегменти з орфографічною корекцією
            segments = []
            segment_texts = []
            
            # Спочатку збираємо всі тексти для пакетної корекції
            for segment in transcription_result.get("segments", []):
                segment_text = segment.get("text", "").strip()
                
                # Переконуємося, що текст правильно кодується в UTF-8
                if isinstance(segment_text, bytes):
                    segment_text = segment_text.decode('utf-8', errors='ignore')
                
                segment_texts.append(segment_text)
            
            # Пакетна орфографічна корекція (якщо доступна)
            if language == "uk" and segment_texts:
                try:
                    corrected_texts = self._correct_text_batch(segment_texts, language)
                    if len(corrected_texts) == len(segment_texts):
                        segment_texts = corrected_texts
                except Exception as e:
                    logger.warning(f"Помилка пакетної корекції: {e}")
            
            # Формуємо фінальні сегменти
            for i, segment in enumerate(transcription_result.get("segments", [])):
                segment_text = segment_texts[i] if i < len(segment_texts) else segment.get("text", "").strip()
                
                segments.append({
                    "start": segment.get("start", 0),
                    "end": segment.get("end", 0),
                    "text": segment_text
                })
            
            # Отримуємо тривалість
            duration = transcription_result.get("duration", 0)
            
            return {
                "text": text,
                "segments": segments,
                "duration": duration,
                "language": language
            }
            
        except Exception as e:
            logger.error(f"Помилка обробки результатів: {e}")
            return {
                "text": "Помилка обробки результатів транскрипції",
                "segments": [],
                "duration": 0,
                "language": language
            }
