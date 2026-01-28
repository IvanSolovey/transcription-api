"""
Модуль для локальних моделей транскрипції аудіо
"""

from .whisper_model import LocalWhisperModel
from .diarization import SimpleDiarizationService
from .transcription_service import LocalTranscriptionService
from .model_manager import model_manager, ModelManager

__all__ = [
    'LocalWhisperModel',
    'SimpleDiarizationService', 
    'LocalTranscriptionService',
    'model_manager',
    'ModelManager',
]
