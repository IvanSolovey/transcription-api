"""
Middleware для авторизації API токенів
"""
from fastapi import HTTPException, Header, Request
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Стандартизовані повідомлення про помилки
class AuthError:
    MISSING_TOKEN = "Missing authorization token"
    INVALID_FORMAT = "Invalid token format. Use: Bearer YOUR_TOKEN"
    INVALID_API_KEY = "Invalid or inactive API key"
    INVALID_MASTER_TOKEN = "Invalid master token"
    MISSING_MASTER_TOKEN_QUERY = "Missing master token in query parameters"

def extract_bearer_token(authorization: Optional[str]) -> str:
    """Витягує токен з заголовка Authorization"""
    if not authorization:
        raise HTTPException(status_code=401, detail=AuthError.MISSING_TOKEN)
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail=AuthError.INVALID_FORMAT)
    
    return authorization[7:]  # Видаляємо "Bearer "

async def verify_api_key(request: Request, authorization: Optional[str] = Header(None)):
    """Перевіряє API ключ з заголовка Authorization"""
    api_key = extract_bearer_token(authorization)
    
    # Імпортуємо менеджер тут щоб уникнути циклічних імпортів
    from api_auth import api_key_manager
    
    if not api_key_manager.verify_api_key(api_key):
        raise HTTPException(status_code=401, detail=AuthError.INVALID_API_KEY)
    
    return api_key

async def verify_master_token(request: Request, authorization: Optional[str] = Header(None)):
    """Перевіряє master токен"""
    master_token = extract_bearer_token(authorization)
    
    # Імпортуємо менеджер тут щоб уникнути циклічних імпортів
    from api_auth import api_key_manager
    
    if not api_key_manager.verify_master_token(master_token):
        raise HTTPException(status_code=401, detail=AuthError.INVALID_MASTER_TOKEN)
    
    return master_token

def get_master_token_from_query(request: Request) -> Optional[str]:
    """Отримує master токен з query параметра"""
    return request.query_params.get("master_token")

def verify_master_token_from_query(request: Request) -> str:
    """Перевіряє master токен з query параметра"""
    master_token = get_master_token_from_query(request)
    if not master_token:
        raise HTTPException(status_code=401, detail=AuthError.MISSING_MASTER_TOKEN_QUERY)
    
    from api_auth import api_key_manager
    
    if not api_key_manager.verify_master_token(master_token):
        raise HTTPException(status_code=401, detail=AuthError.INVALID_MASTER_TOKEN)
    
    return master_token
