
"""
Security utilities for authentication and authorization.
"""
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from app.core.config import settings

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Optional[dict]:
    """Verify and decode JWT token."""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except JWTError:
        return None


def create_refresh_token(data: dict) -> str:
    """Create refresh token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=30)  # 30 days
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def generate_password_reset_token(email: str) -> str:
    """Generate password reset token."""
    delta = timedelta(hours=24)  # Token valid for 24 hours
    now = datetime.utcnow()
    expires = now + delta
    exp = expires.timestamp()
    
    to_encode = {
        "sub": email,
        "exp": exp,
        "type": "password_reset"
    }
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def verify_password_reset_token(token: str) -> Optional[str]:
    """Verify password reset token and return email."""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        if payload.get("type") != "password_reset":
            return None
        return payload.get("sub")
    except JWTError:
        return None


def generate_api_key(user_id: int, permissions: list = None) -> str:
    """Generate API key for user."""
    if permissions is None:
        permissions = ["read"]
    
    payload = {
        "sub": str(user_id),
        "permissions": permissions,
        "type": "api_key",
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def verify_api_key(api_key: str) -> Optional[dict]:
    """Verify API key and return payload."""
    try:
        payload = jwt.decode(api_key, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        if payload.get("type") != "api_key":
            return None
        return payload
    except JWTError:
        return None


def create_session_token(user_id: int, session_data: dict) -> str:
    """Create session token with additional data."""
    payload = {
        "sub": str(user_id),
        "session_data": session_data,
        "type": "session",
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def verify_session_token(token: str) -> Optional[dict]:
    """Verify session token."""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        if payload.get("type") != "session":
            return None
        return payload
    except JWTError:
        return None


def hash_sensitive_data(data: str) -> str:
    """Hash sensitive data like patient IDs."""
    return pwd_context.hash(data)


def verify_sensitive_data(data: str, hashed_data: str) -> bool:
    """Verify sensitive data against hash."""
    return pwd_context.verify(data, hashed_data)


def encrypt_sensitive_data(data: str, key: str = None) -> str:
    """Encrypt sensitive data (basic implementation)."""
    # This is a basic implementation - in production, use proper encryption
    import base64
    if key is None:
        key = settings.SECRET_KEY[:16]  # Use first 16 chars as key
    
    # Simple XOR encryption (not recommended for production)
    encrypted = ''.join(chr(ord(c) ^ ord(k)) for c, k in zip(data, key * (len(data) // len(key) + 1)))
    return base64.b64encode(encrypted.encode()).decode()


def decrypt_sensitive_data(encrypted_data: str, key: str = None) -> str:
    """Decrypt sensitive data (basic implementation)."""
    import base64
    if key is None:
        key = settings.SECRET_KEY[:16]
    
    try:
        encrypted = base64.b64decode(encrypted_data.encode()).decode()
        decrypted = ''.join(chr(ord(c) ^ ord(k)) for c, k in zip(encrypted, key * (len(encrypted) // len(key) + 1)))
        return decrypted
    except Exception:
        return ""


def generate_audit_hash(data: dict) -> str:
    """Generate hash for audit trail."""
    import hashlib
    import json
    
    # Sort keys for consistent hashing
    sorted_data = json.dumps(data, sort_keys=True)
    return hashlib.sha256(sorted_data.encode()).hexdigest()


def is_token_expired(token: str) -> bool:
    """Check if token is expired."""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        exp = payload.get("exp")
        if exp is None:
            return True
        return datetime.utcnow() > datetime.fromtimestamp(exp)
    except JWTError:
        return True


def get_token_expiration(token: str) -> Optional[datetime]:
    """Get token expiration datetime."""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        exp = payload.get("exp")
        if exp is None:
            return None
        return datetime.fromtimestamp(exp)
    except JWTError:
        return None


