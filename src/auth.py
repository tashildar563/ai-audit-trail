"""
Authentication and Session Management
"""

import bcrypt
import secrets
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict
from sqlalchemy.orm import Session
from src.database import Client
from fastapi import HTTPException


# Configuration
SECRET_KEY = secrets.token_urlsafe(32)  # In production, use env variable
ALGORITHM = "HS256"
SESSION_DURATION_HOURS = 24


def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt
    
    Args:
        password: Plain text password
    
    Returns:
        Hashed password
    """
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def verify_password(password: str, password_hash: str) -> bool:
    """
    Verify a password against its hash
    
    Args:
        password: Plain text password
        password_hash: Stored hash
    
    Returns:
        True if password matches, False otherwise
    """
    return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))


def create_session_token(client_id: str, email: str) -> Dict[str, str]:
    """
    Create a JWT session token
    
    Args:
        client_id: Client ID
        email: Client email
    
    Returns:
        Dict with token and expiration
    """
    expires_at = datetime.utcnow() + timedelta(hours=SESSION_DURATION_HOURS)
    
    payload = {
        "client_id": client_id,
        "email": email,
        "exp": expires_at,
        "iat": datetime.utcnow(),
        "type": "session"
    }
    
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    
    return {
        "token": token,
        "expires_at": expires_at.isoformat()
    }


def verify_session_token(token: str) -> Optional[Dict]:
    """
    Verify and decode a session token
    
    Args:
        token: JWT token
    
    Returns:
        Decoded payload if valid, None otherwise
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None  # Token expired
    except jwt.InvalidTokenError:
        return None  # Invalid token


def invalidate_existing_session(db: Session, client_id: str):
    """
    Invalidate any existing session for a client
    (Enforces single session per user)
    
    Args:
        db: Database session
        client_id: Client ID
    """
    client = db.query(Client).filter(Client.client_id == client_id).first()
    if client:
        client.current_session_token = None
        client.session_expires_at = None
        db.commit()


def create_new_session(db: Session, client: Client) -> Dict:
    """
    Create a new session for a client
    Invalidates any existing session
    
    Args:
        db: Database session
        client: Client object
    
    Returns:
        Session data
    """
    # Invalidate existing session (single session enforcement)
    invalidate_existing_session(db, client.client_id)
    
    # Create new session token
    session_data = create_session_token(client.client_id, client.email)
    
    # Update client record
    client.current_session_token = session_data['token']
    client.session_created_at = datetime.utcnow()
    client.session_expires_at = datetime.fromisoformat(session_data['expires_at'])
    client.last_login_at = datetime.utcnow()
    client.login_count = (client.login_count or 0) + 1
    
    db.commit()
    db.refresh(client)
    
    return {
        "session_token": session_data['token'],
        "expires_at": session_data['expires_at'],
        "client_id": client.client_id,
        "email": client.email,
        "company_name": client.company_name
    }


def verify_session(db: Session, session_token: str) -> Optional[Client]:
    """
    Verify a session token and return the client
    
    Args:
        db: Database session
        session_token: Session token to verify
    
    Returns:
        Client object if session is valid, None otherwise
    """
    # Verify token structure
    payload = verify_session_token(session_token)
    if not payload:
        return None
    
    # Find client with this session token
    client = db.query(Client).filter(
        Client.current_session_token == session_token,
        Client.is_active == True
    ).first()
    
    if not client:
        return None
    
    # Check if session has expired
    if client.session_expires_at and client.session_expires_at < datetime.utcnow():
        # Session expired, invalidate it
        invalidate_existing_session(db, client.client_id)
        return None
    
    return client