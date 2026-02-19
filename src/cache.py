import redis
import json
import os
from functools import wraps
import hashlib

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

# Cache TTLs
CACHE_TTL = {
    "api_key_validation": 300,      # 5 minutes
    "usage_stats": 60,              # 1 minute
    "dashboard_summary": 120,        # 2 minutes
    "fairness_report": 3600,        # 1 hour
}


def cache_result(key_prefix, ttl=300):
    """Decorator to cache function results in Redis"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            cache_key = f"{key_prefix}:{hashlib.md5(str(args).encode() + str(kwargs).encode()).hexdigest()}"
            
            # Try to get from cache
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Store in cache
            redis_client.setex(cache_key, ttl, json.dumps(result))
            
            return result
        return wrapper
    return decorator


def increment_usage_counter(client_id: str):
    """Increment usage counter in Redis (fast)"""
    key = f"usage:{client_id}"
    count = redis_client.incr(key)
    
    # Sync to DB every 100 requests to reduce DB load
    if count % 100 == 0:
        from src.database import SessionLocal, Client
        db = SessionLocal()
        client = db.query(Client).filter(Client.client_id == client_id).first()
        if client:
            client.usage_count = count
            db.commit()
        db.close()
    
    return count


def get_usage_count(client_id: str) -> int:
    """Get current usage count from Redis"""
    key = f"usage:{client_id}"  # ← FIX: Added closing quote
    count = redis_client.get(key)
    return int(count) if count else 0


def cache_client_validation(api_key: str, client_data: dict):
    """Cache API key validation result"""
    key = f"auth:{api_key}"
    redis_client.setex(key, CACHE_TTL["api_key_validation"], json.dumps(client_data))


def get_cached_client(api_key: str):
    """Get cached client data"""
    key = f"auth:{api_key}"
    cached = redis_client.get(key)
    return json.loads(cached) if cached else None


def invalidate_cache(pattern: str):
    """Invalidate cache by pattern"""
    for key in redis_client.scan_iter(pattern):
        redis_client.delete(key)


# Rate limiting
def check_rate_limit(client_id: str, limit: int = 100, window: int = 60) -> bool:
    """
    Check if client has exceeded rate limit
    
    Args:
        client_id: Client identifier
        limit: Max requests allowed in window
        window: Time window in seconds
    
    Returns:
        True if under limit, False if exceeded
    """
    key = f"ratelimit:{client_id}"
    current = redis_client.incr(key)
    
    if current == 1:
        redis_client.expire(key, window)
    
    return current <= limit


# Connection health check
def check_redis_health() -> bool:
    """Check if Redis connection is healthy"""
    try:
        redis_client.ping()
        return True
    except:
        return False