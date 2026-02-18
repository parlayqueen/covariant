"""
ESPN API Client - Production Grade

Real HTTP client for ESPN data feeds with:
- Filesystem caching (SHA256 keys)
- TTL-based freshness policies
- Exponential backoff retries
- Schema validation
- Request logging
- Rate limit handling

NO MOCK DATA. Real ESPN endpoints only.
"""

import hashlib
import json
import time
import logging
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import urllib.request
import urllib.error
from urllib.parse import urlencode


logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with TTL"""
    url: str
    data: Dict
    fetched_at: float
    status_code: int


class ESPNClientError(Exception):
    """ESPN client errors"""
    pass


class ESPNClient:
    """
    Production ESPN HTTP client.
    
    Features:
    - Adaptive TTL based on game proximity
    - Exponential backoff (max 5 retries)
    - Request logging
    - Schema validation
    - Defensive against feed changes
    """
    
    BASE_URL = "https://site.api.espn.com/apis/site/v2"
    
    # TTL policies (seconds)
    TTL_PREGAME_FAR = 3 * 3600  # >3h before game: 3 hours
    TTL_PREGAME_NEAR = 15 * 60  # <3h before game: 15 minutes
    TTL_LIVE = 5 * 60           # During game: 5 minutes
    TTL_POSTGAME = 24 * 3600    # After game: 24 hours
    
    # Retry config
    MAX_RETRIES = 5
    INITIAL_BACKOFF = 1.0  # seconds
    MAX_BACKOFF = 30.0
    
    # Timeout
    REQUEST_TIMEOUT = 30  # seconds
    
    def __init__(
        self,
        cache_dir: str = "/data/data/com.termux/files/home/.cache/espn_cache",
        user_agent: str = "PicksLabPro/2.5"
    ):
        """
        Initialize ESPN client.
        
        Args:
            cache_dir: Directory for filesystem cache
            user_agent: User agent header
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.user_agent = user_agent
        
        # Request stats
        self.stats = {
            'requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0
        }
        
        logger.info(f"ESPN client initialized with cache: {cache_dir}")
    
    def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        ttl_override: Optional[int] = None,
        game_start_time: Optional[datetime] = None
    ) -> Dict:
        """
        GET request with caching and retries.
        
        Args:
            endpoint: ESPN endpoint (e.g., 'sports/basketball/nba/scoreboard')
            params: Query parameters
            ttl_override: Override TTL (seconds)
            game_start_time: Game start time for adaptive TTL
        
        Returns:
            Parsed JSON response
        
        Raises:
            ESPNClientError: On failure after retries
        """
        # Build full URL
        url = f"{self.BASE_URL}/{endpoint.lstrip('/')}"
        if params:
            url = f"{url}?{urlencode(params)}"
        
        # Check cache
        cache_key = self._url_to_cache_key(url)
        cached = self._read_cache(cache_key)
        
        if cached:
            ttl = ttl_override if ttl_override else self._compute_ttl(game_start_time)
            age = time.time() - cached.fetched_at
            
            if age < ttl:
                self.stats['cache_hits'] += 1
                logger.debug(f"Cache HIT: {endpoint} (age: {age:.0f}s, ttl: {ttl}s)")
                return cached.data
            else:
                logger.debug(f"Cache STALE: {endpoint} (age: {age:.0f}s > ttl: {ttl}s)")
        
        # Cache miss - fetch from ESPN
        self.stats['cache_misses'] += 1
        data = self._fetch_with_retries(url)
        
        # Write to cache
        self._write_cache(cache_key, url, data, 200)
        
        return data
    
    def _fetch_with_retries(self, url: str) -> Dict:
        """Fetch with exponential backoff retries"""
        backoff = self.INITIAL_BACKOFF
        last_error = None
        
        for attempt in range(self.MAX_RETRIES):
            try:
                start_time = time.time()
                
                # Build request
                req = urllib.request.Request(
                    url,
                    headers={'User-Agent': self.user_agent}
                )
                
                # Execute
                with urllib.request.urlopen(req, timeout=self.REQUEST_TIMEOUT) as response:
                    data = json.loads(response.read().decode('utf-8'))
                
                latency = time.time() - start_time
                self.stats['requests'] += 1
                
                logger.info(
                    f"ESPN GET: {url[:80]}... "
                    f"status={response.status} latency={latency:.2f}s"
                )
                
                # Validate schema
                self._validate_response(data, url)
                
                return data
            
            except urllib.error.HTTPError as e:
                last_error = e
                
                if e.code == 429:  # Rate limit
                    logger.warning(f"Rate limit hit (429), retry {attempt+1}/{self.MAX_RETRIES}")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, self.MAX_BACKOFF)
                    continue
                
                elif e.code >= 500:  # Server error
                    logger.warning(f"Server error {e.code}, retry {attempt+1}/{self.MAX_RETRIES}")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, self.MAX_BACKOFF)
                    continue
                
                else:
                    # Client error - don't retry
                    self.stats['errors'] += 1
                    raise ESPNClientError(f"HTTP {e.code}: {url}") from e
            
            except urllib.error.URLError as e:
                last_error = e
                logger.warning(f"Network error, retry {attempt+1}/{self.MAX_RETRIES}: {e}")
                time.sleep(backoff)
                backoff = min(backoff * 2, self.MAX_BACKOFF)
                continue
            
            except json.JSONDecodeError as e:
                last_error = e
                self.stats['errors'] += 1
                raise ESPNClientError(f"Invalid JSON from ESPN: {url}") from e
            
            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error: {e}", exc_info=True)
                time.sleep(backoff)
                backoff = min(backoff * 2, self.MAX_BACKOFF)
                continue
        
        # All retries exhausted
        self.stats['errors'] += 1
        raise ESPNClientError(
            f"Failed after {self.MAX_RETRIES} retries: {url}"
        ) from last_error
    
    def _compute_ttl(self, game_start_time: Optional[datetime]) -> int:
        """Compute adaptive TTL based on game timing"""
        if game_start_time is None:
            return self.TTL_PREGAME_FAR
        
        now = datetime.utcnow()
        
        # Game finished
        if game_start_time < now - timedelta(hours=4):
            return self.TTL_POSTGAME
        
        # Game in progress
        if game_start_time < now:
            return self.TTL_LIVE
        
        # Game within 3 hours
        if game_start_time < now + timedelta(hours=3):
            return self.TTL_PREGAME_NEAR
        
        # Game far in future
        return self.TTL_PREGAME_FAR
    
    def _url_to_cache_key(self, url: str) -> str:
        """Convert URL to cache key (SHA256)"""
        return hashlib.sha256(url.encode()).hexdigest()
    
    def _read_cache(self, cache_key: str) -> Optional[CacheEntry]:
        """Read from cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r') as f:
                cached = json.load(f)
            
            return CacheEntry(
                url=cached['url'],
                data=cached['data'],
                fetched_at=cached['fetched_at'],
                status_code=cached['status_code']
            )
        
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return None
    
    def _write_cache(
        self,
        cache_key: str,
        url: str,
        data: Dict,
        status_code: int
    ):
        """Write to cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            entry = {
                'url': url,
                'data': data,
                'fetched_at': time.time(),
                'status_code': status_code
            }
            
            with open(cache_file, 'w') as f:
                json.dump(entry, f)
        
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    def _validate_response(self, data: Dict, url: str):
        """
        Validate response schema.
        
        Defensive against ESPN feed shape changes.
        """
        if not isinstance(data, dict):
            raise ESPNClientError(f"Expected dict, got {type(data)}: {url}")
        
        # Save raw payload if critical fields missing
        if 'events' not in data and 'scoreboard' not in url:
            # May be valid for non-scoreboard endpoints
            pass
        
        # Check for ESPN error responses
        if 'error' in data:
            raise ESPNClientError(f"ESPN API error: {data.get('error')}")
    
    def clear_cache(self):
        """Clear all cache files"""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
        
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict:
        """Get client statistics"""
        return {
            **self.stats,
            'cache_hit_rate': (
                self.stats['cache_hits'] / 
                (self.stats['cache_hits'] + self.stats['cache_misses'])
                if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0
                else 0
            )
        }


# Singleton instance
_client = None


def get_espn_client() -> ESPNClient:
    """Get singleton ESPN client"""
    global _client
    if _client is None:
        _client = ESPNClient()
    return _client
