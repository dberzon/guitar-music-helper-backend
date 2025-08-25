# SimpleTTLCache: A minimal TTL cache for FastAPI state
import threading
import time

class SimpleTTLCache:
    def __init__(self, maxsize=50, ttl=3600):
        self.maxsize = maxsize
        self.ttl = ttl
        self._lock = threading.Lock()
        self._data = {}
        self._expire = {}

    def set(self, key, value):
        with self._lock:
            now = time.time()
            self._data[key] = value
            self._expire[key] = now + self.ttl
            self._evict()

    def get(self, key):
        with self._lock:
            now = time.time()
            if key in self._data and self._expire[key] > now:
                return self._data[key]
            if key in self._data:
                del self._data[key]
                del self._expire[key]
            return None

    def _evict(self):
        # Remove expired
        now = time.time()
        expired = [k for k, exp in self._expire.items() if exp <= now]
        for k in expired:
            del self._data[k]
            del self._expire[k]
        # Remove oldest if over maxsize
        while len(self._data) > self.maxsize:
            oldest = min(self._expire, key=self._expire.get)
            del self._data[oldest]
            del self._expire[oldest]
