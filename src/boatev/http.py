"""間隔を空けた取得＋過去日付はディスクにキャッシュ。"""
from __future__ import annotations

import datetime as dt
import hashlib
import time

import requests

from . import config

_session = requests.Session()
_session.headers["User-Agent"] = config.USER_AGENT
_last = 0.0
STATS = {"requests": 0, "errors": 0, "seconds": 0.0, "cache_hits": 0, "last_error": None}


def _wait():
    global _last
    gap = time.monotonic() - _last
    if gap < config.REQUEST_INTERVAL_SEC:
        time.sleep(config.REQUEST_INTERVAL_SEC - gap)
    _last = time.monotonic()


def get(url: str, *, binary: bool = False, cache: bool = False, retries: int = 3):
    """cache=True は確定済み（過去日）のページだけに使う。"""
    path = config.CACHE_DIR / hashlib.sha1(url.encode()).hexdigest()
    if cache and path.exists():
        STATS["cache_hits"] += 1
        data = path.read_bytes()
        return data if binary else data.decode("utf-8")
    for attempt in range(retries):
        _wait()
        t = time.monotonic()
        STATS["requests"] += 1
        try:
            r = _session.get(url, timeout=20)
            STATS["seconds"] += time.monotonic() - t
            if r.status_code == 404:
                return None
            r.raise_for_status()
            if cache:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(r.content)
            if binary:
                return r.content
            if r.encoding in (None, "ISO-8859-1"):  # 文字コード自動判定は遅いので、公式サイトのUTF-8を既定に
                r.encoding = "utf-8"
            return r.text
        except requests.RequestException as e:
            STATS["seconds"] += time.monotonic() - t
            STATS["errors"] += 1
            STATS["last_error"] = f"{type(e).__name__}: {str(e)[:200]} ({url})"
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt * 3)
    return None


def is_past(date: str) -> bool:
    jst = dt.timezone(dt.timedelta(hours=9))
    return date < dt.datetime.now(jst).strftime("%Y%m%d")
