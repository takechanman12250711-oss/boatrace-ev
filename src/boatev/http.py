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
        data = path.read_bytes()
        return data if binary else data.decode("utf-8")
    for attempt in range(retries):
        _wait()
        try:
            r = _session.get(url, timeout=20)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            if cache:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(r.content)
            if binary:
                return r.content
            r.encoding = r.apparent_encoding if r.encoding in (None, "ISO-8859-1") else r.encoding
            return r.text
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt * 3)
    return None


def is_past(date: str) -> bool:
    jst = dt.timezone(dt.timedelta(hours=9))
    return date < dt.datetime.now(jst).strftime("%Y%m%d")
