"""公開デモ用の簡易回数制限（IP ごと・1日あたり）。

コンテナ1台のメモリで数えるだけの簡易版。複数台に増やす場合は Redis などに置き換える。
"""

from __future__ import annotations

import threading
from datetime import date


class DailyRateLimiter:
    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._day = date.today()
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def hit(self, key: str) -> bool:
        """1回分を数え、上限内なら True を返す。"""
        if self._limit <= 0:
            return True
        with self._lock:
            today = date.today()
            if today != self._day:
                self._day, self._counts = today, {}
            count = self._counts.get(key, 0)
            if count >= self._limit:
                return False
            self._counts[key] = count + 1
            return True
