#!/usr/bin/env python3
"""
Какие лимиты провайдер реально даёт НАШЕЙ модели прямо сейчас.

Нужен не для красоты: тариф — это не то, что написано в счёте, а то, что
приходит в заголовках. Бесплатный тариф отдаёт 8000 токенов в минуту, и
прогон на большом файле упирается в них раньше, чем в задержку модели:
эскалации получают 429, превращаются в «не знаю», а p95 при этом выглядит
прекрасно — быстрый путь никуда не делся, а 429 возвращается мгновенно.
Поэтому после смены тарифа проверяем не кабинет, а ответ API.

    python eval/limits.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import http.client  # noqa: E402

from run import HOST, MODEL, PATH, USER_AGENT, read_key  # noqa: E402

# Сколько токенов в минуту нужно, чтобы прогон на 458 вопросах прошёл
# без троттлинга: 66 эскалаций × ~950 токенов промпта.
NEED_TPM = 47_000


def main() -> int:
    conn = http.client.HTTPSConnection(HOST, timeout=20)
    conn.request("POST", PATH, json.dumps({
        "model": MODEL, "max_tokens": 1, "stream": False,
        "reasoning_effort": "none",
        "messages": [{"role": "user", "content": "ping"}]}),
        {"Authorization": f"Bearer {read_key()}",
         "Content-Type": "application/json",
         "User-Agent": USER_AGENT})
    r = conn.getresponse()
    r.read()

    h = {k.lower(): v for k, v in r.getheaders()}
    print(f"модель {MODEL}, HTTP {r.status}")
    for k in ("x-ratelimit-limit-requests", "x-ratelimit-limit-tokens",
              "x-ratelimit-remaining-requests", "x-ratelimit-remaining-tokens",
              "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
        if k in h:
            print(f"  {k:34} {h[k]}")

    tpm = h.get("x-ratelimit-limit-tokens", "")
    n = None
    if tpm.upper().endswith("K"):
        n = float(tpm[:-1]) * 1000
    elif tpm.isdigit():
        n = float(tpm)
    if n is not None:
        ok = n >= NEED_TPM
        print(f"\n  на прогон 458 вопросов нужно ~{NEED_TPM:,} токенов в минуту → "
              + ("ХВАТАЕТ" if ok else
                 f"НЕ ХВАТАЕТ, троттлинг ~{NEED_TPM / n:.0f} мин"))
    return 0 if r.status == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())
