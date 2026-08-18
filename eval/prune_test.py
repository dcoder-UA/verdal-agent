#!/usr/bin/env python3
"""
Проверка гипотезы прунинга: что будет, если быстрый путь отвечает ТОЛЬКО
на вопросы, объяснённые целиком, а всё остальное уходит в модель.

Правило покрытия (общее, а не под конкретный дефект): вопрос идёт по быстрому
пути, если каждое содержательное слово в нём объяснено разбором — названием
товара, цветом, размером, числом, ключевым словом намерения или служебным
словом языка. Осталось необъяснённое слово — значит в вопросе есть то, чего
мы не поняли, и отвечать самим нельзя.

Меряем главное: сколько вопросов эталона, отданных модели, ПО-ПРЕЖНЕМУ
проходят проверку по обязательным и запрещённым числам. Если модель их
держит — прунинг бесплатен. Если сыпется — видно, где именно.

Ожидание 429 из тайминга вырезано: таймер перезапускается после паузы,
иначе мерили бы квоту бесплатного тарифа, а не модель.

Запуск:  python eval/prune_test.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from resolver.answer import Answerer                                  # noqa: E402
from resolver.normalize import key, similar, words                    # noqa: E402
from resolver.resolve import (FUNCTION_WORDS, INTENT_RULES,           # noqa: E402
                              ORDER_RULES, load)
import run as runner                                                  # noqa: E402

# Слова языка, а не каталога: вопросительные, местоимения, обиходные глаголы.
# Список закрытый и от данных не зависит — это та же природа, что FUNCTION_WORDS.
COMMON = """
скільки який яка яке які чи де коли хто що можна маєте дайте дай скажи будь
ласка мені моє мій моя ваш ваша ваші вас вам ще вже штук пар пара набір
розмір размер лишилось гроші річ речі якщо плюс разом стандартна стандартний
платить платити вийде йде візьму беру брати взяти купити замовити хочу треба
потрібно машинка машинці прати носити мати бути робити зробити
is the my your you do does what where when how much many in of have a an it i
me and or to for on at with there are can left size order please tell give
""".split()


def build_vocab(r):
    kw = set()
    for _, kws in INTENT_RULES + ORDER_RULES:
        for k in kws:
            kw.update(key(w) for w in k.removeprefix("=").split())
    v = set(r.ua2en) | set(r.ua2color) | kw | FUNCTION_WORDS | set(r.en_tokens)
    if "--strict" not in sys.argv:
        v |= {key(w) for w in COMMON}
    for _, ph in r.phrases:
        v |= set(ph)
    return v


def unexplained(q: str, vocab: set[str]) -> list[str]:
    out = []
    for w in words(q):
        parts = [w] + w.split("-") if "-" in w else [w]
        if any(_known(key(p), vocab) for p in parts):
            continue
        out.append(w)
    return out


def _known(k: str, vocab: set[str]) -> bool:
    if len(k) < 3 or re.fullmatch(r"[\d\-]+[a-zа-яіїєґ]*", k):
        return True
    return k in vocab or any(similar(k, v) >= 0.85 for v in vocab)


def wait_seconds(text: str) -> float:
    """Groq пишет паузу и как «7.2s», и как «36m43.2s» — читаем оба формата."""
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", text)
    if not m:
        return 20.0
    return float(m.group(1) or 0) * 60 + float(m.group(2))


def main() -> int:
    strict = "--strict" in sys.argv      # без списка обиходных слов — жёсткий прунинг
    store = json.loads((ROOT / "handout" / "store.json").read_text(encoding="utf-8"))
    r = load()
    a = Answerer(store)
    vocab = build_vocab(r)
    gold = [json.loads(l) for l in
            (ROOT / "eval" / "gold.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    model = runner.Model(runner.read_key())
    model.probe()

    rows, waited = [], 0.0
    for g in gold:
        q = g["q"]
        res = r.resolve(q)
        rep = a.reply(res, q)
        u = unexplained(q, vocab)
        # Правило покрытия поверх существующего контракта.
        escalate = rep.decision == "escalate" or bool(u)

        if not escalate:
            rows.append((g, "шаблон", rep.text, 0.0, u))
            continue
        if rep.decision == "abstain":
            rows.append((g, "отказ", rep.text, 0.0, u))
            continue

        data = runner.build_slice(store, res)
        text, ms = None, 0.0
        for _ in range(6):
            t0 = time.perf_counter()
            resp = model._post({
                "model": runner.MODEL, "max_tokens": runner.MAX_TOKENS,
                "stream": False, "reasoning_effort": "none",
                "messages": [{"role": "system", "content": runner.SYSTEM + "\n\nДАНІ:\n" + data},
                             {"role": "user", "content": q}]},
                timeout=15.0)          # щедро: сейчас меряем правильность
            if resp is not None and resp.status_code == 429:
                pause = wait_seconds(resp.text)
                print(f"    [429, жду {pause:.0f} с]", flush=True)
                time.sleep(pause + 1)
                waited += pause + 1
                continue               # таймер начнётся заново — пауза не в тайминге
            ms = (time.perf_counter() - t0) * 1000
            if resp is None:
                break
            if resp.status_code != 200:
                print(f"  ! HTTP {resp.status_code}: {resp.text[:120]}")
                break
            text = (resp.json()["choices"][0]["message"].get("content") or "").strip()
            break
        rows.append((g, "модель", text or "[нет ответа]", ms, u))

    # --- разбор результата ---
    ok_inc = ok_exc = 0
    bad = []
    counts = {"шаблон": 0, "отказ": 0, "модель": 0}
    lat = []
    for g, path, text, ms, u in rows:
        counts[path] += 1
        if path == "модель":
            lat.append(ms)
        miss = [x for x in g["must_include"] if x not in text]
        hit = [x for x in g["must_not_include"] if x in text]
        ok_inc += not miss
        ok_exc += not hit
        if (miss or hit) :
            bad.append((g["id"], path, g["q"], miss, hit, text, u))

    n = len(rows)
    print("\n" + "=" * 78)
    print(f"путь:   шаблон {counts['шаблон']}   отказ {counts['отказ']}   "
          f"модель {counts['модель']}   из {n}")
    print(f"обязательные числа  {ok_inc}/{n}")
    print(f"запрещённые числа   {ok_exc}/{n}   (ложные совпадения)")
    if lat:
        lat.sort()
        import statistics
        p95 = lat[max(0, int(0.95 * len(lat)) - 1)]
        print(f"задержка модели (без ожидания 429): n={len(lat)} "
              f"p50={statistics.median(lat):.0f} p95={p95:.0f} max={lat[-1]:.0f} мс")
        print(f"из них уложились бы в таймаут 450 мс: "
              f"{sum(1 for x in lat if x <= 450)}/{len(lat)}")
    print(f"суммарное ожидание 429, вне тайминга: {waited:.0f} с")
    print("=" * 78)
    print("\nушли в модель:")
    for g, path, text, ms, u in rows:
        if path == "модель":
            print(f"  {g['id']} {g['q'][:44]:46} {u}")
    if bad:
        print("\nПРОВАЛЫ:")
        for i, path, q, miss, hit, text, u in bad:
            print(f"  {i} [{path}] {q[:44]}")
            if miss: print(f"      нет обязательного: {miss}")
            if hit:  print(f"      🔴 есть запрещённое: {hit}")
            print(f"      необъяснённые слова: {u}")
            print(f"      ответ: {text[:150]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
