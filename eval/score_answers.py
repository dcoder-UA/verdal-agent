#!/usr/bin/env python3
"""
Проверка ТЕКСТА ответа, а не только разбора.

Эталон хранит два списка на вопрос: числа, которые в ответе обязаны быть,
и числа, которых быть не должно. Второй список важнее первого: он ловит
ложные совпадения. Если на «скільки коштує светр із мериносу» в ответе
окажется 69 — значит матчинг взял Merino Long Sleeve вместо Merino Wool
Sweater, и никакой другой проверкой это не видно.

Запуск:  python eval/score_answers.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from resolver.answer import Answerer          # noqa: E402
from resolver.resolve import load             # noqa: E402

r = load()
a = Answerer(json.loads((ROOT / "handout" / "store.json").read_text(encoding="utf-8")))

gold = [json.loads(l) for l in (ROOT / "eval" / "gold.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

dec = Counter()
bad_inc, bad_exc = [], []
for g in gold:
    rep = a.reply(r.resolve(g["q"]), g["q"])
    dec[rep.decision] += 1
    if rep.decision != "answer":
        continue
    miss = [x for x in g["must_include"] if x not in rep.text]
    hit = [x for x in g["must_not_include"] if x in rep.text]
    if miss:
        bad_inc.append((g["id"], g["q"], miss, rep.text))
    if hit:
        bad_exc.append((g["id"], g["q"], hit, rep.text))

for title, rows in (("НЕТ ОБЯЗАТЕЛЬНОГО", bad_inc), ("🔴 ЕСТЬ ЗАПРЕЩЁННОЕ", bad_exc)):
    if rows:
        print(f"\n{title}:")
        for i, q, w, t in rows:
            print(f"  {i} {q[:44]:46} {w}\n      {t[:110]}")

n = len(gold)
print("\n" + "=" * 74)
for k in ("answer", "abstain", "escalate"):
    print(f"{k:9} {dec[k]:3}/{n}  {100*dec[k]/n:5.1f}%")
print(f"обязательные числа   {n - len(bad_inc)}/{n}")
print(f"запрещённые числа    {n - len(bad_exc)}/{n}  (ложные совпадения)")
print("=" * 74)
