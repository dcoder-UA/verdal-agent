#!/usr/bin/env python3
"""
Замер четвёртого исхода — передачи человеку.

Зачем отдельный файл, а не строки в `generated.jsonl`: порождённый набор
из 458 вопросов НЕ СОДЕРЖИТ этого класса вовсе — там вопросы покупателя
о товаре, заказе и политиках, а не просьбы что-то изменить или позвать
оператора. Прогон по нему даёт ровно ноль передач, и это полезно как
проверка ложных срабатываний, но промахи он не измеряет никак.

Поэтому здесь два списка в одном файле:
  expect=handoff — должно уйти к оператору;
  expect=keep    — граница, которую правка не должна была сдвинуть
                   (политики возврата и обмена, статус и адрес заказа).

Второй список важнее первого. Пропущенная передача стоит одного плохого
ответа, а лишняя — целого класса вопросов, на которые у магазина есть
готовая политика.

    python eval/handoff_probe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from resolver.answer import Answerer          # noqa: E402
from resolver.resolve import load             # noqa: E402


def main() -> int:
    r = load()
    a = Answerer(json.loads((ROOT / "handout" / "store.json").read_text(encoding="utf-8")))
    rows = [json.loads(l) for l in
            (ROOT / "eval" / "handoff.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]

    miss, false_hit = [], []
    n_handoff = n_keep = 0
    for g in rows:
        res = r.resolve(g["q"])
        rep = a.reply(res, g["q"])
        got = rep.decision == "handoff"
        if g["expect"] == "handoff":
            n_handoff += 1
            if not got:
                miss.append((g, res.intent, rep.decision, rep.text))
        else:
            n_keep += 1
            if got:
                false_hit.append((g, res.intent, rep.decision, rep.text))

    for title, rows_ in (("ПРОПУЩЕНА ПЕРЕДАЧА", miss),
                         ("🔴 ЛИШНЯЯ ПЕРЕДАЧА", false_hit)):
        if rows_:
            print(f"\n{title}:")
            for g, intent, dec, text in rows_:
                print(f"  {g['id']} {g['q'][:52]:54} {intent}/{dec}")
                print(f"      {g['why']}")

    print("\n" + "=" * 74)
    print(f"должны уйти к оператору  {n_handoff - len(miss):2}/{n_handoff}")
    print(f"должны остаться как были {n_keep - len(false_hit):2}/{n_keep}  (ложные передачи)")
    print("=" * 74)
    return 1 if (miss or false_hit) else 0


if __name__ == "__main__":
    raise SystemExit(main())
