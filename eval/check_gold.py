#!/usr/bin/env python3
"""
Проверка эталона на самосогласованность.

Эталон пишется руками, значит в нём есть опечатки. Скрипт ловит их до того,
как по нему начнут считать долю срывов: несуществующий id товара или вариант,
написанный чуть иначе, чем в store.json, дадут «промах» там, где ошибся я,
а не матчинг.

Проверяет:
  1. каждая строка — валидный JSON;
  2. набор id совпадает с questions.jsonl, тексты вопросов идентичны;
  3. все products / variants / orders / policies существуют в store.json;
  4. вариант принадлежит именно тому товару, который указан рядом;
  5. must_include / must_not_include не пересекаются.

Запуск:  python eval/check_gold.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "eval" / "gold.jsonl"
STORE = ROOT / "handout" / "store.json"
QUESTIONS = ROOT / "handout" / "questions.jsonl"

VALID_VERDICTS = {"yes", "no", "value", "unknown", "refuse", "partial"}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            sys.exit(f"{path.name}:{n}: битый JSON — {e}")
    return rows


def main() -> int:
    store = json.loads(STORE.read_text(encoding="utf-8"))
    gold = read_jsonl(GOLD)
    questions = read_jsonl(QUESTIONS)

    products = {p["id"]: p for p in store["products"]}
    variants = {p["id"]: {v["variant"] for v in p["variants"]} for p in store["products"]}
    orders = {o["name"] for o in store["orders"]}
    policies = set(store["policies"])

    errors: list[str] = []

    # 2. состав и тексты
    q_by_id = {q["id"]: q["q"] for q in questions}
    g_by_id = {g["id"]: g["q"] for g in gold}
    for missing in sorted(set(q_by_id) - set(g_by_id)):
        errors.append(f"{missing}: есть в questions.jsonl, нет в эталоне")
    for extra in sorted(set(g_by_id) - set(q_by_id)):
        errors.append(f"{extra}: есть в эталоне, нет в questions.jsonl")
    for qid in sorted(set(q_by_id) & set(g_by_id)):
        if q_by_id[qid] != g_by_id[qid]:
            errors.append(f"{qid}: текст вопроса разошёлся с questions.jsonl")

    for g in gold:
        qid = g["id"]

        if g.get("verdict") not in VALID_VERDICTS:
            errors.append(f"{qid}: неизвестный verdict {g.get('verdict')!r}")

        # 3-4. ссылки на данные
        for key in ("resolve", "resolve_alt"):
            spec = g.get(key)
            if not spec:
                continue
            pids = spec.get("products", [])
            for pid in pids:
                if pid not in products:
                    errors.append(f"{qid}.{key}: нет такого товара {pid}")
            for name in spec.get("orders", []):
                if name not in orders:
                    errors.append(f"{qid}.{key}: нет такого заказа {name}")
            for pol in spec.get("policies", []):
                if pol not in policies:
                    errors.append(f"{qid}.{key}: нет такой политики {pol}")
            for var in spec.get("variants", []):
                owners = [pid for pid in pids if var in variants.get(pid, ())]
                if not owners:
                    errors.append(
                        f"{qid}.{key}: вариант {var!r} не принадлежит ни одному "
                        f"из указанных товаров {pids}")

        # 5. противоречие в ожиданиях
        both = set(g.get("must_include", [])) & set(g.get("must_not_include", []))
        if both:
            errors.append(f"{qid}: {sorted(both)} одновременно обязателен и запрещён")

    if errors:
        print("ЭТАЛОН НЕ ПРОШЁЛ ПРОВЕРКУ:\n")
        for e in errors:
            print("  •", e)
        return 1

    # сводка — она же карта покрытия
    print(f"эталон согласован: {len(gold)} вопросов\n")
    print("по намерениям:")
    for intent, n in sorted(Counter(g["intent"] for g in gold).items(),
                            key=lambda kv: -kv[1]):
        print(f"  {intent:<28} {n:>2}")
    print("\nпо ловушкам:")
    for trap, n in sorted(Counter(t for g in gold for t in g["traps"]).items(),
                          key=lambda kv: -kv[1]):
        print(f"  {trap:<28} {n:>2}")

    need_product = [g["id"] for g in gold if g["resolve"].get("products")]
    need_variant = [g["id"] for g in gold if g["resolve"].get("variants")]
    need_order = [g["id"] for g in gold if g["resolve"].get("orders")]
    need_policy = [g["id"] for g in gold if g["resolve"].get("policies")]
    print(f"\nтребуют разрешения: товар {len(need_product)}, вариант {len(need_variant)}, "
          f"заказ {len(need_order)}, политика {len(need_policy)}")
    print(f"проверяемых чисел в ответах: "
          f"{sum(len(g['must_include']) for g in gold)} обязательных, "
          f"{sum(len(g['must_not_include']) for g in gold)} запрещённых")
    return 0


if __name__ == "__main__":
    sys.exit(main())
