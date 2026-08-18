#!/usr/bin/env python3
"""
Сколько вопросов детерминированный путь закрывает сам, а сколько отдаёт модели.

Считаем ТРИ исхода раздельно, потому что стоят они разного:

  ПРОМАХ            сущность не найдена → вопрос уходит в LLM.
                    Стоит задержки. Это и есть «доля срывов».

  ЛОЖНОЕ СОВПАДЕНИЕ найдена НЕ ТА сущность → быстрый неправильный ответ.
                    В долю срывов не попадает вовсе, а стоит правильности.
                    Опаснее промаха: промах виден, ложное совпадение — нет.

  НЕОДНОЗНАЧНОСТЬ   несколько кандидатов с близким весом. Лечится уточняющим
                    вопросом или отправкой в LLM — решаем отдельно.

Доля считается с доверительным интервалом: на 39 вопросах точечная оценка
почти ничего не значит, и делать вид, что значит, — самообман.

Запуск:  python eval/score.py [-v]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from resolver.resolve import load  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "eval" / "gold.jsonl"

OK, MISS, FALSE, AMBIG = "OK", "ПРОМАХ", "ЛОЖНОЕ", "НЕОДНОЗН"

# Резолвер не различает «товар + доставка» и «доставка», это делает шаблон
# ответа. Для оценки намерения считаем их одним классом.
INTENT_CANON = {
    "product_price_plus_shipping": "product+shipping",
    "policy_shipping_calc": "product+shipping",
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Доверительный интервал для доли. При k=0 даёт честную верхнюю границу."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z / d * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, c - m), min(1.0, c + m)


def canon(intent: str, has_product: bool) -> str:
    if intent == "policy_shipping" and has_product:
        return "product+shipping"
    return INTENT_CANON.get(intent, intent)


def judge(g: dict, r) -> tuple[str, str]:
    """Один вопрос → исход и пояснение."""
    want = g["resolve"]
    alt = g.get("resolve_alt") or {}

    # заказ
    if g["intent"].startswith("order"):
        want_order = (want.get("orders") or [None])[0]
        if want_order is None:
            return (FALSE, f"заказа не существует, а найден {r.order}") \
                if r.order else (OK, "несуществующий заказ распознан")
        if r.order is None:
            return MISS, "номер заказа не найден"
        return (OK, "") if r.order == want_order else (FALSE, f"{r.order} вместо {want_order}")

    # вне области
    if g["intent"] == "out_of_scope":
        return (OK, "") if r.intent == "out_of_scope" else (MISS, f"принято за {r.intent}")

    # товар
    want_p = want.get("products") or []
    # Пустой список products в разметке — это не «всё равно», а требование:
    # такого товара в каталоге нет, и назвать вместо него похожий значит
    # соврать покупателю уверенно и быстро.
    if "products" in want and not want_p:
        if not r.products:
            return OK, "отсутствие товара распознано"
        return FALSE, f"назван {r.products[0]}, а такого товара нет"

    if want_p:
        if not r.products:
            return MISS, "товар не опознан"
        if g["intent"] == "product_compare":
            return (OK, "") if set(r.products) == set(want_p) else \
                (FALSE, f"{r.products} вместо {want_p}")
        # Флаг неоднозначности проверяем ДО верного/неверного товара: система
        # сама сказала «не уверен», значит вопрос уходит в модель и уверенного
        # неверного ответа не будет. Это ужесточает главную метрику — такие
        # случаи начинают считаться срывами, — но описывает поведение честно.
        if r.ambiguous:
            return AMBIG, f"близкие кандидаты: {r.ambiguous}"
        if r.products[0] not in want_p:
            return FALSE, f"{r.products[0]} вместо {want_p[0]}"
        want_v = (want.get("variants") or [None])[0]
        if want_v:
            # Засчитываем попадание в список, а не единственный вариант: если
            # покупатель не назвал цвет, требовать угадать его не за что —
            # ответ строится по всем подходящим вариантам сразу.
            if not r.variants:
                return MISS, f"вариант не найден (нужен {want_v})"
            if want_v not in r.variants:
                return FALSE, f"варианты {r.variants} вместо {want_v}"
        return OK, ""

    # политика
    want_pol = set(want.get("policies") or []) | set(alt.get("policies") or [])
    if want_pol:
        if r.policy is None:
            return MISS, f"политика не определена (нужна {sorted(want_pol)})"
        return (OK, "") if r.policy in want_pol else \
            (FALSE, f"{r.policy} вместо {sorted(want_pol)}")

    return MISS, "эталон ничего не требует — проверь разметку"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true", help="показать все вопросы")
    ap.add_argument("--set", default="gold", help="gold | generated | путь к .jsonl")
    ap.add_argument("--limit", type=int, default=40, help="сколько провалов показать")
    args = ap.parse_args()

    path = {"gold": GOLD, "generated": ROOT / "eval" / "generated.jsonl"}.get(
        args.set, Path(args.set))
    gold = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    r = load()

    rows, intent_hits = [], 0
    for g in gold:
        res = r.resolve(g["q"])
        verdict, why = judge(g, res)
        ok_intent = canon(res.intent, bool(res.products)) == \
            canon(g["intent"], bool(g["resolve"].get("products")))
        intent_hits += ok_intent
        rows.append((g, res, verdict, why, ok_intent))

    n = len(rows)
    tally = Counter(v for _, _, v, _, _ in rows)

    if args.verbose:
        for g, res, verdict, why, ok_i in rows:
            mark = " " if verdict == OK else "!"
            print(f"{mark} {g['id']}  {verdict:<9} {g['q'][:40]:<42} "
                  f"{res.intent:<17}{'' if ok_i else ' ← намерение'} {why}")
    else:
        shown = 0
        for g, res, verdict, why, ok_i in rows:
            if verdict != OK or not ok_i:
                shown += 1
                if shown > args.limit:
                    continue
                print(f"! {g['id']}  {verdict:<9} {g['q'][:44]:<46} "
                      f"{res.intent:<17}{'' if ok_i else ' ← намерение'} {why}")
        if shown > args.limit:
            print(f"  … и ещё {shown - args.limit} (весь список: -v)")

    print("\n" + "=" * 74)
    for name in (OK, MISS, FALSE, AMBIG):
        k = tally[name]
        lo, hi = wilson(k, n)
        print(f"{name:<10} {k:>3}/{n}  {k/n*100:>5.1f}%   "
              f"95% ДИ {lo*100:>4.1f}–{hi*100:>4.1f}%")
    print(f"{'намерение':<10} {intent_hits:>3}/{n}  {intent_hits/n*100:>5.1f}%")
    print("=" * 74)

    miss, false = tally[MISS], tally[FALSE]
    lo, hi = wilson(miss + tally[AMBIG], n)
    print(f"\nв LLM уходит {miss + tally[AMBIG]}/{n} → доля срывов "
          f"{(miss+tally[AMBIG])/n*100:.1f}%, но интервал {lo*100:.1f}–{hi*100:.1f}%")
    if hi > 0.05:
        print(f"  ⚠ верхняя граница {hi*100:.1f}% выше порога 5% — выборки в {n} "
              f"вопросов НЕ ХВАТАЕТ, чтобы обосновать решение. Нужна расширенная.")
    if false:
        print(f"  🔴 ложных совпадений {false} — это быстрые неправильные ответы, "
              f"их не видно ни в одной метрике задержки")
    return 0


if __name__ == "__main__":
    sys.exit(main())
