#!/usr/bin/env python3
"""
Порождение расширенной выборки прямо из store.json.

Зачем. На 39 вопросах даже безошибочный результат даёт верхнюю границу 9%
при пороге 5% — то есть выборка не в состоянии подтвердить или опровергнуть
требование. Плюс резолвер писался, глядя на эти 39, значит замер на них
меряет мою память, а не работу системы.

Как. Каждый вопрос собирается из данных, поэтому правильный ответ известен
по построению и размечать руками нечего: спросили про p-007 — значит ждём
p-007. Покрытие идёт по каталогу, а не по моей фантазии: каждый товар,
каждый вариант, каждый заказ, каждая политика.

Честная граница метода. Формулировки пишу я, и они беднее живой речи.
Поэтому выборка хорошо ловит ПРОМАХИ ПО КАТАЛОГУ и ПУТАНИЦУ МЕЖДУ ТОВАРАМИ
и плохо ловит незнакомые обороты. Разнообразие речи проверяется на живых 39,
покрытие данных — здесь. Одно другое не заменяет.

Запуск:  python eval/generate.py [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "eval" / "generated.jsonl"

PRICE_T = ["Скільки коштує {x}?", "{x} — ціна?", "{x} почім?", "Ціна на {x}?",
           "Скільки за {x}?", "{x} скільки коштує?"]
STOCK_T = ["{x} розмір {s} є?", "Чи є {x} у розмірі {s}?", "{x} {s} — в наявності?",
           "{x}, {s} — є?"]
STOCK_C_T = ["{x} {c} розмір {s} є?", "Чи є {x} {s}, {c}?", "{x} {s} {c} — в наявності?"]
ORDER_T = {
    "order_status":  ["Що з замовленням {n}?", "{n} — статус?", "Де моє замовлення {n}?",
                      "Що там по {n}?"],
    "order_tracking": ["Дай трек-номер по {n}", "Трек по замовленню {n}?",
                       "Який номер відправлення в {n}?"],
    "order_items":   ["Що входить у замовлення {n}?", "Які товари в {n}?",
                      "Що там у замовленні {n}?"],
    "order_address": ["Скажи адресу доставки в замовленні {n}", "Куди їде {n}?"],
}
POLICY_T = {
    "shipping": ["Скільки коштує стандартна доставка?", "Від якої суми доставка безкоштовна?",
                 "Скільки йде експрес-доставка?", "Ви відправляєте у Швейцарію?",
                 "Скільки днів іде звичайна доставка?", "Скільки коштує експрес?"],
    "returns":  ["Чи можна повернути товар?", "Хто платить за зворотню пересилку?",
                 "Коли повернуться гроші?", "Скільки днів на повернення?",
                 "Чи можна повернути річ, якщо я її вже поносив?"],
    "exchange": ["А якщо не підійде розмір?", "Чи можна поміняти розмір?",
                 "Взяв не той розмір, що робити?", "Обмін розміру платний?"],
    "damaged":  ["Прийшов товар з дефектом.", "Прийшла куртка з плямою на рукаві.",
                 "Що робити, якщо річ бракована?", "Товар пошкоджений при доставці."],
    "payment":  ["Наложеним платежем можна?", "Коли з картки спишуть гроші?",
                 "Якими картками можна платити?", "Чи приймаєте PayPal?"],
}
RESTOCK_T = ["Коли завезуть {x} {s} знову?", "Коли буде {x} у розмірі {s} знову?"]

# Товары, которых в каталоге НЕТ, но о которых спрашивают в магазине снаряжения.
# Без них выборка слепа к самому дорогому виду ошибки: система уверенно называет
# цену на то, чего не существует. Порождать такие вопросы из store.json нельзя
# по построению, поэтому список задан прямо — это единственное место в генераторе,
# где что-то перечисляется руками, и иначе никак.
#
# Подбор осторожный: сюда не годится «термос» (Insulated Bottle 1L — это он и есть)
# и «лижні палиці» (слишком близко к Trekking Poles). Только заведомо отсутствующее.
ABSENT = [
    "намет", "спальний мішок", "спальник", "каремат", "велосипед",
    "сонцезахисні окуляри", "налобний ліхтар", "газовий пальник", "казанок",
    "гамак", "парасолька", "компас", "аптечка", "льодоруб", "сидушка туристична",
]
ABSENT_T = ["Скільки коштує {x}?", "{x} у вас є?", "Ціна на {x}?",
            "Чи є {x} в наявності?"]


def typo(s: str, rnd: random.Random) -> str:
    """Одна правдоподобная опечатка: пропуск или перестановка букв."""
    letters = [i for i, ch in enumerate(s) if ch.isalpha() and i > 0]
    if len(letters) < 4:
        return s
    i = rnd.choice(letters[1:-1])
    if rnd.random() < 0.5:
        return s[:i] + s[i + 1:]
    return s[:i] + s[i + 1] + s[i] + s[i + 2:]


def split_variant(raw: str) -> tuple[str, str]:
    if "/" in raw:
        a, b = raw.split("/", 1)
        return a.strip(), b.strip()
    return raw.strip(), ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rnd = random.Random(args.seed)

    store = json.loads((ROOT / "handout" / "store.json").read_text(encoding="utf-8"))
    aliases = json.loads((ROOT / "index" / "aliases.json").read_text(encoding="utf-8"))
    phrases = aliases["product_phrases"]
    colors = aliases["colors"]

    rows: list[dict] = []

    def add(q, intent, resolve, traps, note=""):
        rows.append({"id": f"g{len(rows)+1:04d}", "q": q, "intent": intent,
                     "resolve": resolve, "verdict": "value", "must_include": [],
                     "must_not_include": [], "traps": traps, "note": note})

    for p in store["products"]:
        pid = p["id"]
        forms = phrases.get(pid, [])[:3]

        # цена: каждый товар × несколько формулировок
        for i, form in enumerate(forms):
            for t in rnd.sample(PRICE_T, 3):
                add(t.format(x=form), "product_price", {"products": [pid]},
                    ["generated"])

        # наличие: каждый вариант товара
        if forms:
            form = forms[0]
            for v in p["variants"]:
                size, color = split_variant(v["variant"])
                if size == "one":
                    continue
                traps = ["generated"] + (["zero_stock"] if v["stock"] == 0 else [])
                if color and colors.get(color):
                    ua_c = colors[color][0]
                    add(rnd.choice(STOCK_C_T).format(x=form, s=size, c=ua_c),
                        "product_stock",
                        {"products": [pid], "variants": [v["variant"]]}, traps)
                add(rnd.choice(STOCK_T).format(x=form, s=size), "product_stock",
                    {"products": [pid], "variants": [v["variant"]]}, traps)

            # рестоки: спрашиваем ровно про нулевые остатки
            for v in p["variants"]:
                if v["stock"] == 0:
                    size, _ = split_variant(v["variant"])
                    add(rnd.choice(RESTOCK_T).format(x=form, s=size),
                        "product_restock",
                        {"products": [pid], "variants": [v["variant"]]},
                        ["generated", "not_in_data", "hallucination"])

    # товары, которых нет: правильный ответ — «такого не тримаємо», а не цена
    # похожего. Пустой список products означает «не должен разрешиться ни во что».
    for name in ABSENT:
        for t in rnd.sample(ABSENT_T, 2):
            # намерение берём из самого шаблона, а не ставим одно на всех:
            # «намет у вас є?» — вопрос о наличии, «Ціна на намет?» — о цене
            # «є» ищем как отдельное слово: буква «є» сидит внутри «коштує»,
            # и проверка на вхождение помечала вопрос о цене вопросом о наличии
            intent = ("product_stock" if (" є?" in t or "наявності" in t)
                      else "product_price")
            add(t.format(x=name), intent, {"products": []},
                ["generated", "nonexistent_entity", "hallucination"],
                "товара нет в каталоге")

    # заказы: каждый заказ × каждое намерение
    for o in store["orders"]:
        for intent, templates in ORDER_T.items():
            add(rnd.choice(templates).format(n=o["name"]), intent,
                {"orders": [o["name"]]}, ["generated"])

    # несуществующие заказы
    known = {o["name"] for o in store["orders"]}
    made = 0
    while made < 12:
        n = f"#{rnd.randint(1000, 9999)}"
        if n in known:
            continue
        add(rnd.choice(ORDER_T["order_status"]).format(n=n), "order_status",
            {"orders": []}, ["generated", "nonexistent_entity", "hallucination"])
        made += 1

    # политики
    for pol, templates in POLICY_T.items():
        for t in templates:
            add(t, f"policy_{pol}", {"policies": [pol]}, ["generated"])

    # опечатки: у пятой части вопросов портим одну букву
    clean = list(rows)
    for r in rnd.sample(clean, k=len(clean) // 5):
        q = typo(r["q"], rnd)
        if q != r["q"]:
            add(q, r["intent"], r["resolve"], r["traps"] + ["typo"], "опечатка")

    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                   encoding="utf-8")

    from collections import Counter
    by = Counter(r["intent"] for r in rows)
    print(f"порождено {len(rows)} вопросов → {OUT}\n")
    for k, v in sorted(by.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<20} {v:>4}")
    print(f"\n  из них с опечаткой: {sum('typo' in r['traps'] for r in rows)}")
    print(f"  про нулевой остаток: {sum('zero_stock' in r['traps'] for r in rows)}")
    print(f"  ловушек на галлюцинацию: {sum('hallucination' in r['traps'] for r in rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
