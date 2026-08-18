#!/usr/bin/env python3
"""
Сборка таблицы украинских форм для каталога.  ЭТАП СБОРКИ, НЕ РАНТАЙМ.

Зачем: «светр» и «sweater» не имеют ни одной общей буквы, поэтому никакое
сравнение строк само по себе украинский вопрос с английским каталогом
не свяжет. Нужен мост между языками, и вопрос только в том, где он стоит.

Здесь он стоит на этапе сборки: модель один раз порождает украинские формы
для словаря каталога, результат ложится в index/aliases.json и коммитится.
В рантайме остаётся сравнение строк — миллисекунды, без сети и без ключей.
Ограничение 500 мс меряется от вопроса до ответа, сборка в него не входит.

Словарь берётся из store.json автоматически, руками ничего не перечисляется:
поменяется каталог — поменяется и словарь. Хеш store.json пишется в файл,
чтобы расхождение было видно, а не молчаливо.

Запуск:
    python build_aliases.py            # собрать (нужен ANTHROPIC_API_KEY)
    python build_aliases.py --vocab    # только показать словарь, без вызова
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STORE = ROOT / "handout" / "store.json"
OUT = ROOT / "index" / "aliases.json"
MODEL = "claude-sonnet-5"

# Слова английских названий, которые ничего не различают.
STOP = {"one", "and", "the", "of", "with"}


def load_store() -> dict:
    return json.loads(STORE.read_text(encoding="utf-8"))


def vocabulary(store: dict) -> tuple[list[str], list[str]]:
    """Из каталога вытаскиваем слова названий и названия цветов."""
    tokens: set[str] = set()
    for p in store["products"]:
        for w in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", p["title"].lower()):
            if w not in STOP:
                tokens.add(w)
    colors: set[str] = set()
    for p in store["products"]:
        for v in p["variants"]:
            if "/" in v["variant"]:
                colors.add(v["variant"].split("/", 1)[1].strip().lower())
    return sorted(tokens), sorted(colors)


PROMPT = """Ти складаєш словник для пошуку товарів в українському інтернет-магазині \
спорядження. Каталог англійською, покупці пишуть українською (часто з русизмами, \
скороченнями й помилками).

Каталог:
{catalog}

Кольори у варіантах: {colors}

Поверни СУВОРО JSON без пояснень, такої структури:

{{
  "tokens": {{
    "<англійське слово зі списку>": ["<українська форма>", ...]
  }},
  "colors": {{
    "<англійський колір зі списку>": ["<українська форма>", ...]
  }},
  "product_phrases": {{
    "<id товару>": ["<українська фраза, що називає товар цілком>", ...]
  }}
}}

Список англійських слів для "tokens": {tokens}

Вимоги:
- Для КОЖНОГО слова зі списку дай 3-8 українських форм у називному відмінку однини.
- Додавай синоніми, розмовні варіанти, поширені русизми ("светр", "кросівки",
  "рюкзак"), транслітерації ("софтшел", "флісовий") і часті помилки.
- У "product_phrases" клади ті фрази, які називають товар одним словом або
  сталим виразом і не збираються зі слів назви: наприклад термобілизна для
  Thermal Base Layer. Для кожного товару 2-6 фраз. Якщо товар нормально
  збирається зі слів — однаково дай 2-3 найчастіші фрази.
- Тільки українською. Не повторюй англійські слова у значеннях.
- Жодних коментарів, тільки JSON."""


def call_model(catalog: str, tokens: list[str], colors: list[str]) -> dict:
    import anthropic

    key = re.search(r"^\s*(?:export\s+)?ANTHROPIC_API_KEY\s*=\s*(.+)$",
                    (ROOT / ".env").read_text(encoding="utf-8"), re.M)
    if not key:
        sys.exit("нет ANTHROPIC_API_KEY в .env")
    client = anthropic.Anthropic(api_key=key.group(1).strip().strip("\"'"))
    prompt = PROMPT.format(catalog=catalog, tokens=", ".join(tokens),
                           colors=", ".join(colors))
    msg = client.messages.create(
        model=MODEL, max_tokens=8000,
        messages=[{"role": "user", "content": prompt}],
    )
    # у модели включено рассуждение, поэтому в content лежит не только ответ
    text = next(b.text for b in msg.content if b.type == "text").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    print(f"  вход {msg.usage.input_tokens} ток., выход {msg.usage.output_tokens} ток.")
    return json.loads(text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", action="store_true", help="показать словарь и выйти")
    args = ap.parse_args()

    store = load_store()
    tokens, colors = vocabulary(store)
    catalog = "\n".join(f'{p["id"]}  {p["title"]}  {p["price"]:.0f} EUR'
                        for p in store["products"])

    if args.vocab:
        print(f"слов в названиях: {len(tokens)}\n  {', '.join(tokens)}")
        print(f"\nцветов: {len(colors)}\n  {', '.join(colors)}")
        return 0

    print(f"словарь: {len(tokens)} слов, {len(colors)} цветов, "
          f"{len(store['products'])} товаров → {MODEL}")
    table = call_model(catalog, tokens, colors)

    missing = [t for t in tokens if t not in table.get("tokens", {})]
    if missing:
        print(f"  ⚠ модель пропустила слова: {missing}")
    no_phrases = [p["id"] for p in store["products"]
                  if not table.get("product_phrases", {}).get(p["id"])]
    if no_phrases:
        print(f"  ⚠ товары без фраз: {no_phrases}")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": MODEL,
        "store_sha256": hashlib.sha256(STORE.read_bytes()).hexdigest(),
        "tokens": table.get("tokens", {}),
        "colors": table.get("colors", {}),
        "product_phrases": table.get("product_phrases", {}),
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    n_forms = sum(len(v) for v in table.get("tokens", {}).values())
    n_phr = sum(len(v) for v in table.get("product_phrases", {}).values())
    print(f"записано: {OUT}\n  {n_forms} форм слов, {n_phr} фраз товаров")
    return 0


if __name__ == "__main__":
    sys.exit(main())
