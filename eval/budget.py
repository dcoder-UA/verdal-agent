#!/usr/bin/env python3
"""
Бюджет задержки по компонентам — третий пункт сдачи.

Меряет НЕ копию конвейера, а сам конвейер: `run.answer_one` вызывается как
есть, а компоненты подменяются таймерными обёртками поверх боевых методов.
Копия разошлась бы с боевым кодом на первой же правке, и таблица осталась бы
правдоподобной, но неверной.

Время считается ИСКЛЮЧИТЕЛЬНОЕ (exclusive): из времени `resolve` вычтено то,
что ушло во вложенные замеренные вызовы. Иначе `normalize`, который зовётся
и из разбора, и из сборки текста, попал бы в сумму дважды, и колонка
перестала бы складываться в целое.

    python eval/budget.py                          # handout/questions.jsonl
    python eval/budget.py --in eval/generated.jsonl
    python eval/budget.py --no-model               # только быстрый путь
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import resolver.answer as answer_mod    # noqa: E402
import resolver.resolve as resolve_mod  # noqa: E402
import run as run_mod                   # noqa: E402
from resolver.answer import Answerer    # noqa: E402
from resolver.resolve import load       # noqa: E402

# --- сбор времени ------------------------------------------------------

CUR: dict[str, float] = {}       # исключительное время по компонентам, мс
STACK: list[float] = []          # сколько времени текущий вызов отдал детям


def timed(name: str, fn):
    def wrap(*a, **kw):
        STACK.append(0.0)
        t0 = perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            el = (perf_counter() - t0) * 1000
            child = STACK.pop()
            if STACK:
                STACK[-1] += el
            CUR[name] = CUR.get(name, 0.0) + (el - child)
    return wrap


def patch(resolver, answerer) -> None:
    """
    Обёртки ставятся на экземпляры и на модули, а не на классы: так замер
    не переживает импорт и не может случайно уехать в боевой прогон.
    """
    R, A = type(resolver), type(answerer)
    for owner, obj, attr, label in (
        (resolver, R, "_intent",         "намерение"),
        (resolver, R, "_unexplained",    "покрытие словарём"),
        (resolver, R, "_slots",          "слоты (размер/цвет/сумма)"),
        (resolver, R, "_match_products", "матчинг товара"),
        (resolver, R, "_variants",       "варианты"),
        (resolver, R, "resolve",         "разбор: прочее"),
        (answerer, A, "gate",            "ворота эскалации"),
        (answerer, A, "reply",           "шаблон ответа"),
    ):
        setattr(owner, attr, timed(label, getattr(obj, attr).__get__(owner, obj)))

    # normalize зовётся из обоих модулей — оборачиваем в каждом пространстве имён
    for mod in (resolve_mod, answer_mod):
        mod.normalize = timed("нормализация", mod.normalize)

    run_mod.build_slice = timed("срез магазина", run_mod.build_slice)
    if run_mod.Model is not None:
        run_mod.Model.ask = timed("вызов модели (сеть+генерация)", run_mod.Model.ask)


# --- отчёт -------------------------------------------------------------

def pct(v: list[float], p: float) -> float:
    s = sorted(v)
    import math
    return s[max(0, math.ceil(p * len(s)) - 1)]


def table(title: str, rows: list[dict], total: list[float]) -> None:
    n = len(total)
    if not n:
        return
    names: list[str] = []
    for r in rows:
        for k in r:
            if k not in names:
                names.append(k)

    p95_i = sorted(range(n), key=lambda i: total[i])[max(0, __import__("math").ceil(0.95 * n) - 1)]

    print(f"\n### {title} — {n} вопросов")
    print("| компонент | медиана, мс | среднее, мс | у p95-вопроса, мс |")
    print("|---|---:|---:|---:|")
    for name in names:
        col = [r.get(name, 0.0) for r in rows]
        print(f"| {name} | {statistics.median(col):.3f} | {statistics.fmean(col):.3f} "
              f"| {rows[p95_i].get(name, 0.0):.3f} |")
    measured = [sum(r.values()) for r in rows]
    over = [total[i] - measured[i] for i in range(n)]
    print(f"| *накладные конвейера* | {statistics.median(over):.3f} "
          f"| {statistics.fmean(over):.3f} | {over[p95_i]:.3f} |")
    print(f"| **итого сквозное** | **{statistics.median(total):.3f}** "
          f"| **{statistics.fmean(total):.3f}** | **{total[p95_i]:.3f}** |")
    print(f"\np50 {statistics.median(total):.1f} · p95 {pct(total, 0.95):.1f} · "
          f"max {max(total):.1f} мс")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default="handout/questions.jsonl")
    ap.add_argument("--no-model", action="store_true")
    args = ap.parse_args()

    questions = [json.loads(l) for l in
                 Path(args.src).read_text(encoding="utf-8").splitlines() if l.strip()]
    store = json.loads((ROOT / "handout" / "store.json").read_text(encoding="utf-8"))
    resolver, answerer = load(), Answerer(store)

    model = None
    if not args.no_model:
        model = run_mod.Model(run_mod.read_key())
        model.probe()

    patch(resolver, answerer)

    def once(q: str):
        CUR.clear()
        STACK.clear()
        t0 = perf_counter()
        _, path = run_mod.answer_one(q, resolver, answerer, store, model)
        return (perf_counter() - t0) * 1000, path, dict(CUR)

    for it in questions[:3]:          # прогрев, как в боевом прогоне
        once(it["q"])
    if model is not None:
        model.warm()

    by_path: dict[str, list] = defaultdict(list)
    for it in questions:
        ms, path, parts = once(it["q"])
        by_path[path].append((ms, parts))
        if path == "timeout" and model is not None:
            model.reconnect()

    # handoff — тоже быстрый путь: решение принимает резолвер, сети нет.
    fast = [x for p in ("answer", "abstain", "handoff") for x in by_path.get(p, [])]
    slow = [x for p in ("model", "timeout", "no_model") for x in by_path.get(p, [])]

    print("# Бюджет задержки по компонентам\n")
    print(f"источник: `{args.src}`, модель "
          + ("выключена (`--no-model`)" if model is None else run_mod.MODEL))
    if fast:
        table("Быстрый путь (разбор + шаблон, без сети)",
              [p for _, p in fast], [m for m, _ in fast])
    if slow:
        table("Запасной путь (вопрос ушёл в модель)",
              [p for _, p in slow], [m for m, _ in slow])

    allt = [m for m, _ in fast + slow]
    print(f"\n### Смесь\n\nбыстрых {len(fast)} · наверх {len(slow)} · "
          f"доля наверх {len(slow) / max(1, len(allt)):.1%}")
    # Раздельный счёт исходов: в странице бюджета доля невыполненных вызовов
    # обязана идти из ТОГО ЖЕ прогона, что и задержки, иначе числа склеены
    # из разных нагрузок провайдера.
    print("  " + " · ".join(f"{k} {len(v)}" for k, v in sorted(by_path.items())))
    print(f"p50 {statistics.median(allt):.1f} · p95 {pct(allt, 0.95):.1f} · "
          f"max {max(allt):.1f} мс из бюджета 500")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
