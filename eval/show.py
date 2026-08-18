#!/usr/bin/env python3
"""
Читалка результатов: вопрос, ответ, путь и время в одном месте.

`results.jsonl` сдаётся в их формате — `{id, answer, ms}`, — и глазами по нему
работать нельзя: нет ни текста вопроса, ни того, каким путём получен ответ.
А смотреть надо именно на путь: шаблон, обоснованный отказ, модель или
несработавший таймаут — это четыре разных повода к правкам.

Путь восстанавливается, а не хранится: детерминированное решение
пересчитывается резолвером (это доли миллисекунды и никакой сети), а среди
ушедших наверх «не успела» отличается от «модель ответила» по тексту —
у несработавшего таймера своя заготовка, отдельная от обоснованного отказа.

    python eval/show.py                                  # живые 39
    python eval/show.py --q eval/generated.jsonl --r eval/results_generated.jsonl
    python eval/show.py --path model                     # только ответы модели
    python eval/show.py --path timeout --path abstain     # можно несколько
    python eval/show.py --slow 300                        # всё медленнее 300 мс
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from resolver.answer import Answerer          # noqa: E402
from resolver.resolve import load             # noqa: E402

MARK = {"answer": "шаблон", "abstain": "отказ", "model": "МОДЕЛЬ",
        "timeout": "НЕ УСПЕЛА", "unknown": "?"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", dest="src", default="handout/questions.jsonl")
    ap.add_argument("--r", dest="res", default="results.jsonl")
    ap.add_argument("--path", action="append", default=[],
                    help="answer / abstain / model / timeout; можно повторять")
    ap.add_argument("--slow", type=float, default=None,
                    help="показать только медленнее N мс")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    qs = {}
    for line in Path(args.src).read_text(encoding="utf-8").splitlines():
        if line.strip():
            it = json.loads(line)
            qs[it["id"]] = it["q"]
    rows = [json.loads(l) for l in
            Path(args.res).read_text(encoding="utf-8").splitlines() if l.strip()]

    store = json.loads((ROOT / "handout" / "store.json").read_text(encoding="utf-8"))
    resolver, answerer = load(), Answerer(store)
    late = {answerer._not_in_time("ua"), answerer._not_in_time("en")}

    shown, seen = 0, Counter()
    out = []
    for r in rows:
        q = qs.get(r["id"], "")
        decision = answerer.reply(resolver.resolve(q), q).decision if q else "unknown"
        if decision == "escalate":
            # Ушёл наверх. Текст «не встиг» в ответе означает, что сработал
            # наш таймер: у модели свой текст, у обоснованного отказа свой.
            decision = "timeout" if r["answer"] in late else "model"
        seen[decision] += 1
        if args.path and decision not in args.path:
            continue
        if args.slow is not None and r["ms"] < args.slow:
            continue
        out.append((r["id"], decision, r["ms"], q, r["answer"]))

    for rid, decision, ms, q, ans in out:
        if args.limit and shown >= args.limit:
            print(f"\n… ещё {len(out) - shown}, показ ограничен --limit")
            break
        print(f"\n[{rid}] {MARK[decision]:>9} {ms:>4} мс")
        print(f"  ? {q}")
        print(f"  → {ans}")
        shown += 1

    print("\n" + "─" * 60)
    print("всего " + " · ".join(f"{MARK[k]} {v}" for k, v in seen.most_common())
          + (f"   (показано {shown})" if shown != sum(seen.values()) else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
