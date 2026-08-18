#!/usr/bin/env python3
"""
Истинный хвост запасного пути: сколько модель отвечает, если её НЕ обрывать.

Боевой таймаут 470 мс срезает распределение, и по прогону видно только,
что «14 из 66 не успели». Чего не видно — не успели на 20 мс или на секунду.
Разница решает всё: в первом случае лечится порогом, во втором — нет.

Заодно снимаем разбивку самого провайдера (`usage`): очередь, префилл,
генерация. Она отвечает на вопрос, за что мы платим временем, — и
оптимизировать вслепую больше не придётся.

    python eval/tail.py                       # все эскалации из generated.jsonl
    python eval/tail.py --in eval/hard.jsonl --limit 10
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import run as R                                # noqa: E402
from resolver.answer import Answerer           # noqa: E402
from resolver.resolve import load              # noqa: E402

WIDE_S = 5.0        # щедрый таймаут: нам нужно распределение, а не отсечка


def pct(v, p):
    import math
    s = sorted(v)
    return s[max(0, math.ceil(p * len(s)) - 1)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default="eval/generated.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", dest="dst", default="eval/tail.json")
    args = ap.parse_args()

    store = json.loads((ROOT / "handout" / "store.json").read_text(encoding="utf-8"))
    resolver, answerer = load(), Answerer(store)

    todo = []
    for line in Path(args.src).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        it = json.loads(line)
        res = resolver.resolve(it["q"])
        if answerer.reply(res, it["q"]).decision == "escalate":
            todo.append((it["id"], it["q"], R.build_slice(store, res)))
    if args.limit:
        todo = todo[:args.limit]
    print(f"эскалаций: {len(todo)}", file=sys.stderr)

    m = R.Model(R.read_key())
    m.probe()
    m.warm()

    rows = []
    for i, (qid, q, data) in enumerate(todo, 1):
        t0 = time.perf_counter()
        r = m._post({
            "model": R.MODEL, "max_tokens": R.MAX_TOKENS, "stream": False,
            "reasoning_effort": "none",
            "messages": [{"role": "system", "content": R.SYSTEM + "\n\nДАНІ:\n" + data},
                         {"role": "user", "content": q}]}, timeout=WIDE_S)
        ms = (time.perf_counter() - t0) * 1000
        if r is None or r.status_code != 200:
            print(f"  {qid}: HTTP {getattr(r, 'status_code', 'timeout')}", file=sys.stderr)
            m.reconnect()
            continue
        body = r.json()
        u = body.get("usage", {})
        rows.append({
            "id": qid, "ms": round(ms, 1),
            "queue_ms": round(u.get("queue_time", 0) * 1000, 1),
            "prompt_ms": round(u.get("prompt_time", 0) * 1000, 1),
            "completion_ms": round(u.get("completion_time", 0) * 1000, 1),
            "total_ms": round(u.get("total_time", 0) * 1000, 1),
            "in_tok": u.get("prompt_tokens"), "out_tok": u.get("completion_tokens"),
        })
        if i % 10 == 0:
            print(f"  {i}/{len(todo)}", file=sys.stderr)

    Path(args.dst).write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    ms = [x["ms"] for x in rows]
    print(f"\nзамеров {len(ms)} → {args.dst}")
    print(f"  сквозное  p50 {statistics.median(ms):6.0f}  p75 {pct(ms,.75):6.0f}  "
          f"p90 {pct(ms,.90):6.0f}  p95 {pct(ms,.95):6.0f}  max {max(ms):6.0f}")
    for k, label in (("queue_ms", "очередь"), ("prompt_ms", "префилл"),
                     ("completion_ms", "генерация")):
        v = [x[k] for x in rows]
        print(f"  {label:10} p50 {statistics.median(v):6.0f}  p95 {pct(v,.95):6.0f}  "
              f"max {max(v):6.0f}")
    out = [x["out_tok"] for x in rows]
    print(f"  токенов на выходе: p50 {statistics.median(out):.0f}  max {max(out)}")
    net = [x["ms"] - x["total_ms"] for x in rows]
    print(f"  сеть (сквозное минус учтённое провайдером): "
          f"p50 {statistics.median(net):.0f}  p95 {pct(net,.95):.0f} мс")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
