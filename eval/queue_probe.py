#!/usr/bin/env python3
"""
Как провайдер раздаёт слот: по порядку или заново на каждый запрос.

Схема замера (предложена владельцем, и она чище предыдущей):

    раз в секунду  — пара запросов, второй через 10 мс после первого.
    30 пар.

Почему так. Секунда между парами не даёт нам самим накопить очередь —
в прошлом опыте пары шли впритык и мерили нашу же нагрузку. А 10 мс внутри
пары — это «тот же момент» с сохранённым порядком: видно не только,
расходятся ли ожидания, но и В КАКУЮ сторону.

Что означает результат:

    второй систематически ждёт дольше первого  → очередь общая, по порядку.
        Дублирующий запрос бесполезен: он встанет в хвост за первым.
    знак разницы делится примерно поровну      → слот разыгрывается заново.
        Дубль работает — вторая попытка не наследует невезение первой.
    оба ждут одинаково долго или одинаково мало → состояние общее, но
        не очередь: затор у провайдера накрывает оба запроса.

    python eval/queue_probe.py --pairs 30
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import run as R                                # noqa: E402
from resolver.answer import Answerer           # noqa: E402
from resolver.resolve import load              # noqa: E402


def one(key: str, q: str, data: str, slot: dict, tag: str, delay_ms: float) -> None:
    """Один вызов на своём соединении — общее сериализовало бы запросы у нас же."""
    if delay_ms:
        time.sleep(delay_ms / 1000)
    m = R.Model(key)
    t0 = time.perf_counter()
    try:
        r = m._post({
            "model": R.MODEL, "max_tokens": R.MAX_TOKENS, "stream": False,
            "reasoning_effort": "none",
            "messages": [{"role": "system", "content": R.SYSTEM + "\n\nДАНІ:\n" + data},
                         {"role": "user", "content": q}]}, timeout=8.0)
    except Exception as e:                     # noqa: BLE001
        slot[tag] = {"error": repr(e)}
        return
    ms = (time.perf_counter() - t0) * 1000
    if r is None or r.status_code != 200:
        slot[tag] = {"error": f"HTTP {getattr(r, 'status_code', 'timeout')}"}
        return
    u = r.json().get("usage", {})
    slot[tag] = {"ms": ms, "queue_ms": u.get("queue_time", 0) * 1000,
                 "svc_ms": u.get("total_time", 0) * 1000}


def pct(v, p):
    s = sorted(v)
    return s[max(0, math.ceil(p * len(s)) - 1)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=30)
    ap.add_argument("--gap", type=float, default=10.0, help="мс между первым и вторым")
    ap.add_argument("--every", type=float, default=1.0, help="с между парами")
    ap.add_argument("--out", default="eval/queue_probe.json")
    args = ap.parse_args()

    store = json.loads((ROOT / "handout" / "store.json").read_text(encoding="utf-8"))
    resolver, answerer = load(), Answerer(store)
    todo = []
    for line in (ROOT / "eval" / "generated.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        it = json.loads(line)
        res = resolver.resolve(it["q"])
        if answerer.reply(res, it["q"]).decision == "escalate":
            todo.append((it["q"], R.build_slice(store, res)))

    key = R.read_key()
    R.Model(key).probe()
    print(f"{args.pairs} пар, второй через {args.gap:.0f} мс, пара раз в "
          f"{args.every:.0f} с", file=sys.stderr)

    rows = []
    for i in range(args.pairs):
        started = time.perf_counter()
        q, data = todo[i % len(todo)]
        slot: dict = {}
        t1 = threading.Thread(target=one, args=(key, q, data, slot, "first", 0.0))
        t2 = threading.Thread(target=one, args=(key, q, data, slot, "second", args.gap))
        t1.start(); t2.start(); t1.join(); t2.join()
        if "error" not in slot.get("first", {}) and "error" not in slot.get("second", {}):
            rows.append({"i": i, "first": slot["first"], "second": slot["second"]})
        else:
            print(f"  пара {i}: {slot}", file=sys.stderr)
        left = args.every - (time.perf_counter() - started)
        if left > 0:
            time.sleep(left)

    Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                              encoding="utf-8")

    q1 = [r["first"]["queue_ms"] for r in rows]
    q2 = [r["second"]["queue_ms"] for r in rows]
    d = [b - a for a, b in zip(q1, q2)]
    print(f"\nпар удачных: {len(rows)} → {args.out}\n")
    print(f"{'пара':>4} | {'очередь 1':>10} | {'очередь 2':>10} | {'разница':>9} | кто быстрее")
    print("-" * 62)
    for r, dd in zip(rows, d):
        a, b = r["first"]["queue_ms"], r["second"]["queue_ms"]
        who = "второй" if dd < -5 else ("первый" if dd > 5 else "поровну")
        print(f"{r['i']:>4} | {a:10.0f} | {b:10.0f} | {dd:+9.0f} | {who}")

    slower = sum(1 for x in d if x > 5)
    faster = sum(1 for x in d if x < -5)
    same = len(d) - slower - faster
    ma, mb = statistics.mean(q1), statistics.mean(q2)
    cov = sum((x - ma) * (y - mb) for x, y in zip(q1, q2))
    den = math.sqrt(sum((x - ma) ** 2 for x in q1) * sum((y - mb) ** 2 for y in q2))
    print(f"\nвторой ждал ДОЛЬШЕ первого : {slower}/{len(d)}")
    print(f"второй ждал МЕНЬШЕ первого : {faster}/{len(d)}")
    print(f"разница в пределах ±5 мс   : {same}/{len(d)}")
    print(f"медиана разницы            : {statistics.median(d):+.0f} мс")
    print(f"корреляция очередей в паре : {cov / den if den else 0:+.2f}")
    print(f"\nочередь 1: p50 {statistics.median(q1):6.0f}  p95 {pct(q1, .95):6.0f}  max {max(q1):6.0f}")
    print(f"очередь 2: p50 {statistics.median(q2):6.0f}  p95 {pct(q2, .95):6.0f}  max {max(q2):6.0f}")
    big = [(a, b) for a, b in zip(q1, q2) if max(a, b) > 300]
    if big:
        both = sum(1 for a, b in big if min(a, b) > 300)
        print(f"\nпар, где хоть один ждал >300 мс: {len(big)}; из них ждали оба: {both}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
