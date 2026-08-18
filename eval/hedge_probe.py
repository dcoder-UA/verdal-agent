#!/usr/bin/env python3
"""
Независима ли очередь у провайдера — то есть работает ли дублирующий запрос.

Весь план с дублем держится на одном допущении: второй запрос разыгрывает
слот ЗАНОВО. Если очередь общая и обслуживается по порядку, дубль встанет
за первым запросом и сделает только хуже. Политику планировщика Groq мы
не знаем, поэтому проверяем поведением, а не документацией.

Два опыта:

  ПАРА     два запроса уходят одновременно, на разных соединениях.
           Если очередь — общее состояние, их `queue_time` будут похожи
           (высокая корреляция). Если лотерея — разойдутся.

  ДУБЛЬ    ровно то, что мы собираемся делать: первый запрос, через
           HEDGE мс второй, побеждает тот, кто вернулся раньше. Считаем,
           сколько раз дубль реально спас вызов.

    python eval/hedge_probe.py --trials 40
"""
from __future__ import annotations

import argparse
import json
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

WIDE_S = 5.0


def call(key: str, q: str, data: str, out: dict, tag: str, delay: float = 0.0) -> None:
    """Один вызов на СВОЁМ соединении: общее соединение сериализовало бы запросы."""
    if delay:
        time.sleep(delay)
    m = R.Model(key)
    t0 = time.perf_counter()
    try:
        r = m._post({
            "model": R.MODEL, "max_tokens": R.MAX_TOKENS, "stream": False,
            "reasoning_effort": "none",
            "messages": [{"role": "system", "content": R.SYSTEM + "\n\nДАНІ:\n" + data},
                         {"role": "user", "content": q}]}, timeout=WIDE_S)
    except Exception as e:                     # noqa: BLE001
        out[tag] = {"error": repr(e)}
        return
    ms = (time.perf_counter() - t0) * 1000
    if r is None or r.status_code != 200:
        out[tag] = {"error": f"HTTP {getattr(r, 'status_code', 'timeout')}"}
        return
    u = r.json().get("usage", {})
    out[tag] = {"ms": ms, "queue_ms": u.get("queue_time", 0) * 1000,
                "started": delay * 1000}


def corr(a: list[float], b: list[float]) -> float:
    import math
    ma, mb = statistics.mean(a), statistics.mean(b)
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return cov / den if den else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--hedge", type=float, default=250.0, help="мс до дубля")
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

    print(f"ОПЫТ 1 — ПАРА ОДНОВРЕМЕННЫХ ЗАПРОСОВ ({args.trials} пар)", file=sys.stderr)
    pairs = []
    for i in range(args.trials):
        q, data = todo[i % len(todo)]
        out: dict = {}
        ts = [threading.Thread(target=call, args=(key, q, data, out, tag))
              for tag in ("a", "b")]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        if "error" in out.get("a", {}) or "error" in out.get("b", {}):
            continue
        pairs.append((out["a"], out["b"]))
    qa = [x["queue_ms"] for x, _ in pairs]
    qb = [y["queue_ms"] for _, y in pairs]
    print(f"\nпар удачных: {len(pairs)}")
    print(f"  очередь A: p50 {statistics.median(qa):6.0f}  max {max(qa):6.0f}")
    print(f"  очередь B: p50 {statistics.median(qb):6.0f}  max {max(qb):6.0f}")
    print(f"  КОРРЕЛЯЦИЯ очередей в паре: {corr(qa, qb):+.2f}")
    both = sum(1 for x, y in pairs if x["queue_ms"] > 300 and y["queue_ms"] > 300)
    one = sum(1 for x, y in pairs if (x["queue_ms"] > 300) != (y["queue_ms"] > 300))
    print(f"  оба ждали >300 мс: {both};  ждал только один: {one}")

    print(f"\nОПЫТ 2 — ДУБЛЬ ЧЕРЕЗ {args.hedge:.0f} мс ({args.trials} попыток)", file=sys.stderr)
    saved = worse = 0
    firsts, effs = [], []
    for i in range(args.trials):
        q, data = todo[(i + 7) % len(todo)]
        out = {}
        t1 = threading.Thread(target=call, args=(key, q, data, out, "first"))
        t2 = threading.Thread(target=call, args=(key, q, data, out, "hedge", args.hedge / 1000))
        t1.start(); t2.start(); t1.join(); t2.join()
        if "error" in out.get("first", {}) or "error" in out.get("hedge", {}):
            continue
        first = out["first"]["ms"]
        hedged = args.hedge + out["hedge"]["ms"]
        eff = min(first, hedged) if first > args.hedge else first
        firsts.append(first); effs.append(eff)
        if eff < first - 1:
            saved += 1
        if first > 470 and eff > 470:
            worse += 1
    print(f"\nпопыток удачных: {len(firsts)}")
    print(f"  без дубля: p50 {statistics.median(firsts):6.0f}  p95 "
          f"{sorted(firsts)[int(0.95 * len(firsts))]:6.0f}  max {max(firsts):6.0f}")
    print(f"  с дублем : p50 {statistics.median(effs):6.0f}  p95 "
          f"{sorted(effs)[int(0.95 * len(effs))]:6.0f}  max {max(effs):6.0f}")
    print(f"  дубль оказался быстрее: {saved}/{len(firsts)}")
    print(f"  уложилось в 470 мс: без дубля "
          f"{sum(1 for t in firsts if t <= 470)}/{len(firsts)}, "
          f"с дублем {sum(1 for t in effs if t <= 470)}/{len(effs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
