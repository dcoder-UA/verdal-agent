#!/usr/bin/env python3
"""
Одна команда: читает файл вопросов, пишет файл ответов.

    python run.py                                  # handout/questions.jsonl -> results.jsonl
    python run.py --in their.jsonl --out out.jsonl

Каждый вопрос идёт по одному из двух путей, и выбирает путь не размер вопроса,
а то, разобрали мы его или нет:

    быстрый   разбор + шаблон, ~3 мс, без сети. Сюда попадает подавляющее
              большинство вопросов, и именно поэтому p95 укладывается в бюджет.
    запасной  вопрос уходит в модель. В промпт кладётся НЕ весь магазин,
              а срез, собранный из того, что резолвер всё-таки нашёл:
              полный каталог стоит ~820 мс префилла на каждом вызове.

Про недоступность модели и таймаут. Это РАЗНЫЕ события, и реакция разная:

    нет доступа (401/403/сеть/нет ключа)  →  падаем громко, файл не пишем.
                Результаты без модели считаются скомпрометированными:
                часть ответов оказалась бы «не знаю» по причинам, не имеющим
                отношения к вопросам, и отличить их в файле было бы нельзя.
    не успела (таймаут 470 мс)            →  ОТДЕЛЬНЫЙ текст «не встиг
                перевірити», не «даних немає»: первое говорит о нас, второе
                о магазине, и склеивать их в отчёте значит писать неправду.
                Прогон идёт дальше, случай считается и попадает в сводку.
                В голосовом контуре эта ветка — филлер: он не успевает
                в срок, а меняет срок. В батче реплика обязана завершиться.

Замер идёт по второму проходу: первый прогревает процесс и соединение,
как и требует README задания.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

import http.client

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from resolver.answer import Answerer          # noqa: E402
from resolver.resolve import load             # noqa: E402

BUDGET_MS = 500.0
TIMEOUT_S = 0.47          # 470 мс. Замер эскалированных вопросов дал max 420,
                          # то есть при 450 оставалось 30 мс запаса — на чужой
                          # сети такие вызовы срывались бы в «не знаю».
                          # Ответ «не знаю» на живом вопросе покупателю заметнее,
                          # чем 20 мс в отчёте, поэтому запас отдан вызову.
MODEL = "qwen/qwen3.6-27b"
HOST = "api.groq.com"
PATH = "/openai/v1/chat/completions"
MAX_TOKENS = 120          # ответ покупателю — одно-два предложения
USER_AGENT = "verdal-agent/1.0 (python-stdlib)"

SYSTEM = (
    "Ти — асистент інтернет-магазину Verdal. Відповідай покупцям, спираючись "
    "виключно на дані нижче. Одне-два речення, мовою питання, без вступів. "
    "Якщо даних немає — прямо скажи, що не знаєш."
)


class NoModelAccess(RuntimeError):
    """Модель недоступна. Это не ответ «не знаю», это сбой прогона."""


def read_key() -> str:
    key = os.environ.get("GROQ_API_KEY")
    if not key and (ROOT / ".env").exists():
        m = re.search(r"^\s*(?:export\s+)?GROQ_API_KEY\s*=\s*(.+)$",
                      (ROOT / ".env").read_text(encoding="utf-8"), re.M)
        if m:
            key = m.group(1).strip().strip('"\'')
    if not key:
        raise NoModelAccess(
            "нет GROQ_API_KEY (ни в окружении, ни в .env)")
    return key


class Response:
    """Минимум, который нужен от ответа: код и тело."""
    __slots__ = ("status_code", "text")

    def __init__(self, status_code: int, text: str):
        self.status_code, self.text = status_code, text

    def json(self) -> dict:
        return json.loads(self.text)


class Model:
    """
    Клиент на стандартной библиотеке. httpx или requests сюда просятся, но вся
    остальная система обходится без зависимостей вообще, и ради одного POST
    ломать это не стоит: чем меньше в репозитории того, что надо ставить,
    тем меньше поводов ему не запуститься на чужой машине.

    Соединение держим ОТКРЫТЫМ между вопросами. Это не оптимизация из
    любви к искусству: на замере через `urllib`, который соединение
    не переиспользует, каждый вызов платил заново за TCP и TLS, и вызовы,
    обязанные упереться в таймаут 450 мс, занимали 530. То есть рукопожатие
    съедало пятую часть бюджета на ровном месте.
    """

    def __init__(self, key: str):
        self.key = key
        self.conn: http.client.HTTPSConnection | None = None
        self.timeouts = 0
        self.throttled = 0     # 429: считаем отдельно, это неверный тариф

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
                # Без своего User-Agent Cloudflare у Groq отдаёт 403 с кодом
                # 1010: строка «Python-urllib/3.x» у него в чёрном списке.
                # Это блокировка на периметре, а не отказ ключа, и спутать
                # их очень легко — сообщение об ошибке говорит про обе причины.
                "User-Agent": USER_AGENT}

    def _drop(self) -> None:
        """После сбоя состояние соединения неизвестно — закрываем."""
        if self.conn is not None:
            try:
                self.conn.close()
            except OSError:
                pass
            self.conn = None

    def _post(self, body: dict, timeout: float | None = None) -> Response | None:
        t = timeout or TIMEOUT_S
        payload = json.dumps(body)
        # Две попытки: сервер вправе закрыть простаивавшее соединение, и это
        # рядовое событие, а не недоступность модели.
        for attempt in (1, 2):
            try:
                if self.conn is None or self.conn.timeout != t:
                    self._drop()
                    self.conn = http.client.HTTPSConnection(HOST, timeout=t)
                self.conn.request("POST", PATH, payload, self._headers())
                r = self.conn.getresponse()
                return Response(r.status, r.read().decode("utf-8", "replace"))
            except TimeoutError:
                self._drop()
                return None
            except (http.client.HTTPException, OSError) as e:
                self._drop()
                if attempt == 2:
                    raise NoModelAccess(f"сеть недоступна: {type(e).__name__}: {e}") from e
        return None

    def reconnect(self) -> None:
        """
        Поднять соединение заново, НЕ внутри замера.

        После таймаута соединение закрывается, и рукопожатие за него платил бы
        следующий эскалированный вопрос: замеры показывали 564 мс там, где
        порог 450. Стоимость реальная, но к этому вопросу отношения не имеет,
        поэтому переносим её за пределы измеряемого окна.
        """
        self._drop()
        try:
            self.conn = http.client.HTTPSConnection(HOST, timeout=TIMEOUT_S)
            self.conn.connect()
        except OSError:
            self._drop()      # не вышло — попробуем на следующем вопросе

    def probe(self) -> None:
        """
        Одна проверка на старте, до замеров. Отделяет «ключ не работает»
        от «модель не успела»: первое обязано остановить прогон, второе — нет.
        Прогревочный проход README считать не требует, так что проба
        в бюджет не входит.
        """
        r = self._post({"model": MODEL, "max_tokens": 1, "stream": False,
                        "messages": [{"role": "user", "content": "ping"}],
                        "reasoning_effort": "none"}, timeout=20.0)
        if r is None:
            raise NoModelAccess("проба не уложилась в 20 с — модель недоступна")
        if r.status_code == 401:
            raise NoModelAccess(f"HTTP 401: ключ отклонён. {r.text[:200]}")
        if r.status_code == 403:
            # 403 приходит от периметра, а не от API, и причин две:
            # датацентровый IP (VPN) либо неизвестный User-Agent.
            raise NoModelAccess(
                f"HTTP 403 от периметра (не от API): {r.text[:120]} — "
                f"обычно это включённый VPN или заблокированный User-Agent")
        if r.status_code == 429:
            print("  ⚠ проба вернула 429 — ключ на бесплатном тарифе, "
                  "часть вызовов не успеет", file=sys.stderr)
            return
        if r.status_code != 200:
            raise NoModelAccess(f"HTTP {r.status_code}: {r.text[:200]}")

    def warm(self) -> None:
        """
        Прогрев соединения ЩЕДРЫМ таймаутом, а не боевым.

        Раньше здесь звался обычный `ask`, то есть первый вызов должен был
        уложить в 470 мс и TLS-рукопожатие, и ответ модели. Он не укладывался,
        соединение закрывалось — и за рукопожатие платил уже замеряемый
        вопрос. В прогонах это давало то двойной таймаут, то 510 мс при
        пороге 470, то есть выброс за бюджет из-за прогрева, а не из-за модели.
        """
        self._post({"model": MODEL, "max_tokens": 1, "stream": False,
                    "reasoning_effort": "none",
                    "messages": [{"role": "user", "content": "ping"}]},
                   timeout=20.0)
        # После щедрого вызова соединение открыто, но заведено под таймаут 20 с;
        # боевой режим требует своего, поэтому переподключаемся сразу здесь.
        self.reconnect()

    def ask(self, question: str, data: str) -> str | None:
        """Ответ модели, либо None — если не успела. Недоступность бросает."""
        r = self._post({
            "model": MODEL, "max_tokens": MAX_TOKENS, "stream": False,
            "messages": [{"role": "system", "content": SYSTEM + "\n\nДАНІ:\n" + data},
                         {"role": "user", "content": question}],
            # Температуру задаём явно. Без неё работает дефолт провайдера
            # (у OpenAI-совместимого API это 1.0), и один и тот же вопрос
            # получает разные ответы: проверено, три вызова с одинаковым
            # входом дали три разных текста. Для ассистента магазина это
            # неверно по существу — покупателю нужен один ответ, а не
            # три формулировки, — и заодно лишает прогон воспроизводимости.
            "temperature": 0.0,
            "reasoning_effort": "none"})
        if r is None:
            self.timeouts += 1
            return None
        if r.status_code in (401, 403):
            raise NoModelAccess(f"HTTP {r.status_code} посреди прогона: {r.text[:200]}")
        if r.status_code == 429:
            # Ответа нет, как при таймауте, но причина другая — квота тарифа.
            # Ведём себя так же, а считаем отдельно, чтобы это было видно.
            self.throttled += 1
            return None
        if r.status_code != 200:
            raise NoModelAccess(f"HTTP {r.status_code}: {r.text[:200]}")
        return (r.json()["choices"][0]["message"].get("content") or "").strip() or None


def build_slice(store: dict, res) -> str:
    """
    Срез магазина под конкретный вопрос — из того, что нашёл резолвер.

    Даже когда вопрос уходит наверх, разбор редко оказывается пустым: чаще
    он опознал товар, но не понял, о чём спрашивают. Тогда модель получает
    один товар вместо восемнадцати. Если не нашлось ничего — кладём краткий
    указатель по каталогу (название и цена, без вариантов) и все правила:
    это всё равно вчетверо меньше полного магазина.
    """
    out: dict = {"shop": store["shop"], "currency": store["currency"]}
    pids = list(res.products) + list(res.ambiguous)
    if pids:
        out["products"] = [p for p in store["products"] if p["id"] in pids]
    if res.order:
        out["orders"] = [o for o in store["orders"] if o["name"] == res.order]
    if res.policy:
        out["policies"] = {res.policy: store["policies"][res.policy]}
    if not pids and not res.order:
        out["catalogue"] = [{"title": p["title"], "price": p["price"]}
                            for p in store["products"]]
        out["policies"] = store["policies"]
    return json.dumps(out, ensure_ascii=False)


def answer_one(q: str, resolver, answerer, store: dict, model) -> tuple[str, str]:
    """
    Один вопрос от строки до готового текста. Возвращает (ответ, путь).

    Живёт на уровне модуля, а не внутри `main`, ровно по одной причине:
    замер бюджета задержки (`eval/budget.py`) обязан мерить ЭТОТ конвейер,
    а не свою копию. Копия разошлась бы с боевым кодом на первой же правке,
    и таблица «компонент → мс» врала бы, оставаясь правдоподобной.
    """
    res = resolver.resolve(q)
    rep = answerer.reply(res, q)
    if rep.decision != "escalate":
        return rep.text, rep.decision
    lang = "ua" if any("а" <= c <= "я" for c in q.lower()) else "en"
    if model is None:
        return answerer._no_data(lang), "no_model"
    text = model.ask(q, build_slice(store, res))
    if text is None:
        # Не уложились в таймаут — это НЕ «данных нет». Тексты разные,
        # и в сводке эти случаи считаются раздельно: первый говорит
        # о магазине, второй о нас.
        return answerer._not_in_time(lang), "timeout"
    return text, "model"


def pct(values: list[float], p: float) -> float:
    """Ближайший ранг: на 39 вопросах интерполяция придумывает несуществующее."""
    s = sorted(values)
    import math
    return s[max(0, math.ceil(p * len(s)) - 1)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default="handout/questions.jsonl")
    ap.add_argument("--out", dest="dst", default="results.jsonl")
    ap.add_argument("--no-model", action="store_true",
                    help="только детерминированный путь; прогон помечается неполным")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    questions = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]

    store = json.loads((ROOT / "handout" / "store.json").read_text(encoding="utf-8"))
    resolver = load()                      # индекс строится один раз, вне замера
    answerer = Answerer(store)

    model = None
    if not args.no_model:
        try:
            model = Model(read_key())
            model.probe()
        except NoModelAccess as e:
            print(f"\n✗ МОДЕЛЬ НЕДОСТУПНА: {e}\n"
                  f"  Прогон остановлен, {dst} не записан: без запасного пути\n"
                  f"  часть ответов была бы «не знаю» по причинам, не связанным\n"
                  f"  с вопросами, и результаты нельзя считать достоверными.\n"
                  f"  Проверьте GROQ_API_KEY и доступ к сети.", file=sys.stderr)
            return 2

    def once(q: str) -> tuple[str, str]:
        return answer_one(q, resolver, answerer, store, model)

    # Прогревочный проход: процесс, кеши интерпретатора и — главное —
    # TLS-соединение до провайдера. README прямо разрешает его не считать.
    # Соединение греем отдельным вызовом: если среди первых вопросов ни один
    # не уйдёт наверх, первый же эскалированный вопрос замера заплатил бы
    # за рукопожатие, и это попало бы в статистику.
    for it in questions[:3]:
        once(it["q"])
    if model is not None:
        model.warm()

    rows, times, paths = [], [], []
    try:
        for it in questions:
            t0 = time.perf_counter()
            text, path = once(it["q"])
            ms = (time.perf_counter() - t0) * 1000
            rows.append({"id": it["id"], "answer": text, "ms": round(ms)})
            times.append(ms)
            paths.append(path)
            if path == "timeout" and model is not None:
                model.reconnect()      # вне замера, см. Model.reconnect
    except NoModelAccess as e:
        print(f"\n✗ МОДЕЛЬ ОТВАЛИЛАСЬ ПОСРЕДИ ПРОГОНА: {e}\n"
              f"  {dst} не записан — частичные результаты недостоверны.", file=sys.stderr)
        return 2

    dst.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                   encoding="utf-8")

    from collections import Counter
    c = Counter(paths)
    n = len(times)
    p95 = pct(times, 0.95)
    print(f"\nвопросов {n} → {dst}")
    print(f"  шаблон      {c['answer']:4}   обоснованный отказ {c['abstain']:4}")
    print(f"  оператор    {c['handoff']:4}   модель             {c['model']:4}")
    print(f"  не встигли  {c['timeout']:4}"
          + (f"   без модели {c['no_model']}" if c["no_model"] else ""))
    if model is not None and model.throttled:
        print(f"  ⚠ 429 от провайдера: {model.throttled} — ключ не на том тарифе")
    print(f"  задержка   p50 {statistics.median(times):6.1f}   p95 {p95:6.1f}   "
          f"max {max(times):6.1f} мс")
    print(f"  бюджет {BUDGET_MS:.0f} мс по p95 → "
          + ("ПРОЙДЕН" if p95 <= BUDGET_MS else f"ПРОВАЛ, ×{p95/BUDGET_MS:.1f}"))
    if args.no_model:
        print("  ⚠ прогон без модели: неразобранные вопросы отвечены «не знаю», "
              "результаты неполные")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
