"""
Детерминированное разрешение вопроса: намерение + сущности.

Никаких вызовов модели и никакой сети. Индекс строится один раз при старте,
дальше каждый вопрос — работа со словарями и короткими строками.

Порядок разбора не случаен, он идёт от самого надёжного признака к самому
шаткому, и первый же сработавший закрывает вопрос:

  1. номер заказа (#1005)     — регулярка, ошибиться нечем;
  2. намерение по ключевым словам — «пляма» это брак, а не куртка в каталоге;
  3. товар                    — единственное место, где возможна неоднозначность.

Больше половины вопросов заканчиваются на шагах 1-2 и до поиска по каталогу
не доходят вовсе.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from .geo import STEMS as GEO_STEMS
from .normalize import key, normalize, similar, translit, words

ROOT = Path(__file__).resolve().parent.parent

MATCH = 0.85          # порог похожести двух слов
AMBIGUOUS_GAP = 1.25  # во столько раз лидер должен опережать второго

# Номер заказа опознаём тремя слоями, от явного к голому. Решётка — это форма
# записи, а не свойство номера: в живой речи её нет вообще, и покупатель скажет
# «номер 1006» или просто «1006».
ORDER_RE = re.compile(r"#\s*(\d{3,6})")
ORDER_WORD_RE = re.compile(
    r"(?:замовлен\w*|заказ\w*|order|номер\w*|№)\s*(?:№\s*)?(\d{3,6})")
# Голое число берём от четырёх цифр: трёхзначные — это цены каталога (249, 119),
# двузначные — размеры. Четыре цифры подряд в этих данных спутать не с чем.
# Исключение — величина с единицей измерения («термос 1000 мл»): в нашем
# каталоге такого нет, но в их файле может встретиться.
ORDER_BARE_RE = re.compile(
    r"(?<!\d)(\d{4,6})(?!\d)(?!\s*(?:мл|мм|см|кг|шт|л|г|м|євро|евро|eur|€))")
MONEY_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:євро|евро|eur|€)")
SIZE_LETTER_RE = re.compile(r"(?<![a-zа-яіїєґ])(xxl|xl|xs|[smlсмл])(?![a-zа-яіїєґ])")
SIZE_NUM_RE = re.compile(r"(?<!\d)(2[89]|3\d|4[0-8])(?!\d)")

CYR_SIZE = {"с": "S", "м": "M", "л": "L"}

# Служебные слова. Модель порождает формы вроде «від дощу» или «з мериносу»,
# и без этого фильтра предлог «від» становится синонимом слова rain, после чего
# «ВІДправляєте» опознаётся как дождевик. Список короткий и закрытый — это
# грамматика языка, она не меняется вместе с каталогом.
FUNCTION_WORDS = {
    "від", "для", "з", "із", "зі", "на", "по", "в", "у", "та", "і", "й", "до",
    "при", "під", "над", "без", "про", "як", "що", "це", "цей", "той", "а",
    "але", "чи", "не", "ні", "є", "он", "их", "ий", "ої", "один", "одна",
    "одне", "дуже", "такий", "штук", "пар", "пари", "три", "два",
}

# Обиходные слова языка: вопросительные, местоимения, бытовые глаголы.
#
# ПРАВИЛО ПОПОЛНЕНИЯ, и оно здесь важнее самого списка: сюда можно класть
# только то, что попало бы в него у человека, НЕ ВИДЕВШЕГО файла вопросов.
# «Можна», «вже», «хто», «гроші» — да, это язык. «Каремат», «спальник»,
# «гарантія» — нет: это предметные существительные, и их незнание как раз
# и есть сигнал, что вопрос про то, чего в каталоге нет. Стоит начать
# добавлять сюда слова из разбора конкретных провалов — и правило покрытия
# перестанет работать, а подгонка под известные вопросы просто переедет
# из правил в этот список.
COMMON_WORDS = set("""
скільки який яка яке які чи де коли хто що можна маєте дайте дай скажи будь
ласка мені моє мій моя ваш ваша ваші вас вам ще вже штук пар пара набір
розмір размер лишилось гроші річ речі якщо плюс разом стандартна стандартний
платить платити вийде йде візьму беру брати взяти купити замовити замовляю
хочу треба потрібно статус євро евро сума суми якої відсотк агент магазин
магазину робити зробити носити мати бути прийшла
is the my your you do does what where when how much many in of have a an it i
me and or to for on at with there are can left size order please tell give
""".split())

# Слова, по которым намерение определяется однозначно. Порядок важен:
# проверяем сверху вниз, первое совпадение выигрывает.
#
# Два языка в одном списке намеренно. Бот двуязычный по прямому требованию
# заказчика, а раздельные списки означали бы два порядка приоритетов вместо
# одного — и расхождение между ними всплыло бы на первом же смешанном вопросе.
# Английские ключи подобраны так, чтобы не перехватывать чужие правила:
# одиночное «pay» не берём, иначе «Who pays for return shipping?» уходило бы
# в оплату вместо возврата.
#
# Ключ с «=» в начале сверяется как ЦЕЛОЕ СЛОВО. Нужно для коротких слов вроде
# «є»: подстрокой оно находится внутри «коштує», «буде», «немає», то есть
# практически в любом вопросе.
INTENT_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("out_of_scope",     ("знижк", "скидк", "промокод", "дешевш ціну зроб",
                          "discount", "promo code", "coupon")),
    ("policy_damaged",   ("плям", "брак", "пошкодж", "дефект", "порван", "розірв",
                          "дірк", "зламал", "зламав", "прийшла з", "бракован",
                          "damaged", "faulty", "defect", "torn", "stain")),
    ("policy_exchange",  ("не підійде", "не підійд", "обмін", "поміня", "заміни",
                          "не той розмір", "інший розмір",
                          "exchange", "wrong size", "another size")),
    ("policy_payment",   ("наложен", "післяплат", "готівк", "карт", "оплат",
                          "платеж", "спишут", "списан", "paypal",
                          "cash on delivery", "payment", "charged",
                          "credit card", "apple pay", "google pay")),
    ("product_restock",  ("завезут", "завоз", "рестік", "ресток", "поповнен",
                          "коли буде знову", "знову",
                          "restock", "back in stock", "when will you have")),
    ("policy_returns",   ("поверн", "зворотн", "пересилк", "поносив", "носив",
                          "прат", "повертат", "рефанд",
                          "return", "refund", "send back", "worn", "washed")),
    # «британ»/«britain»/«uk» отсюда убраны: страна, вписанная под один
    # видимый вопрос, намерение не определяет. Вопрос про доставку опознаётся
    # глаголом («відправляєте», «надсилаєте»), а куда именно — решает
    # `geo.destination`. Вместо страны добавлен оборот про саму границу.
    ("policy_shipping",  ("доставк", "відправля", "відправ", "експрес", "шип",
                          "безкоштовн", "надсила", "за межі", "межі",
                          "shipping", "delivery", "deliver", "express",
                          "outside")),
    ("product_compare",  ("дешевш", "дорожч", "вигідніш", "порівня",
                          "cheaper", "more expensive", "compare")),
    ("product_price",    ("ціна", "цін", "коштує", "коштуют", "почім", "почем",
                          "скільки", "сколько", "стоит",
                          "price", "cost", "how much")),
    # Наличие. Раньше это намерение НЕ имело собственных слов и назначалось
    # дефолтом любому вопросу со знаком вопроса — то есть каждый непонятый
    # вопрос объявлялся вопросом о наличии и получал уверенный неверный ответ.
    # Теперь у наличия есть свои признаки, а непонятое честно остаётся unknown.
    # Ключа «розмір» здесь СОЗНАТЕЛЬНО НЕТ. Он тут был — ради одного вопроса
    # эталона («А 43 розмір кросівок?») — и немедленно превратил вопросы
    # о посадке в вопросы о наличии: «Чи підійде светр розміру М, я 165 см?»
    # получало бодрое «так, 12 шт.», хотя размерной сетки в данных нет вообще.
    # Один спасённый вопрос не стоит целого класса уверенно неверных ответов;
    # без этого ключа такие вопросы честно уходят наверх.
    ("product_stock",    ("наявн", "залишк", "є в", "лишилос", "=є", "чи є",
                          "=маєте", "=есть", "в наличии",
                          "in stock", "available", "do you have", "left")),
]

# Намерения внутри заказа — проверяются только если в вопросе есть номер.
ORDER_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("order_tracking", ("трек", "ттн", "накладн", "номер відправ",
                        "tracking", "track number", "waybill")),
    # «що там» само по себе признаком состава не является: «що там ПО замовленню»
    # — это статус. Признаком служит только более длинный оборот.
    ("order_items",    ("що входит", "склад", "які товар", "там у замовленн",
                        "там в замовленн",
                        "what is in", "whats in", "contents", "items in")),
    # «where» сюда не берём: «Where is my order?» — это статус, а не адрес.
    # В английском различие несёт предлог, а не вопросительное слово.
    ("order_address",  ("адрес", "куди", "місто",
                        "address", "deliver to", "city")),
]


# --- передача человеку -------------------------------------------------
#
# Четвёртый исход рядом с «ответили / отказали / отдали модели». Основание
# у двух подклассов РАЗНОЕ, и это важнее самих списков.
#
# `handoff_change` — просьба ИЗМЕНИТЬ данные: отменить, переоформить,
# сменить адрес или размер в конкретном заказе. Этот подкласс замкнут
# свойством системы, а не перечнем слов: данные у нас только на чтение,
# поэтому просьба поменять запись не может быть выполнена НИКОГДА —
# ни моделью, ни шаблоном. Отдавать её наверх прямо вредно: замер показал,
# что на «Хочу скасувати замовлення #1002» модель отвечает «Замовлення вже
# відправлено, тому скасувати його неможливо» — политики отмены в store.json
# нет вовсе, то есть правило выдумано целиком.
#
# `handoff_human` — просьба позвать человека: жалоба, счёт на компанию,
# перезвон. Здесь это честное перечисление, и оно неполно по устройству.
# Но промах безопасен: не опознали — вопрос уходит наверх, как уходил
# до этой правки. Замер: «Передзвоніть мені» → «Ми не надаємо послуги
# телефонного зв'язку», и это ответ агента компании, которая строит
# голосовой слой.
#
# Границу с политиками НЕ трогаем: «Чи можна поміняти розмір?» — вопрос
# о правилах магазина, на него есть чем ответить. Разделяет их не глагол,
# а наличие НОМЕРА заказа: просьба к конкретной записи — изменение,
# вопрос без записи — политика. Поэтому глаголы смены живут отдельным
# списком и срабатывают только вместе с номером.
HANDOFF_ALWAYS: list[tuple[str, tuple[str, ...]]] = [
    # Отмена. Отдельно от глаголов смены: политики отмены в магазине нет,
    # поэтому и «Чи можна скасувати замовлення?» отвечать нечем — вопрос
    # о правиле, которого не существует, это тоже повод позвать человека.
    ("handoff_change",  ("скасу", "відміни", "отмени", "анулю",
                         "cancel", "annul")),
    # Позвать человека.
    ("handoff_human",   ("оператор", "менеджер", "жив людин", "живою людин",
                         "з людиною", "людиною", "передзвон", "перезвон",
                         "зателефонуйте", "подзвоніть", "скарг", "рекламаці",
                         "рахунок на компанію", "рахунок-фактур",
                         "operator", "manager", "human", "complaint",
                         "call me", "call back", "callback", "invoice")),
]

# Глаголы смены. Срабатывают ТОЛЬКО при найденном номере заказа — иначе
# «поміня»/«заміни» перехватили бы политику обмена, а «змінит» — вопросы
# о том, что вообще можно изменить.
#
# Ключа «оформ» здесь СОЗНАТЕЛЬНО НЕТ, хотя оформление заказа — тоже запись.
# Он ломает «Як оформити повернення?»: это вопрос о процедуре возврата,
# на него в магазине есть политика, и уводить его к оператору — потеря.
# Один класс, пойманный ценой другого, — не выигрыш.
HANDOFF_ON_ORDER: list[tuple[str, tuple[str, ...]]] = [
    ("handoff_change",  ("змінит", "зміні", "зміню", "поміня", "заміни",
                         "переоформ", "перенес", "додайте", "приберіть",
                         "change", "update", "edit", "amend", "reschedule")),
]

HANDOFF_INTENTS = ("handoff_change", "handoff_human")


def _pattern(kw: str) -> re.Pattern:
    """
    Ключ → шаблон. Обычный ключ — это ОСНОВА слова и ищется от начала слова:
    «плям» находит «плямою», но «stain» не находит «sustainable», а «карт»
    не находит «відкарт...». Ключ с «=» сверяется как целое слово — нужно
    коротким словам («є», «маєте»), которые иначе живут внутри чужих:
    «маєте» находилось внутри «приймаєте» и уводило вопрос про PayPal
    в наличие товара.
    """
    if kw.startswith("="):
        return re.compile(r"\b" + re.escape(kw[1:]) + r"\b")
    return re.compile(r"\b" + re.escape(kw))


def _compile(rules):
    return [(n, tuple(_pattern(k) for k in kws)) for n, kws in rules]


INTENT_PATTERNS = _compile(INTENT_RULES)
ORDER_PATTERNS = _compile(ORDER_RULES)
HANDOFF_ALWAYS_PATTERNS = _compile(HANDOFF_ALWAYS)
HANDOFF_ON_ORDER_PATTERNS = _compile(HANDOFF_ON_ORDER)


@dataclass
class Resolution:
    intent: str = "unknown"
    # Как получено намерение: 'exact' — сработало ключевое слово, 'fuzzy' — оно же
    # с допуском на опечатку, 'default' — не сработало ничего. Без этого поля
    # «опознали по слову» и «не опознали ничего» выглядят снаружи одинаково,
    # и непонятый вопрос идёт отвечать наравне с понятым.
    intent_src: str = "default"
    products: list[str] = field(default_factory=list)
    variants: list[str] = field(default_factory=list)
    variant: str | None = None      # заполнен, только когда вариант ровно один
    order: str | None = None
    policy: str | None = None
    ambiguous: list[str] = field(default_factory=list)
    # Содержательные слова вопроса, которые разбор ничем не объяснил.
    # Непустой список означает «в вопросе есть то, чего мы не поняли» —
    # и это условие отвечать САМИМ, а не перечень отдельных дефектов.
    unexplained: list[str] = field(default_factory=list)
    size: str | None = None
    color: str | None = None
    money: float | None = None
    qty: int = 1
    reason: str = ""

    @property
    def resolved(self) -> bool:
        """Нашли хоть что-то, на чём можно построить ответ."""
        return bool(self.products or self.order or self.policy) or \
            self.intent in ("out_of_scope",)


class Resolver:
    """Индекс строится в конструкторе — один раз на процесс, вне бюджета."""

    def __init__(self, store: dict, aliases: dict):
        self.store = store
        self.products = {p["id"]: p for p in store["products"]}
        self.orders = {o["name"]: o for o in store["orders"]}
        self.policies = store["policies"]

        # слова английских названий → товары
        self.tok_of: dict[str, set[str]] = {}
        # Слово-носитель названия: то, которое НАЗЫВАЕТ вещь, а не описывает её.
        # `Merino Wool Sweater` → sweater, `Hiking Backpack 35L` → backpack.
        # Берём последнее слово названия, отбрасывая размерные хвосты (35L, 1L,
        # 3-pack). Определения (alpine, hiking, camp, sun) носителями не бывают:
        # «сонцезахисні окуляри» цепляются за sun, но очков в каталоге нет,
        # и назвать в ответ Sun Hat — соврать уверенно и быстро.
        self.head_of: dict[str, str] = {}
        for p in store["products"]:
            toks = [w for w in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", p["title"].lower())
                    if w not in {"one", "and", "the", "of", "with"}]
            for w in toks:
                self.tok_of.setdefault(w, set()).add(p["id"])
            core = [w for w in toks if not re.fullmatch(r"\d+l|\d+-pack|\d+", w)]
            self.head_of[p["id"]] = core[-1] if core else toks[-1]

        n = len(self.products)
        self.idf = {t: math.log(n / len(ps)) + 1.0 for t, ps in self.tok_of.items()}

        # Украинская форма → слова каталога, ОДИН КО МНОГИМ. Язык устроен так,
        # что одно слово нередко покрывает несколько товаров: «штани» — это
        # и Rain Pants, и Convertible Trousers. При соответствии один к одному
        # второй товар становился недостижим, и вопрос про брюки-трансформеры
        # уверенно разрешался в дождевые штаны.
        self.ua2en: dict[str, set[str]] = {}
        for en, forms in aliases.get("tokens", {}).items():
            if en not in self.tok_of:
                continue
            for f in forms:
                for w in words(f):
                    k = key(w)
                    if k in FUNCTION_WORDS or len(k) < 3:
                        continue
                    self.ua2en.setdefault(k, set()).add(en)

        # украинская форма цвета → английский цвет
        self.ua2color: dict[str, str] = {}
        for en, forms in aliases.get("colors", {}).items():
            self.ua2color[key(en)] = en
            for f in forms:
                for w in words(f):
                    k = key(w)
                    if k in FUNCTION_WORDS or k in {"кольор", "колір"} or len(k) < 3:
                        continue
                    self.ua2color.setdefault(k, en)

        # устойчивые фразы, которые не собираются из слов названия
        self.phrases: list[tuple[str, frozenset[str]]] = []
        for pid, phrases in aliases.get("product_phrases", {}).items():
            if pid not in self.products:
                continue
            for ph in phrases:
                st = frozenset(key(w) for w in words(ph) if len(w) > 2)
                if st:
                    self.phrases.append((pid, st))

        self.en_tokens = list(self.tok_of)

        # Словарь всего, что система умеет узнавать. Нужен правилу покрытия:
        # слово вопроса, не нашедшееся здесь, — это признак, что спросили
        # о чём-то за пределами наших данных.
        self.vocab: set[str] = set(self.ua2en) | set(self.ua2color) | FUNCTION_WORDS
        self.vocab |= set(self.en_tokens) | COMMON_WORDS
        for _, kws in INTENT_RULES + ORDER_RULES:
            for k in kws:
                self.vocab.update(key(w) for w in k.removeprefix("=").split())
        for _, phrase in self.phrases:
            self.vocab |= set(phrase)
        # Названия стран — тоже слова, которые система умеет узнавать.
        # Без них «Ви відправляєте в Іспанію?» уходил наверх по правилу
        # покрытия: слово «іспанію» не объяснено ничем.
        self.vocab |= GEO_STEMS

    # --- разбор вопроса -------------------------------------------------

    VOLUME_RE = re.compile(r"(\d+)\s*(?:л\b|літр\w*|l\b)")

    def _stems(self, q: str) -> list[str]:
        out = []
        for w in words(q):
            out.append(key(w))
            if "-" in w:                      # «софтшел-куртка» — это два слова
                out.extend(key(part) for part in w.split("-") if part)
        # «35 літрів» и «1 л» — это объём из названия товара (35L, 1L), а не
        # размер и не количество. Без этого рюкзак путается с термосом.
        for num in self.VOLUME_RE.findall(normalize(q)):
            out.append(f"{num}l")
        return out

    def _to_en(self, stems: list[str]) -> dict[str, float]:
        """
        Слова вопроса → слова каталога. Три моста: таблица форм, опечатка
        в форме, прямое совпадение с транслитерацией.

        Побеждает лучший мост, но найденное им слово может быть не одно:
        «штани» ведут сразу к pants и trousers, и оба должны получить вес,
        иначе один из товаров недостижим в принципе.
        """
        found: dict[str, float] = {}
        for s in stems:
            if len(s) < 3:
                continue
            hits: dict[str, float] = {}
            if s in self.ua2en:                                   # таблица форм
                for en in self.ua2en[s]:
                    hits[en] = 1.0
            else:
                best_key, best_sc = None, 0.0
                for ua in self.ua2en:                             # форма с опечаткой
                    sc = similar(s, ua)
                    if sc >= MATCH and sc > best_sc:
                        best_key, best_sc = ua, sc
                if best_key is not None:
                    for en in self.ua2en[best_key]:
                        hits[en] = best_sc
                lat = translit(s)
                for en in self.en_tokens:                         # прямое и транслит
                    sc = max(similar(s, en), similar(lat, en))
                    if sc >= MATCH and sc > hits.get(en, 0.0):
                        hits[en] = sc
            for en, sc in hits.items():
                if sc > found.get(en, 0.0):
                    found[en] = sc
        return found

    def _slots(self, q: str, res: Resolution) -> None:
        norm = normalize(q)
        stems = self._stems(q)

        m = MONEY_RE.search(norm)
        if m:
            res.money = float(m.group(1).replace(",", "."))

        for w in ("два", "дві", "2", "двох", "пара"):
            if w in stems or w in norm.split():
                res.qty = 2
                break

        sz = SIZE_LETTER_RE.search(norm)
        if sz:
            g = sz.group(1)
            res.size = CYR_SIZE.get(g, g.upper())
        else:
            num = SIZE_NUM_RE.search(norm)
            # «35 літрів» — это объём рюкзака, а не размер
            if num and not re.search(num.group(1) + r"\s*(?:л\b|літр)", norm):
                res.size = num.group(1)

        for s in stems:
            if s in self.ua2color:
                res.color = self.ua2color[s]
                break
            for ua, en in self.ua2color.items():
                if similar(s, ua) >= MATCH:
                    res.color = en
                    break
            if res.color:
                break

    def _fuzzy_hit(self, kw: str, stems: list[str]) -> bool:
        """Ключевое слово против слов вопроса, с допуском на опечатку."""
        kw = kw.removeprefix("=")
        if " " in kw:          # составные обороты сверяем только точно
            return False
        return any(similar(s, kw) >= MATCH for s in stems if len(s) >= 4)

    def _intent(self, q: str, has_order: bool) -> tuple[str, str]:
        """
        Намерение в два прохода: сначала точное вхождение по всем правилам,
        и только если не нашлось ничего — приблизительное сравнение.

        Так опечатка («терк-номер» вместо «трек-номер») не теряет намерение,
        но за это не платят вопросы, набранные без ошибок: на них второй проход
        не запускается вовсе. Порядок проходов задаёт и приоритет — точное
        совпадение по любому правилу сильнее приблизительного по любому другому,
        иначе правило, стоящее выше по списку, перехватывало бы чужие вопросы.

        Возвращаем намерение вместе с происхождением, потому что дальше эти два
        случая расходятся: по слову — отвечаем сами, дефолтом — отдаём выше.

        Знак вопроса больше НЕ считается признаком вопроса о наличии. Прежний
        дефолт `product_stock if "?"` объявлял вопросом о наличии вообще любой
        нераспознанный вопрос и отвечал на него уверенно и неверно: «Чи тепла
        куртка на -10?» получало «так, є в наявності». Теперь у наличия свои
        слова (см. INTENT_RULES), а непонятое остаётся `unknown` и уходит выше.
        Единственный сохранённый дефолт — статус заказа: там основанием служит
        не знак вопроса, а сам найденный номер.
        """
        norm = normalize(q)
        # Передача человеку идёт ПЕРВОЙ. Иначе «Поміняйте адресу доставки
        # в #1003» перехватывается правилом `order_address` и получает шаблон,
        # который сообщает текущий адрес в ответ на просьбу его сменить —
        # худшее место всего быстрого пути.
        handoff = HANDOFF_ALWAYS_PATTERNS + (
            HANDOFF_ON_ORDER_PATTERNS if has_order else [])
        handoff_rules = HANDOFF_ALWAYS + (HANDOFF_ON_ORDER if has_order else [])

        patterns = handoff + (ORDER_PATTERNS if has_order else INTENT_PATTERNS)

        for name, pats in patterns:
            if any(p.search(norm) for p in pats):
                return name, "exact"

        stems = self._stems(q)
        rules = handoff_rules + (ORDER_RULES if has_order else INTENT_RULES)
        for name, kws in rules:
            if any(self._fuzzy_hit(k, stems) for k in kws):
                return name, "fuzzy"

        return ("order_status" if has_order else "unknown"), "default"

    def _match_products(self, q: str, res: Resolution) -> list[tuple[str, float]]:
        stems = self._stems(q)
        en = self._to_en(stems)
        scores: dict[str, float] = {}
        named: set[str] = set()     # товары, которые в вопросе действительно названы
        hits: dict[str, int] = {}   # сколько слов названия совпало
        for tok, conf in en.items():
            for pid in sorted(self.tok_of.get(tok, ())):
                scores[pid] = scores.get(pid, 0.0) + self.idf[tok] * conf
                hits[pid] = hits.get(pid, 0) + 1
                if tok == self.head_of[pid]:
                    named.add(pid)
        # Вещь называют и по модели, без родового слова: «Trail Runner 41» —
        # ни одного носителя (shoes), но два слова названия подряд. Требовать
        # непременно носитель значит не понимать половину живых формулировок.
        # Просто «уникальное для одного товара» тут не подходит: sun тоже
        # уникален для Sun Hat, и «сонцезахисні окуляри» вернулись бы.
        named |= {pid for pid, c in hits.items() if c >= 2}

        stemset = set(stems)
        for pid, phrase in self.phrases:            # устойчивая фраза целиком
            if phrase <= stemset:
                scores[pid] = scores.get(pid, 0.0) + 1.5 * len(phrase)
                # Фраза называет товар целиком («термопляшка»), и разбирать её
                # на части нечего — это такое же именование, как слово-носитель.
                named.add(pid)

        # Определения оставляем в весе — они различают товары с общим носителем
        # (Alpine Shell Jacket против Softshell Jacket), — но сами по себе
        # товар опознать не могут.
        scores = {pid: s for pid, s in scores.items() if pid in named}

        if not scores:
            # Второй проход, тот же приём, что и с намерениями: цельные фразы
            # сверяем с допуском на опечатку. «термобіилзна» — переставлены две
            # буквы, и точное вхождение слова фразы обнуляется целиком, хотя
            # для отдельных слов такая опечатка давно прощается.
            # Проход запускается, только если не нашлось вообще ничего, поэтому
            # на исправных вопросах он не выполняется и стоит ноль.
            long_stems = [s for s in stemset if len(s) >= 4]
            for pid, phrase in self.phrases:
                if all(any(similar(w, s) >= MATCH for s in long_stems)
                       for w in phrase):
                    scores[pid] = scores.get(pid, 0.0) + 1.5 * len(phrase)

        if not scores:
            return []
        # При равных весах порядок решает id товара, а не хеш-затравка Python.
        # Множества строк обходятся в непредсказуемом порядке (PYTHONHASHSEED
        # случаен при каждом запуске), поэтому без явного правила ничья
        # разрешалась по-разному от прогона к прогону: один и тот же вопрос
        # давал то p-001, то p-007. Для системы, которую запускают у себя
        # и сверяют с нашим results.jsonl, это недопустимо.
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))

        # цена, названная в вопросе, снимает неоднозначность («шапка за 25 євро»)
        if res.money is not None and len(ranked) > 1:
            exact = [(p, s) for p, s in ranked
                     if abs(self.products[p]["price"] - res.money) < 0.01]
            if exact:
                return exact
        return ranked

    def _variants(self, pid: str, res: Resolution) -> list[str]:
        """
        Все варианты товара, подходящие под названные размер и цвет.

        Их может оказаться несколько, и это не ошибка, а обычный случай:
        «светр M є?» — покупатель не назвал цвет, а M бывает и тёмно-синий,
        и бежевый. Раньше здесь возвращалось «ничего», и вопрос уходил
        в модель на ровном месте; теперь список отдаётся целиком, а решение
        принимает шаблон ответа:

            все подходящие в наличии → «так, є»
            ни одного               → «ні, немає»
            расходятся              → назвать цвета поимённо

        Так уточняющий вопрос покупателю не нужен ни в одном случае каталога,
        а лишний ход диалога в голосовом контуре стоит дороже всего.
        """
        want_size = (res.size or "").lower()
        want_color = (res.color or "").lower()
        if not (want_size or want_color):
            return []
        out = []
        for v in self.products[pid]["variants"]:
            raw = v["variant"]
            if "/" in raw:
                size, color = (x.strip().lower() for x in raw.split("/", 1))
            else:
                size, color = raw.strip().lower(), ""
            if want_size and want_size != size and not (
                    "-" in size and want_size in size.split("-")):
                continue
            if want_color and want_color != color:
                continue
            out.append(raw)
        return out

    def _known_word(self, k: str) -> bool:
        if len(k) < 3 or re.fullmatch(r"[\d\-]+[a-zа-яіїєґ]*", k):
            return True                       # числа, размеры, объёмы — не слова
        if k in self.vocab:
            return True
        return any(similar(k, v) >= MATCH for v in self.vocab)

    def _unexplained(self, q: str) -> list[str]:
        """
        Содержательные слова, которых система не знает вовсе.

        Это ворота «отвечаем сами» в положительной форме: не перечень того,
        чего мы не умеем (список бесконечен и растёт под каждый новый пример),
        а условие, при котором вопрос разобран ЦЕЛИКОМ. «Чи є гарантія
        на куртку?» его не проходит: товар нашёлся, намерение нашлось,
        а «гарантія» не объяснена ничем — значит спросили не о наличии.
        """
        out = []
        for w in words(q):
            parts = [w] + w.split("-") if "-" in w else [w]
            if not any(self._known_word(key(p)) for p in parts):
                out.append(w)
        return out

    # --- точка входа ----------------------------------------------------

    def resolve(self, q: str) -> Resolution:
        res = Resolution()
        norm = normalize(q)

        m = (ORDER_RE.search(norm) or ORDER_WORD_RE.search(norm)
             or ORDER_BARE_RE.search(norm))
        order_no = f"#{m.group(1)}" if m else None
        res.intent, res.intent_src = self._intent(q, has_order=bool(m))

        res.unexplained = self._unexplained(q)

        # Передача человеку не зависит ни от товара, ни от политики: решение
        # уже принято словом. Номер заказа сохраняем — оператору он нужен.
        if res.intent in HANDOFF_INTENTS:
            if m and order_no in self.orders:
                res.order = order_no
            res.reason = "нужен оператор"
            return res

        if m:
            if order_no in self.orders:
                res.order = order_no
                res.reason = "номер заказа найден"
            else:
                res.reason = f"заказа {order_no} нет в данных"
            return res

        if res.intent.startswith("policy_"):
            res.policy = res.intent.removeprefix("policy_")
            res.reason = "политика по ключевому слову"

        if res.intent == "out_of_scope":
            res.reason = "вне области ответственности"
            return res

        self._slots(q, res)

        # доставка + товар в одном вопросе: нужны оба
        needs_product = res.intent.startswith("product_") or (
            res.intent == "policy_shipping" and res.money is None)

        ranked = self._match_products(q, res) if res.intent != "policy_returns" else []
        if ranked:
            top, top_score = ranked[0]
            rivals = [p for p, s in ranked[1:] if s * AMBIGUOUS_GAP > top_score]
            if res.intent == "product_compare":
                res.products = [p for p, s in ranked[:2]]
            elif rivals:
                res.products = [top]
                res.ambiguous = rivals
                res.reason = "несколько кандидатов с близким весом"
            else:
                res.products = [top]

            if res.products and (res.size or res.color):
                res.variants = self._variants(res.products[0], res)
                res.variant = res.variants[0] if len(res.variants) == 1 else None

            if res.policy is None and res.intent.startswith("product_") and \
                    re.search(r"доставк|експрес|шип", norm):
                res.policy = "shipping"   # «товар + доставка» — составной вопрос
        elif needs_product and not res.policy:
            res.reason = "товар не опознан"

        return res


def load(root: Path = ROOT) -> Resolver:
    store = json.loads((root / "handout" / "store.json").read_text(encoding="utf-8"))
    aliases = json.loads((root / "index" / "aliases.json").read_text(encoding="utf-8"))
    return Resolver(store, aliases)
