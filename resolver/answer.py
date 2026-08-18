"""
Ответ по разбору: текст покупателю либо честный отказ.

Слой устроен как ОДНА таблица, а не две. Для каждого намерения записано
и что оно обязано иметь на входе, и как из этого собирается текст — потому что
это один и тот же вопрос, заданный с двух сторон. Разводить их значило бы
завести два списка намерений, которые немедленно разойдутся.

Три исхода, и они разного происхождения:

  ANSWER    собрали текст из данных;
  ABSTAIN   данных нет В САМОМ КАТАЛОГЕ (дата поставки, чужой заказ) —
            это правильный ответ, а не сбой;
  ESCALATE  не поняли вопрос или не опознали сущность — отдаём модели.

Отказ по недоступности модели сюда не входит вовсе: это не ответ, а сбой
прогона, и живёт он в раннере.

Язык ответа берём от языка вопроса: файл вопросов двуязычный по требованию
заказчика, и отвечать украинским на английский вопрос — тот же дефект,
что ответить неверно.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .normalize import normalize
from .geo import destination
from .resolve import Resolution

CYRILLIC = re.compile(r"[а-яіїєґё]")

# Коды стран из заказов. В данных лежит только код, улицы нет вообще —
# поэтому ответ про адрес честно неполный, выдумывать нечего.
#
# Здесь весь Евросоюз, а не те 15 кодов, что встретились в store.json, и это
# не подгонка под примеры: политика магазина возит только по ЕС («за межі
# Європейського Союзу ми не доставляємо»), то есть множество замкнуто ГРАНИЦЕЙ
# ДОМЕНА, известной заранее, а не перечнем увиденных значений. Собранный по
# данным словарь уже дал сбой: Греции в нём не было, и #1015 отвечал
# «їде до GR (GR)».
COUNTRY = {
    "AT": ("Австрії", "Austria"),      "BE": ("Бельгії", "Belgium"),
    "BG": ("Болгарії", "Bulgaria"),    "HR": ("Хорватії", "Croatia"),
    "CY": ("Кіпру", "Cyprus"),         "CZ": ("Чехії", "Czechia"),
    "DK": ("Данії", "Denmark"),        "EE": ("Естонії", "Estonia"),
    "FI": ("Фінляндії", "Finland"),    "FR": ("Франції", "France"),
    "DE": ("Німеччини", "Germany"),    "GR": ("Греції", "Greece"),
    "HU": ("Угорщини", "Hungary"),     "IE": ("Ірландії", "Ireland"),
    "IT": ("Італії", "Italy"),         "LV": ("Латвії", "Latvia"),
    "LT": ("Литви", "Lithuania"),      "LU": ("Люксембургу", "Luxembourg"),
    "MT": ("Мальти", "Malta"),         "NL": ("Нідерландів", "Netherlands"),
    "PL": ("Польщі", "Poland"),        "PT": ("Португалії", "Portugal"),
    "RO": ("Румунії", "Romania"),      "SK": ("Словаччини", "Slovakia"),
    "SI": ("Словенії", "Slovenia"),    "ES": ("Іспанії", "Spain"),
    "SE": ("Швеції", "Sweden"),
}

STATUS = {
    "delivered":  ("доставлене", "delivered"),
    "shipped":    ("відправлене", "shipped"),
    "processing": ("в обробці", "being processed"),
    "cancelled":  ("скасоване", "cancelled"),
}


@dataclass
class Reply:
    decision: str      # answer | abstain | escalate
    text: str = ""
    why: str = ""      # служебное, покупателю не показываем


def _money(x: float) -> str:
    """89.0 → «89», 12.5 → «12.50». В голосовом контуре «крапка нуль» лишняя."""
    return f"{x:.0f}" if abs(x - round(x)) < 0.005 else f"{x:.2f}"


def _lang(q: str) -> str:
    return "ua" if CYRILLIC.search(normalize(q)) else "en"


class Answerer:
    def __init__(self, store: dict):
        self.products = {p["id"]: p for p in store["products"]}
        self.orders = {o["name"]: o for o in store["orders"]}
        self.policies = store["policies"]
        self.currency = store.get("currency", "EUR")

    # --- ворота ---------------------------------------------------------

    def gate(self, res: Resolution) -> Reply | None:
        """
        Контракт: что намерение обязано иметь, чтобы ответ вообще собрался.
        Возвращает Reply, если отвечать нельзя, и None, если можно.
        """
        i = res.intent

        if i == "unknown":
            return Reply("escalate", why="намерение не опознано")

        # Дефолт означает «ни одно правило не сработало». Единственное
        # исключение — статус заказа: там основанием служит найденный номер,
        # а не отсутствие признаков.
        if res.intent_src == "default" and i != "order_status":
            return Reply("escalate", why=f"намерение {i} назначено дефолтом")

        if i == "out_of_scope":
            # Наверх НЕ отдаём никогда. Замер показал почему: на «дай знижку
            # 20 відсотків» модель повторяет процент в ответе, а наш шаблон
            # намеренно его не называет — иначе в ответе появляется число,
            # которого магазин не обещал. Есть вопросы, где правильный ответ
            # требует сдержанности, и это как раз они.
            return None

        if i == "product_restock":
            # Дат поставки в каталоге нет ни одной. Отправлять такой вопрос
            # в модель нельзя: она их выдумает — это и есть ловушка q012.
            # Проверяем ДО правила покрытия: отказ здесь безопаснее эскалации.
            return Reply("abstain", why="дат поставки нет в каталоге")

        # Правило покрытия. Ворота сформулированы положительно: отвечаем сами,
        # только если вопрос разобран целиком. Это замена перечню частных
        # запретов — тот рос бы под каждый новый пример и всё равно отставал
        # от чужого файла вопросов.
        if res.unexplained:
            return Reply("escalate",
                         why=f"не объяснены слова: {', '.join(res.unexplained)}")

        if i.startswith("order_"):
            return None                      # «заказа нет» — тоже ответ

        if i.startswith("policy_"):
            return None                      # политики есть всегда

        if i == "product_compare":
            return None if len(res.products) == 2 else Reply(
                "escalate", why="для сравнения нужно два товара")

        if i.startswith("product_"):
            if not res.products:
                return Reply("escalate", why="товар не опознан")
            if res.ambiguous:
                return Reply("escalate", why="несколько кандидатов с близким весом")
            return None

        return Reply("escalate", why="намерение без правила ответа")

    # --- точка входа ----------------------------------------------------

    def reply(self, res: Resolution, q: str) -> Reply:
        stop = self.gate(res)
        if stop is not None:
            if stop.decision == "abstain" and not stop.text:
                stop.text = self._no_data(_lang(q))
            return stop

        lang, norm = _lang(q), normalize(q)
        handler = getattr(self, "_" + res.intent, None)
        if handler is None:
            return Reply("escalate", why=f"нет обработчика для {res.intent}")
        return handler(res, lang, norm)

    # --- служебные тексты ----------------------------------------------

    def _no_data(self, lang: str) -> str:
        """Данных нет в магазине. Это ПРАВИЛЬНЫЙ ответ, а не сбой."""
        return ("На жаль, таких даних у мене немає. З’єднати вас з оператором?"
                if lang == "ua" else
                "I don't have that information. Shall I connect you to an agent?")

    def _not_in_time(self, lang: str) -> str:
        """
        Не уложились в таймаут. Другой случай и другой текст.

        Раньше сюда шёл `_no_data`, и по `results.jsonl` нельзя было отличить
        «в магазине этого нет» от «наш таймер сработал раньше модели». Первое
        говорит о магазине, второе — о нас, и склеивать их значит писать
        в отчёт неправду: часть «не знаю» была не отказом системы, а её
        секундомером.

        Почему текст именно такой, а не филлер «Секунду, дивлюся». В голосовом
        контуре здесь стоял бы именно филлер: он не успевает в срок, а меняет
        срок — пока агент молчит, покупатель считает секунды молчания, а как
        только заговорил, считает секунды проверки. Но в файле реплика обязана
        быть законченной, незавершённая выглядит обрывом. Поэтому текст
        называет причину и предлагает выход, оставаясь целым ответом.
        """
        return ("Не встиг перевірити це достатньо швидко. З’єднати вас "
                "з оператором, він відповість точно?"
                if lang == "ua" else
                "I couldn't check that quickly enough. Shall I connect you "
                "to an agent who can answer precisely?")

    def _title(self, pid: str) -> str:
        return self.products[pid]["title"]

    # --- товары ---------------------------------------------------------

    def _product_price(self, res: Resolution, lang: str, norm: str) -> Reply:
        p = self.products[res.products[0]]
        price = _money(p["price"])
        if lang == "ua":
            return Reply("answer", f"{p['title']} коштує {price} {self.currency}.")
        return Reply("answer", f"{p['title']} costs {price} {self.currency}.")

    def _product_stock(self, res: Resolution, lang: str, norm: str) -> Reply:
        p = self.products[res.products[0]]
        title = p["title"]
        by_variant = {v["variant"]: v["stock"] for v in p["variants"]}

        # Размер или цвет названы — отвечаем про подходящие варианты.
        if res.size or res.color:
            if not res.variants:
                # Вариант не существует вовсе — это не то же самое, что нулевой
                # остаток, и покупателю разница важна.
                want = " / ".join(x for x in (res.size, res.color) if x)
                return Reply("answer", (
                    f"Варіанта {want} у моделі {title} немає взагалі."
                    if lang == "ua" else
                    f"There is no {want} variant of {title} at all."))
            avail = [(v, by_variant[v]) for v in res.variants if by_variant[v] > 0]
            if not avail:
                return Reply("answer", (
                    f"Ні, {title} у цьому варіанті зараз немає в наявності."
                    if lang == "ua" else
                    f"No, {title} is out of stock in that variant."))
            if len(avail) == 1:
                v, n = avail[0]
                return Reply("answer", (
                    f"Так, {title} {v} є — {n} шт."
                    if lang == "ua" else
                    f"Yes, {title} {v} is in stock — {n} left."))
            # Подходящих несколько (назвали размер без цвета) — перечисляем,
            # а не переспрашиваем: лишний ход диалога в голосе дороже всего.
            lst = ", ".join(f"{v} — {n}" for v, n in avail)
            return Reply("answer", (
                f"Так, {title} є: {lst}."
                if lang == "ua" else
                f"Yes, {title} is available: {lst}."))

        # Размер не назван — общая картина по товару.
        avail = [(v, n) for v, n in by_variant.items() if n > 0]
        if not avail:
            return Reply("answer", (
                f"Ні, {title} зараз немає в наявності."
                if lang == "ua" else f"No, {title} is out of stock."))
        lst = ", ".join(f"{v} — {n}" for v, n in avail)
        return Reply("answer", (
            f"{title} є в наявності: {lst}."
            if lang == "ua" else f"{title} is in stock: {lst}."))

    def _product_compare(self, res: Resolution, lang: str, norm: str) -> Reply:
        a, b = (self.products[p] for p in res.products)
        cheap, dear = (a, b) if a["price"] <= b["price"] else (b, a)
        if lang == "ua":
            return Reply("answer",
                f"Дешевше {cheap['title']} — {_money(cheap['price'])} {self.currency}, "
                f"проти {_money(dear['price'])} за {dear['title']}.")
        return Reply("answer",
            f"{cheap['title']} is cheaper — {_money(cheap['price'])} {self.currency}, "
            f"versus {_money(dear['price'])} for {dear['title']}.")

    # --- заказы ---------------------------------------------------------

    def _order(self, res: Resolution) -> dict | None:
        return self.orders.get(res.order) if res.order else None

    def _unknown_order(self, lang: str) -> Reply:
        # Номер прозвучал, но такого заказа нет. Отдельный ответ, а не «не знаю»:
        # это установленный факт, а не пробел в данных.
        return Reply("answer", (
            "Замовлення з таким номером у системі немає. Перевірте, будь ласка, номер."
            if lang == "ua" else
            "There is no order with that number. Could you double-check it?"))

    def _status_word(self, o: dict, lang: str) -> str | None:
        """Статус словами — или None, если такого статуса мы не знаем."""
        v = STATUS.get(o.get("status"))
        return None if v is None else v[0 if lang == "ua" else 1]

    def _unknown_value(self, field: str, value) -> Reply:
        """
        В данных значение, которого наши словари не знают, — отдаём наверх.

        Раньше на его месте стояла подстановка «как есть», и это давало
        «Замовлення #1001 — refunded» посреди украинского ответа и «їде до
        GR (GR)» с кодом вместо страны. Подстановка выглядела безобидно ровно
        потому, что в нашем файле незнакомых значений не было; в их файле они
        будут — статусов у Shopify больше четырёх.

        Отправляем наверх по тому же правилу, на котором стоит вся
        конструкция: отвечаем сами, только если поняли вопрос целиком.
        Незнакомое значение в данных — это «не целиком». Цена ошибки
        несимметрична: наверх — 400 мс внутри бюджета, сами — сырое английское
        слово покупателю.
        """
        return Reply("escalate", why=f"незнакомое значение {field}={value!r}")

    def _order_status(self, res: Resolution, lang: str, norm: str) -> Reply:
        o = self._order(res)
        if o is None:
            return self._unknown_order(lang)
        st = self._status_word(o, lang)
        if st is None:
            return self._unknown_value("status", o.get("status"))
        if lang == "ua":
            return Reply("answer", f"Замовлення {o['name']} — {st}.")
        return Reply("answer", f"Order {o['name']} is {st}.")

    def _order_tracking(self, res: Resolution, lang: str, norm: str) -> Reply:
        o = self._order(res)
        if o is None:
            return self._unknown_order(lang)
        if not o.get("tracking"):
            st = self._status_word(o, lang)
            if st is None:
                return self._unknown_value("status", o.get("status"))
            return Reply("answer", (
                f"Трек-номера ще немає — замовлення {o['name']} {st}."
                if lang == "ua" else
                f"No tracking number yet — order {o['name']} is {st}."))
        if lang == "ua":
            return Reply("answer", f"Трек-номер замовлення {o['name']}: {o['tracking']}.")
        return Reply("answer", f"Tracking number for {o['name']}: {o['tracking']}.")

    def _order_items(self, res: Resolution, lang: str, norm: str) -> Reply:
        o = self._order(res)
        if o is None:
            return self._unknown_order(lang)
        parts = [f"{it['title']} {it['variant']}"
                 + (f" ×{it['qty']}" if it.get("qty", 1) > 1 else "")
                 for it in o["items"]]
        lst = "; ".join(parts)
        if lang == "ua":
            return Reply("answer", f"У замовленні {o['name']}: {lst}.")
        return Reply("answer", f"Order {o['name']} contains: {lst}.")

    def _order_address(self, res: Resolution, lang: str, norm: str) -> Reply:
        o = self._order(res)
        if o is None:
            return self._unknown_order(lang)
        code = o.get("ships_to")
        pair = COUNTRY.get(code)
        if pair is None:
            return self._unknown_value("ships_to", code)
        name = pair[0 if lang == "ua" else 1]
        # Ответ неполный по устройству данных, и об этом сказано прямо.
        if lang == "ua":
            return Reply("answer",
                f"Замовлення {o['name']} їде до {name} ({code}). "
                f"Повної адреси я не бачу — її покаже оператор.")
        return Reply("answer",
            f"Order {o['name']} ships to {name} ({code}). "
            f"I can't see the full street address — an agent can.")

    # --- политики -------------------------------------------------------

    def _policy_returns(self, res: Resolution, lang: str, norm: str) -> Reply:
        if any(k in norm for k in ("поносив", "носив", "прат", "worn", "washed", "used")):
            return Reply("answer", (
                "Ні. Повернути можна лише неношені та непрані речі з бирками."
                if lang == "ua" else
                "No. Items must be unworn, unwashed and with original tags attached."))
        if any(k in norm for k in ("коли", "гроші", "when", "refund")):
            return Reply("answer", (
                "Гроші повертаємо на той самий спосіб оплати протягом "
                "5 робочих днів після надходження посилки на склад."
                if lang == "ua" else
                "Refunds go back to the original payment method within "
                "5 business days of the parcel reaching our warehouse."))
        if any(k in norm for k in ("хто платит", "пересилк", "who pays", "shipping")):
            return Reply("answer", (
                "Зворотну пересилку оплачує покупець — крім випадків браку "
                "або помилки магазину, тоді платимо ми."
                if lang == "ua" else
                "Return shipping is paid by the customer, except for faulty "
                "or incorrectly sent items, where we cover it."))
        return Reply("answer", (
            "Повернення — протягом 14 днів з дня доставки. Річ має бути "
            "неношена, непрана, з бирками."
            if lang == "ua" else
            "Returns are accepted within 14 days of delivery. Items must be "
            "unworn, unwashed and with original tags attached."))

    def _policy_exchange(self, res: Resolution, lang: str, norm: str) -> Reply:
        return Reply("answer", (
            "Один безкоштовний обмін розміру на замовлення — протягом 14 днів "
            "з дня доставки, якщо потрібний розмір є в наявності. "
            "Обмін кольору оформлюється як повернення плюс нове замовлення."
            if lang == "ua" else
            "One free size exchange per order within 14 days of delivery, "
            "provided the requested size is in stock. Colour exchanges are "
            "treated as a return plus a new order."))

    def _policy_damaged(self, res: Resolution, lang: str, norm: str) -> Reply:
        return Reply("answer", (
            "Повідомте про пошкодження протягом 48 годин після доставки і "
            "додайте фото. Зворотну пересилку оплачуємо ми і надсилаємо заміну, "
            "а якщо товару немає — повертаємо гроші повністю."
            if lang == "ua" else
            "Report the damage within 48 hours of delivery with a photo. "
            "We cover return shipping and send a replacement, or refund in "
            "full if the item is out of stock."))

    def _policy_payment(self, res: Resolution, lang: str, norm: str) -> Reply:
        if any(k in norm for k in ("наложен", "післяплат", "готівк", "cash on delivery")):
            return Reply("answer", (
                "Ні, накладеного платежу немає. Приймаємо Visa, Mastercard, "
                "Apple Pay, Google Pay і PayPal."
                if lang == "ua" else
                "No, we do not offer cash on delivery. We accept Visa, "
                "Mastercard, Apple Pay, Google Pay and PayPal."))
        if any(k in norm for k in ("коли", "спишут", "списан", "when", "charged")):
            return Reply("answer", (
                "Гроші з картки списуємо під час відправки замовлення, "
                "а не при оформленні."
                if lang == "ua" else
                "Cards are charged when the order is dispatched, not when "
                "it is placed."))
        return Reply("answer", (
            "Приймаємо Visa, Mastercard, Apple Pay, Google Pay і PayPal."
            if lang == "ua" else
            "We accept Visa, Mastercard, Apple Pay, Google Pay and PayPal."))

    # --- доставка: единственная политика с арифметикой -------------------

    STANDARD, EXPRESS, FREE_FROM = 12.0, 29.0, 150.0

    def _policy_shipping(self, res: Resolution, lang: str, norm: str) -> Reply:
        express = bool(re.search(r"експрес|express", norm))

        # 1. Составной вопрос «товар + доставка» — считаем сумму.
        if res.products:
            p = self.products[res.products[0]]
            goods = p["price"] * res.qty
            ship = 0.0 if (not express and goods > self.FREE_FROM) else (
                self.EXPRESS if express else self.STANDARD)
            total = goods + ship
            g, s, t = _money(goods), _money(ship), _money(total)
            if ship == 0:
                return Reply("answer", (
                    f"Разом {t} {self.currency}: {g} за товар, доставка "
                    f"безкоштовна від {_money(self.FREE_FROM)}."
                    if lang == "ua" else
                    f"{t} {self.currency} in total: {g} for the goods, "
                    f"shipping is free over {_money(self.FREE_FROM)}."))
            kind = ("експрес-доставка" if express else "стандартна доставка") \
                if lang == "ua" else ("express delivery" if express else "standard delivery")
            return Reply("answer", (
                f"Разом {t} {self.currency}: {g} за товар плюс {s} — {kind}."
                if lang == "ua" else
                f"{t} {self.currency} in total: {g} for the goods plus {s} for {kind}."))

        # 2. Страна назначения вне ЕС — отвечаем раньше цен и сроков, потому
        #    что она их отменяет: на «скільки коштує доставка в Аргентину»
        #    правильный ответ «не возимо», а не «12 EUR».
        #
        #    Раньше здесь стояло `британ|britain|uk` — страна, вписанная руками
        #    под единственный видимый вопрос. В их файле будет другая.
        #    Оборот `за меж|outside` оставлен: это про саму границу, не про
        #    страну.
        #
        #    Проверка страны живёт ВНУТРИ вопроса о доставке и только здесь.
        #    Снаружи она давала бы ложные срабатывания на обычных словах:
        #    «дані» неотличимо от «Данія», «китайський» от «Китай»,
        #    «індійський» от «Індія». Намерение — и есть тот контекст,
        #    который делает слово страной.
        dest = destination(norm)
        if dest == "non_eu" or re.search(r"за меж|outside", norm):
            return Reply("answer", (
                "Ні, за межі Європейського Союзу ми не доставляємо."
                if lang == "ua" else
                "No, we do not ship outside the European Union."))

        # 3. Порог безкоштовної доставки.
        if re.search(r"безкоштовн|безплатн|free", norm):
            return Reply("answer", (
                f"Доставка безкоштовна для замовлень від "
                f"{_money(self.FREE_FROM)} {self.currency}."
                if lang == "ua" else
                f"Shipping is free on orders over {_money(self.FREE_FROM)} "
                f"{self.currency}."))

        # 4. Сроки.
        if re.search(r"скільки йде|скільки днів|як довго|коли прийде|how long|takes|days", norm):
            return Reply("answer", (
                ("Експрес-доставка — 1-2 робочі дні." if express else
                 "Стандартна доставка по ЄС — 3-5 робочих днів.")
                if lang == "ua" else
                ("Express delivery takes 1-2 business days." if express else
                 "Standard delivery across the EU takes 3-5 business days.")))

        # 5. Страна из ЕС и ничего конкретнее не спросили — отвечаем «возимо».
        #    Стоит ПОСЛЕ сроков и порога бесплатной доставки, а не рядом
        #    с проверкой не-ЕС: на «скільки йде доставка в Іспанію» человек
        #    спрашивает про срок, а не про сам факт. Для не-ЕС такого выбора
        #    нет — там факт отменяет вопрос.
        if dest == "eu":
            return Reply("answer", (
                f"Так, доставляємо. Стандартна доставка по ЄС — 3-5 робочих "
                f"днів, {_money(self.STANDARD)} {self.currency}, "
                f"а від {_money(self.FREE_FROM)} безкоштовно."
                if lang == "ua" else
                f"Yes, we do. Standard delivery across the EU takes 3-5 "
                f"business days and costs {_money(self.STANDARD)} "
                f"{self.currency}, free over {_money(self.FREE_FROM)}."))

        # 6. Стоимость.
        cost = self.EXPRESS if express else self.STANDARD
        kind = ("Експрес-доставка" if express else "Стандартна доставка") \
            if lang == "ua" else ("Express delivery" if express else "Standard delivery")
        if lang == "ua":
            return Reply("answer", f"{kind} коштує {_money(cost)} {self.currency}.")
        return Reply("answer", f"{kind} costs {_money(cost)} {self.currency}.")

    # --- вне области ----------------------------------------------------

    def _out_of_scope(self, res: Resolution, lang: str, norm: str) -> Reply:
        # Скидку не выдумываем и процент из вопроса не повторяем — иначе
        # в ответе появится число, которого магазин не обещал.
        return Reply("answer", (
            "Я не можу надавати знижки. Актуальні ціни — ті, що в каталозі."
            if lang == "ua" else
            "I can't offer discounts. The catalogue prices are the current ones."))
