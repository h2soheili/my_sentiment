import json
import random
import re
import unicodedata

import jdatetime
import jsonlines
import polars as pl
import requests


def fetch_all_data():
    url = "https://bvapi.emofid.com/mobile/telegramMessages?channelId=0"

    def get(message_id=None):
        params = dict(count=1000)
        if message_id is not None:
            params["messageId"] = message_id
        headers = {"session-code": "6d7d1361-6665-440a-a7f0-29271c4a2e01"}
        res = requests.get(url, params=params, headers=headers)
        res = res.json()

        with open('bv_news_crawl.jsonl', 'a+', encoding='utf8') as f:
            for i in res:
                f.write(json.dumps(i, ensure_ascii=False) + "\n")
        return res or []

    total = 0
    rows = get()
    total += len(rows)
    print("count ", total)
    last = rows[-1]
    while last:
        rows = get(last["id"])
        count = len(rows)
        print("count ", count)
        total += count
        if count > 0:
            last = rows[-1]
        else:
            break

    print("end ... total", total)


emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002500-\U00002BEF"  # chinese char
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           u"\U0001f926-\U0001f937"
                           u"\U00010000-\U0010ffff"
                           u"\u2640-\u2642"
                           u"\u2600-\u2B55"
                           u"\u200d"
                           u"\u23cf"
                           u"\u23e9"
                           u"\u231a"
                           u"\ufe0f"  # dingbats
                           u"\u3030"
                           "]+", flags=re.UNICODE)

persian_months = [
    "فروردین",
    "اردیبهشت",
    "خرداد",
    "تیر",
    "مرداد",
    "شهریور",
    "مهر",
    "آبان",
    "آذر",
    "دی",
    "بهمن",
    "اسفند",
]

# dont add ? to this remove_items list
remove_items = [
    "@",
    "#",
    "twitter_bourse",
    "telegram",
    "iroilmarket",
    "twitter",
    "instagram",
    "::",
    "«",
    "»",
    "pdf",
    "marketsummary",
    "solutien fardaname",
    "fardaname",
    "ibena_news",
    "website",
    "iroilmarket",
    "ava.agah.co",
    "agahmoshaver",
    "iroilmarket",
    "office607",
    "ibena",
    "www.",
    ".ir",
    ".com",
    "oilmarket",
    "afzayeshs",
    "‏",
    "news",
    "gandomfinance",
    "commolady",
    "!",
    "دانلود",
    "لینک",
    "فیلم",
    "ویدیو",
    "عکس",
    "تصویر",
    "در ساعت",
    "امروز",
    "روزانه",
    "مورخه",
    "مورخ",
    "بلومبرگ فارسی",
    "کانال تحلیلی پارسیس",
    "اختصاصی بازار نفت گاز پتروشیمی",
    "ادامه مطلب",
    "کلیک نمایید",
    "اکوایران",
    "تسنیم",
    "در پست بعدی",
    "در مطلب بعدی",
    "پست بعدی",
    "مطلب بعدی",
    "base_oil",
    "…",
    "sulfur",
    "shahrebours",
    "karoacademy",
    "تجارت نیوز",
    "تجارت  نیوز",
    "000 gqu",
    "gqu 000",
    "padash",
    "tazehayeeghtesad",
    "farabourse",
    "media",
    "--",
    ":",
    ";",
    "::",
    "#اختصاصی",
    " ",
    "000gss",
    " ",
    "solutien",
    "⃣",
    "  ",
    "   ",
    "​",
    "000grs",
    "�",
    '"',
    "بورس نیوز",
    "⏰",
    "⏺",
    "‼",
    "!",
    "st geopoliticintel",
    "dt geopoliticintel",
    "geopoliticintel",
    "کانال ما را در اینستاگرام دنبال کنید",
    "*",
    "|",
    "linkedin",
    "tahlildaro",
    "000btz",
    "validhelalat",
    "⏫",
    "↖",
    "ء",
    "https",
    "http",
    "کارگزاری بانک صنعت و معدن",
    "کارگزاری مفید"
    "کارگزاری آگاه",
    "کانال molady",
    "کارگزاری صنعت و معدن",
    "خبرگزاری مهر",
    "برای مشاهده لطفا کلیک کنید",
    "بیشتر بخوانید",
    "000bpb",
    "000boq",
    "برای مشاهده لطفا کلیک کنید",
    "کلیک نمایید",
    "مرجع آخرین اخبار نفت و محصولات نفتی",
    "مرجع آخرین اخبار",
    "facebook",
    "smartbourse",
    "kaladade",
    "‎",
    "⁩",
    "⁧",
    "‌",
    "سبدگردان ویستا",
    "•",
    "™",
    "↔",
    "↕",
    "↙",
    "↪",
    "⌨",
    "⏪",
    "⏳",
    "⏸",
    "⏹",
    "`",
    "©",
    "«",
    "»",
    "¬",
    "±",
    "×",
    "÷",
    "ø",
    "،",
    "؛",
    "؟",
    "ء",
    "{",
    "}",
    "|",
    "~",
    "‘",
    "’",
    "“",
    "”",
    "•",
    "…",
    "\n",
    "'",
    "\\",
    "­",
    "̇",
    " ",
    " ",
    "⁠",
    "⁦",
]

replace_items = {
    "درصدی": "%",
    "درصد ی": "%",
    "درصد": "%",
    "دلاری": "$",
    "دلار": "$",
    "¢": "$",
    "یورویی": "€",
    "یورو": "€",
    "%": "%",
    "٪": "%",
    "℅": "%",
    "پوندی": "£",
    "پوند": "£",
    "...": " ",
    "..": " ",
    "میلیاردریال": " میلیارد ریال ",
    "میلیاردتومان": " میلیارد تومان ",
    "میلیونریال": " میلیون ریال ",
    "میلیونتومان": " میلیون تومان ",
    "هزارریال": " هزار ریال ",
    "ریالی": "ریال",
    "تومانی": "تومان",
    'بتا سهم': '',
    'بازار نفت گاز پتروشیمی': '',
    'پارسیس': '',
    'خلاصه بازار': '',
    'ایبِنا': '',
    'کامولیدی': '',
    'planner': '',
    'آموزش مفید': '',
    'تحلیل هفتگی بازار سرمایه': '',
    'شرکت کارگزاری مفید': '',
    'ژیوپلیتیک': '',
    'بورس ویو': '',
    'کامودیتی': '',
    'methanol': 'متانول',
    'الفینی': 'الفین',
    'معاملات های': 'معاملات ',
    "_": " ",
    "(": " ",
    ")": " ",
    "oilmarket": "",
    " , ": ",",
    "⁉": "?",
    "میدهد": "می‌دهد",
    "میشدند": "می‌شدند",
    "سهام+حق تقدم": "سهام + حق تقدم",
    "داراییها": "دارایی ها",
    "میکنید": "می کنید",
    "میشوند": "می شوند",
    "بازارهای": "بازار های",
    "سود مجمع داشته است": "سود محقق کرده است",
    "سود مجمع محقق کرده است": "سود محقق کرده است",
    "صورتهای مالی": "صورت‌های مالی",
    "ال ان جی": "lng",
    "lng lng": "lng",
    "میشود": "می‌شود",
    ",": "",
    "٬": "",
    "میایون": "میلیون",
}

for m in persian_months:
    replace_items[f"{m}ماه"] = m
    replace_items[f"{m}ماه "] = m

# print(remove_pattern)

skip_these_channels = ["planner",
                       "آموزش مفید",
                       "تحلیل هفتگی بازار سرمایه",
                       "خلاصه بازار",
                       "شرکت کارگزاری مفید"
                       ]

skip_these_topics = ["کارگاه آموزشی",
                     "تپ سواپ",
                     "شمارش آرا",
                     "انتخابات",
                     "بازی جدید تلگرامی",
                     "بازی جدید تلگرام",
                     "بازی تلگرامی",
                     "بازی تلگرام",
                     "صندوق های رای",
                     "صندوق رای",
                     "رای‌گیری",
                     "برگزار می‌گردد",
                     "برگزار میگردد",
                     "بیشترین ورود نقدینگی حقیقی و حقوقی در کدام صنایع بوده است",
                     "آماده انجام معامله می‌باشند",
                     "سمینار بررسی گزارشات",
                     "سمینار بررسی صورت های مالی",
                     "سمینار رایگان بررسی گزارشات",
                     "با توجه به توقف نماد اصلی متوقف شدند",
                     "با توجه به توقف نماد اصلی متوقف شد",
                     "سمینار",
                     "مجمع عمومی عادی سالیانه",
                     "جلسه مجمع عمومی",
                     "تعلیق می‌گردد"
                     ]

replace_months = {
    "ژانویه": "january",
    "فوریه": "february",
    "مارس": "march",
    "آوریل": "april",
    "ماه می": "may",
    # "می": "may",
    "ژوئن": "juan",
    "ژوین": "juan",
    "جولای": "july",
    "ژوئیه": "july",
    "ژوییه": "july",
    "آگوست": "august",
    "سپتامبر": "september",
    "اکتبر": "october",
    "نوامبر": "november",
    "دسامبر": "december",
}

map_to_standard_char = {'ك': 'ک',
                        'دِ': 'د',
                        'بِ': 'ب',
                        'زِ': 'ز',
                        'ذِ': 'ذ',
                        'شِ': 'ش',
                        'سِ': 'س',
                        'ى': 'ی',
                        'ي': 'ی',
                        '۰': '0',
                        '۱': '1',
                        '۲': '2',
                        '۳': '3',
                        '۴': '4',
                        '۵': '5',
                        '۶': '6',
                        '۷': '7',
                        '۸': '8',
                        '۹': '9',
                        '.': '.',
                        '١': '1',
                        '٢': '2',
                        '٣': '3',
                        '٤': '4',
                        '٥': '5',
                        '٦': '6',
                        '٧': '7',
                        '٨': '8',
                        '٩': '9',
                        "،": ",",
                        "ئ": "ی",
                        "ٱ": "ا",
                        "اً": "ا",
                        "إً": "ا",
                        "أ": "ا",
                        "ؤ": "و",
                        "ة": "ه",
                        "٠": ".",
                        "ı": "1",
                        "₂": "2",
                        "/": ".",
                        }


def to_standard_english(_text):
    normalized = unicodedata.normalize('NFD', _text)
    text = u"".join([c for c in normalized if not unicodedata.combining(c)])
    return text


def change_minus_position(_text):
    text = _text
    text = text.strip()

    if re.match(r"-\d+", text):
        return text.replace("-", "") + "-"
    return text


items_to_add_padding = [
                           "تلفیقی",
                           "تن",
                           "کیلو",
                           "کیلوگرم",
                           "گرم",
                           "تومانی",
                           "تومان",
                           "ریالی",
                           "ریال",
                           "تریلیون",
                           "میلیارد",
                           "میلیون",
                           "هزار",
                           "ماهه",
                           "ماه",
                           "فروش",
                           "خرید",
                       ] + persian_months


def add_padding(_text):
    text = _text
    text = text.strip()

    for i in items_to_add_padding:
        if re.match(f"\d+{i}", text):
            text = text.replace(i, f" {i}")
    return text


def clean_text1(_text):
    if _text is None or _text == "":
        return None
    text = _text
    # link
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    # date
    text = re.sub(r'\d{4}/\d{1,2}/\d{1,2}', '', text)
    text = re.sub(r'\d{4}.\d{1,2}.\d{1,2}', '', text)
    text = re.sub(r'\d{4}-\d{1,2}-\d{1,2}', '', text)
    # time
    text = re.sub(r'\d{1,2}:\d{2}', '', text)
    # emoji
    text = emoji_pattern.sub(r' ', text)
    text = re.sub(r"\s{2,}", " ", text)
    for k, v in map_to_standard_char.items():
        text = text.replace(k, v)
    text = " ".join(map(change_minus_position, text.split(" ")))
    text = " ".join(map(add_padding, text.split(" ")))
    for i in remove_items:
        text = text.replace(i, " ")
    text = re.sub(r"\s{2,}", " ", text)
    for k, v in replace_items.items():
        text = text.replace(k, v)
    text = re.sub(r"\s{2,}", " ", text)
    for topic in skip_these_topics:
        if topic in _text:
            return None
    for k, v in replace_months.items():
        text = text.replace(f"{k}", f" {v} ")
    text = re.sub(r"\s{2,}", " ", text)
    text = to_standard_english(text)
    return text


import hazm

hazm_nor = hazm.Normalizer()


def clean_text0(_text):
    if _text is None or _text == "":
        return None
    text = _text.strip()
    text = re.sub(r"\s{2,}", " ", text)
    text = hazm_nor.normalize(text)
    text = text.replace('|', "\n ")
    text = text.lower()
    arr = text.split('\n')
    arr = map(clean_text1, arr)
    arr = list(filter(lambda x: x is not None, arr))
    if len(arr) == 0:
        return None
    text = " ".join(arr)
    text = clean_text1(text)
    text = hazm_nor.normalize(text or "")
    text = clean_text1(text)
    if text == "" or text is None:
        return None
    text = re.sub(r"\s{2,}", " ", text)
    return text


def bv_news_cleaner_fn(obj):
    channel = (obj['channelName'] or "").lower()
    if channel in skip_these_channels:
        return None
    _date = jdatetime.datetime.strptime(obj["date"], "%Y/%m/%d-%H:%M:%S") if obj["date"] else None
    _date = _date.togregorian().strftime("%Y-%m-%dT%H:%M:%S") if _date else None
    _text = obj["text"]
    _text = clean_text0(_text)
    if _text == "" or _text is None:
        return None
    result = dict()
    result["id"] = obj['id']
    result["date"] = _date
    result["text"] = _text
    result["org_text"] = obj["text"]
    result["channel"] = channel
    r = random.random()
    sentiment = "positive" if r > 0.66 else "negative" if r < 0.33 else "neutral"
    result["sentiment"] = sentiment
    result["target"] = random.randint(0, 4)
    return result


def process_news():
    """"
    {"id": 406956,
    "date": "1403/04/19-13:55:59",
     "text": "دبیر انجمن خودروسازان:\n▪️شرکت های ایران خودرو و سایپا در انتظار ابلاغ قیمت های جدید از سوی وزارت صمت هستند.",
      "hyperText": null,
       "mediaPath": null,
       "IsDeleted": false,
        "channelName": "بتا سهم"}
    """

    with jsonlines.open('./bv_news_crawl.jsonl') as reader:
        rows = [dict(obj) for obj in reader]
        import multiprocessing as mp
        with mp.Pool(8) as pool:
            results = pool.map(bv_news_cleaner_fn, rows)
            results = list(filter(lambda x: x is not None, results))
            # print("results", len(results))
            df = pl.from_records(results)
            df.write_csv('./bv_news.csv')


def process_news2():
    with jsonlines.open('./bv_news_crawl.jsonl') as reader:
        rows = []
        for obj in reader:
            rows.append(bv_news_cleaner_fn(obj))
            if len(rows) > 4000:
                break

        rows = list(filter(lambda x: x is not None, rows))
        df = pl.from_records(rows)
        df.write_csv('./bv_news2.csv')


def extract_vocab():
    import hazm
    df = pl.read_csv('./bv_news.csv')
    vocab = dict()
    for row in df.rows(named=True):
        _text = row["text"]
        _text = _text.replace("\u200c", " ")
        _text = re.sub(r"_|-", " ", _text)
        # print(row["text"])
        # print(_text)
        for t in hazm.word_tokenize(_text):
            t2 = re.sub(r'،|;|:', "", t).replace(".", "")
            t2 = ''.join([i for i in t2 if not i.isdigit()])
            # t3 = t2.split("\u200c")
            # for i in t3:
            #     if len(i) > 0:
            #         vocab[i] = None

            vocab[t2] = None

    d = list(vocab.keys())
    # print(len(d))
    # print(d)
    with open('./d.txt', 'w+', encoding='utf8') as f:
        d2 = "\n".join(d)
        f.write(d2)


def process_news3():
    df = pl.read_csv('./bv_news_.csv')

    texts = df['text'].to_list()
    print(len(texts))
    import multiprocessing as mp
    from ai.tiktoken_trained import new_items
    with mp.Pool(8) as pool:
        results = pool.map(clean_text0, texts)
        print(len(results))
        df = df.with_columns(pl.Series('text2', results))
        df.write_csv('./bv_news_2.csv')
        with open('./sometext.txt', 'r', encoding='utf8') as f:
            t = list(f.readlines())
            t = pool.map(clean_text0, t)
            with open('./text.txt', 'w+', encoding='utf8') as ff:
                txt = "\n".join(map(lambda x: x or "", results))
                txt += "\n".join(map(lambda x: x or "", new_items))
                txt += "\n".join(map(lambda x: x or "", t))
                ff.write(txt)


def process4():
    with open('./sometext.txt', 'r', encoding='utf8') as f:
        texts = list(f.readlines())
        import multiprocessing as mp
        with mp.Pool(8) as pool:
            results = pool.map(clean_text0, texts)
            print(len(results))
            items = dict()
            for i in results:
                k = (i or "").strip()
                if re.match(r"\d+", k):
                    continue
                items[k] = len(k)
            items = {k: v for k, v in sorted(items.items(), key=lambda item: item[1])}
            results = items.keys()
            print(len(results))
            text = " \n".join(results)

            with open('./sometext.txt', 'w', encoding='utf8') as ff:
                ff.write(text)


def process5():
    with open('./text.txt', 'r', encoding='utf8') as f:
        texts = list(f.readlines())
        for i in range(200, 210):
            print(clean_text0(texts[i]))
        import multiprocessing as mp
        with mp.Pool(8) as pool:
            results = pool.map(clean_text0, texts)
            results = list(filter(lambda x: x is not None, results))
            text = "\n".join(results)
            with open('./text2.txt', 'w', encoding='utf8') as ff:
                ff.write(text)


def process6():
    with open('./text2.txt', 'r', encoding='utf8') as f:
        texts = list(f.readlines())
        for i in range(200, 210):
            print(clean_text0(texts[i]))
        import multiprocessing as mp
        with mp.Pool(8) as pool:
            results = pool.map(clean_text0, texts)
            results = list(filter(lambda x: x is not None, results))
            text = "\n".join(results)
            with open('./text3.txt', 'w', encoding='utf8') as ff:
                ff.write(text)

if __name__ == '__main__':
    # fetch_all_data()

    # process_news()
    # process_news2()
    # process_news3()
    # process4()
    # process5()
    process6()

    # s = "شرکت تام ایران خودرو سهامی عام در گزارش 12ماهه به سود 1314 میلیارد ریال رسیده است"
    # print(s)
    # print(clean_text2(s))
    # extract_vocab()

    t = ""
    print(clean_text0(t))
