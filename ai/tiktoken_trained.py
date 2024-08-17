VOCAB_SIZE = 3499


# VOCAB_SIZE = 8191


def fn():
    import polars as pl
    from data.bv_news_crawler import clean_text0

    df = pl.read_csv('./stocks.csv')

    stocks = dict()
    for r in df.rows(named=True):
        stocks[r["Ticker"]] = None

    # print(list(stocks.keys()))

    def x2(x):
        return f"{clean_text0(x[0])}"

    # print(1)
    new_col = df.select(pl.col('Ticker').alias("Name2")).apply(x2)
    df = df.with_columns(new_col)

    df.write_csv('./stocks2.csv')


def count_all_uniq_words(text):
    """
    about 81082 for bv_news.csv
    """
    import json
    import re
    d = dict()
    arr = text.split(" ")
    for i in arr:
        if re.match(r'\d+', i):
            continue
        d[i] = None

    print(len(d.keys()))
    print("saving to tiktoken_trained.iso")
    with open('./f.json', 'w+', encoding='utf8') as f:
        arr = list(map(lambda x: str(x), d.keys()))
        f.write(json.dumps(arr))
    exit()


new_items = [
    'آباد', 'آبادا', 'آپ', 'آسیا', 'اپال', 'اتکام', 'اخابر', 'اردستان', 'اسیاتک', 'افق', 'البرز',
    'امید', 'امین', 'انرژی', 'بالبر', 'بترانس', 'برکت', 'بسویچ', 'بشهاب', 'بفجر', 'بکاب', 'بکام',
    'بموتو',
    'بنیرو', 'بورس', 'بوعلی', 'پارس', 'پارسان', 'پارسیان', 'پاسا', 'پاکشو', 'پتایر', 'پترول', 'پدرخش',
    'پردیس',
    'پسهند', 'پکرمان', 'پکویر', 'پلاسک', 'پی پاد', 'تاپیکو', 'تایرا', 'تپمپی', 'تکاردان', 'تکمبا',
    'تکنو', 'تملت', 'تنوین', 'تیپیکو', 'ثاخت', 'ثامان', 'ثامید', 'ثبهساز', 'ثشاهد', 'ثشرق', 'ثفارس',
    'ثمسکن',
    'ثنوسا', 'جم', 'جم پیلن', 'چافست', 'چدن', 'چفیبر', 'چکاپا', 'چکارن', 'چکاوه', 'حپترو', 'حتاید',
    'حتوکا', 'حفارس', 'حفاری', 'حکشتی', 'خاذین', 'خاهن', 'خبهمن', 'خپویش', 'ختراک', 'ختور', 'ختوقا',
    'خچرخش',
    'خراسان', 'خریخت', 'خرینگ', 'خزامیا', 'خزر', 'خساپا', 'خشرق', 'خفنر', 'خکار', 'خکمک', 'خگستر',
    'خلنت', 'خمحرکه', 'خمحور', 'خمهر', 'خنصیر', 'خودرو', 'خوساز', 'دابور', 'داتام', 'دارو', 'داسوه',
    'دالبر',
    'دامین', 'دانا', 'دپارس', 'دتماد', 'دجابر', 'ددام', 'درازک', 'دروز', 'دزهراوی', 'دسبحا', 'دسبحان',
    'دسینا', 'دشیمی', 'دعبید', 'دفارا', 'دفرا', 'دکوثر', 'دکیمی', 'دلر', 'دلقما', 'دیران', 'ذوب',
    'رانفور', 'رتاپ', 'رکیش', 'رمپنا', 'زپارس', 'زکوثر', 'زمگسا', 'سآبیک', 'ساراب', 'ساربیل', 'ساروم',
    'سبجنو',
    'سبهان', 'سپ', 'سپاها', 'سپید', 'ستران', 'سخاش', 'سخزر', 'سخوز', 'سدشت', 'سدور', 'سرود', 'سشرق',
    'سشمال', 'سصفها', 'سصوفی', 'سغرب', 'سفار', 'سفارس', 'سفانو', 'سقاین', 'سکرد', 'سکرما', 'سمازن',
    'سنیر', 'سهرمز', 'سهگمت', 'سیتا', 'سیدکو', 'سیستم', 'سیلام', 'سیمرغ', 'شاراک', 'شاملا', 'شبریز',
    'شبندر',
    'شبهرن', 'شپارس', 'شپاکسا', 'شپدیس', 'شپنا', 'شتران', 'شخارک', 'شدوص', 'شراز', 'شسپا', 'شستا',
    'شسینا', 'شغدیر', 'شفا', 'شفارس', 'شفن', 'شکربن', 'شکلر', 'شگل', 'شلعاب', 'شنفت', 'شهر', 'شوینده',
    'شیراز',
    'شیران', 'صبا', 'غاذر', 'غالبر', 'غبشهر', 'غبهنوش', 'غپاک', 'غپینو', 'غچین', 'غدام', 'غدشت', 'غزر',
    'غسالم', 'غشاذر', 'غشان', 'غشصفا', 'غشهد', 'غکورش', 'غگرجی', 'غگل', 'غمهرا', 'غنوش', 'فاذر',
    'فاراک', 'فارس', 'فاسمین', 'فاما', 'فایرا', 'فباهنر', 'فپنتا', 'فجام', 'فجر', 'فخاس', 'فخوز',
    'فرآور',
    'فروس', 'فسازان', 'فسبزوار', 'فسپا', 'فسرب', 'فلامی', 'فلوله', 'فمراد', 'فملی', 'فنوال', 'فنورد',
    'فولاد',
    'فولاژ', 'قپیرا', 'قثابت', 'قرن', 'قزوین', 'قشکر', 'قشهد', 'قصفها', 'قلرست', 'قمرو', 'قنیشا',
    'قهکمت', 'کاذر', 'کالا', 'کاما', 'کاوه', 'کبافق', 'کپارس', 'کپشیر', 'کترام', 'کچاد', 'کحافظ',
    'کخاک', 'کدما',
    'کرازی', 'کرماشا', 'کروی', 'کساپا', 'کساوه', 'کسرا', 'کسرام', 'کسعدی', 'کطبس', 'کفپارس', 'کفرا',
    'کگاز', 'کگل', 'کلوند', 'کماسه', 'کمنگنز', 'کنور', 'کهمدا', 'کویر', 'کیمیا', 'کیمیاتک', 'لابسا',
    'لبوتان',
    'لپارس', 'لخزر', 'لسرما', 'لوتوس', 'ما', 'مبین', 'مداران', 'ملت', 'میدکو', 'نمرینو', 'نوری',
    'های وب', 'همراه', 'وآذر', 'وآفری', 'واتی', 'واعتبار', 'والبر', 'وامید', 'وایران', 'وبانک', 'وبشهر',
    'وبصادر',
    'وبملت', 'وبهمن', 'وبوعلی', 'وبیمه', 'وپارس', 'وپاسار', 'وپترو', 'وپخش', 'وپست', 'وتجارت', 'وتوس',
    'وتوسم', 'وتوشه', 'وتوصا', 'وتوکا', 'وخارزم', 'وخاور', 'ورنا', 'وساپا', 'وساخت', 'وساربیل',
    'وساشرقی', 'وساغربی', 'وسبوشهر', 'وسپه', 'وسخراج', 'وسخراش', 'وسخوز', 'وسرضوی', 'وسزنجان', 'وسصفا',
    'وسفارس',
    'وسقم', 'وسکاب', 'وسکرد', 'وسکرشا', 'وسکرمان', 'وسکهبو', 'وسگلستا', 'وسگیلا', 'وسلرستا', 'وسمازن',
    'وسمرکز', 'وسهرمز', 'وسهمدا', 'وسیزد', 'وسیستا', 'وسیلام', 'وسینا', 'وصنا', 'وصندوق', 'وصنعت',
    'وغدیر', 'وکار', 'وکغدیر', 'ولپارس', 'ولساپا', 'ولصنم', 'ولغدر', 'ولکار', 'ولملت', 'ومدیر', 'ومعادن',
    'وملی',
    'ومهان', 'ونفت', 'ونوین', 'ونیرو', 'ونیکی', 'آ س پ', 'آردینه', 'آریا', 'آریان', 'اپرداز', 'اتکای',
    'ارفع', 'اعتلا', 'افرا', 'انتخاب', 'اوان', 'بالاس', 'بپاس', 'بپیوند', 'بجهرم', 'بخاور', 'بزاگرس',
    'بساما', 'بکابل', 'بکهنوج', 'بگیلان', 'بمپنا', 'بمولد', 'بنو', 'بهپاک', 'بیوتیک', 'پارتا', 'پخش',
    'پیزد', 'تاپکیش', 'تبرک', 'تپسی', 'تجلی', 'تلیسه', 'تماوند', 'توریل', 'توسن', 'ثالوند', 'ثباغ',
    'ثپردیس', 'ثتران', 'ثجنوب', 'ثرود', 'ثعمرا', 'ثغرب', 'چخزر', 'حآسا', 'حآفرین', 'حپارسا', 'حپرتو',
    'حخزر', 'حریل', 'حسیر', 'حسینا', 'حگهر', 'خاور', 'خدیزل', 'خکرمان', 'خنور', 'داوه', 'دبالک',
    'دتوزیع', 'دتولید', 'ددانا', 'درازی', 'درهآور', 'دسانکو', 'دقاضی', 'دکپسول', 'دماوند', 'رافزا',
    'رنیک',
    'ریشمک', 'زاگرس', 'زبینا', 'زدشت', 'زشریف', 'زشگزا', 'زفجر', 'زفکا', 'زقیام', 'زکشت', 'زگلدشت',
    'زماهان',
    'زملارد', 'زنگان', 'ساروج', 'سامان', 'ساوه', 'ساینا', 'سبزوا', 'سپیدار', 'سجام', 'سدبیر', 'سرچشمه',
    'سغدیر', 'سمگا', 'شاروم', 'شاوان', 'شبصیر', 'شپاس', 'شتوکا', 'شجم', 'شرانل', 'شصدف', 'شفام', 'شکام',
    'شگویا', 'شملی', 'عالیس', 'غپآذر', 'غپونه', 'غدانه', 'غدیس', 'غشهداب', 'غصینو', 'غفارس', 'غگلپا',
    'غگلستا', 'غگیلا', 'غمایه', 'غمینو', 'غویتا', 'فالوم', 'فتوسا', 'فجهان', 'فرابورس', 'فرود', 'فروژ',
    'فروسیل', 'فروی', 'فزر', 'فزرین', 'فسوژ', 'فصبا', 'فغدیر', 'فگستر', 'فن افزار', 'فنر', 'فولای',
    'قاسم', 'قچار', 'قشیر', 'کاسپین', 'کایزد', 'کپرور', 'کتوسعه', 'کتوکا', 'کرمان', 'کرومیت', 'کزغال',
    'کشرق',
    'کگهر', 'کلر', 'کمرجان', 'کوثر', 'کی بی سی', 'گدنا', 'گکوثر', 'گلدیرا', 'گوهران', 'لطیف', 'مادیرا',
    'مارون', 'مدیریت', 'معین', 'مفاخر', 'میهن', 'ناما', 'نخریس', 'نطرین', 'نوین', 'نیان', 'هجرت',
    'هرمز', 'وآوا', 'والماس', 'وامین', 'وپویا', 'وتعاون', 'ودی', 'وسبحان', 'وسپهر', 'وطوبی', 'وکبهمن',
    'وگردش',
    'وگستر', 'ولبهمن', 'ولتجار', 'ولشرق', 'ومعلم', 'وملل', 'وهامون', 'وهور', 'شتهران', 'کارام', 'کازرو',
    'ممسنی', 'شلرد', 'وشمال', 'زنجان', 'غمارگ', 'تکیمیا', 'وبرق', 'استقلال', 'پرسپولیس', 'نیرو', 'سنوین',
    'آینده', 'تفارس', 'ودانا', 'بمیلا', 'تکنار', 'داراب', 'فسا', 'جهرم', 'حرهشا', 'بپردیس', 'سخواف',
    'آرمان', 'حاریا', 'فلات', 'وکادو', 'غشوکو', 'سکارون', 'شمواد', 'حبندر', 'شپلی', 'ولراز', 'ورازی',
    'ثعتما', 'ویسا', 'ثنظام', 'وزمین', 'دی', 'دحاوی', 'ثجوان', 'کمینا', 'کباده', 'وآفر', 'واحصا',
    'واحیا', 'سفاسی', 'وملت', 'خکاوه', 'تشتاد', 'وفتخار', 'گکیش', 'جوین', 'فبیرا', 'لکما', 'وثوق',
    'کابگن',
    'لازما', 'پلاست', 'قجام', 'بایکا', 'غیوان', 'لپیام', 'غناب', 'وامیر', 'قنقش', 'سمایه', 'خصدرا',
    'شفارا',
    'وثنو', 'کهرام', 'وسالت', 'گنگین', 'فنفت', 'قیستو', 'شلیا', 'قشرین', 'وهنر', 'شزنگ', 'کایتا',
    'لخانه',
    'ثاژن', 'وشهر', 'خبازرس', 'ومشان', 'ثنور', 'ثقزوی', 'سپرمی', 'ساذری', 'سایرا', 'چنوپا', 'ولانا',
    'سباقر',
    'سمتاز', 'تپکو', 'فجوش', 'دتهران', 'وآیند', 'وسرمد', 'وسنا', 'باران', 'وارس', 'رتکو', 'وآتوس',
    'نبروج', 'ولیز', 'تمحرکه', 'فبستم', 'وحافظ', 'وثخوز', 'ولقمان', 'قتربت', 'تاتمس', 'خبنیان',
    'بازرگام',
    'سفارود', 'فکمند', 'نتوس', 'وسدید', 'شتولی', 'ثزاگرس', 'خلیبل', 'وسین', 'وتوسکا', 'فافزا', 'خودکفا',
    'شکف',
    'کفرآور', 'فافق', 'حشکوه', 'بهیر', 'خفولا', 'حگردش', 'تپولا', 'قاروم', 'گپارس', 'وایرا', 'کاریز',
    'ثتوسا', 'فاهواز', 'وجامی', 'آبین', 'شکبیر', 'دهدشت', 'معیار', 'وحکمت', 'خعمرا', 'دشیری', 'کیسون',
    'وپسا', 'بتک', 'شستان', 'شساخت', 'خفناور', 'سلار', 'فسدید', 'پشاهن', 'کقزوی', 'اتکاسا', 'غبهار',
    'وآرین', 'ثنام', 'شپترو', 'ثاصفا', 'شسم', 'پلوله', 'کورز', 'گشان', 'نشار', 'تفیرو', 'رفاه', 'تابا',
    'فوکا', 'خپارس', 'بتهران', 'فنرژی', 'وفردا', 'پرداخت', 'تاصیکو', 'خموتور', 'تکشا',

    'ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض',
    'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ل', 'م', 'ن', 'ه', 'و', 'پ', 'چ', 'ک', 'گ', 'ی',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
    'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', "$", "%", "-", "+",

    "january", "february", "march", "april", "may", "juan", "july",
    "july", "august", "september", "october", "november", "december",

    "فروردین", "اردیبهشت", "خرداد", "تیر", "مرداد", "شهریور",
    "مهر", "آبان", "آذر", "دی", "بهمن", "اسفند"
]


def train_tokenizer(csv_path):
    import polars as pl

    df = pl.read_csv(csv_path)
    special_tokens = []
    # special_tokens = ["positive", "neutral", "negative"]

    text = " ".join(df["text"].to_list())
    text += " " + " ".join(new_items)

    print("new_items  ", len(new_items))

    import tiktoken._educational as ed

    pat_str = "|".join(
        [
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""\p{N}{1,3}""",
            r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
            r"""\s*[\r\n]+""",
            r"""\s+(?!\S)""",
            r"""\s+""",
        ]
    )

    enc = ed.SimpleBytePairEncoding(pat_str=pat_str, mergeable_ranks=dict())

    trained_vocab_size = VOCAB_SIZE

    enc = enc.train(training_data=text, vocab_size=trained_vocab_size, pat_str=pat_str)

    print(len(enc.mergeable_ranks.keys()))
    print(list(enc.mergeable_ranks.keys()))
    import pickle
    print("saving to tiktoken_trained.iso")
    file = open('./tiktoken_trained.iso', 'wb')
    pickle.dump(enc.mergeable_ranks, file)
    file.close()


def train_tokenizer2(text, save_to):
    import tiktoken._educational as ed
    import pickle
    import os
    pat_str = "|".join(
        [
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""\p{N}{1,3}""",
            r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
            r"""\s*[\r\n]+""",
            r"""\s+(?!\S)""",
            r"""\s+""",
        ]
    )

    enc = ed.SimpleBytePairEncoding(pat_str=pat_str, mergeable_ranks=dict())

    trained_vocab_size = VOCAB_SIZE

    enc = enc.train(training_data=text, vocab_size=trained_vocab_size, pat_str=pat_str)

    print(len(enc.mergeable_ranks.keys()))
    print(list(enc.mergeable_ranks.keys()))
    print("saving to tiktoken_trained.iso")
    file = open(os.path.join(save_to, "tiktoken_trained.iso"), 'wb')
    pickle.dump(enc.mergeable_ranks, file)
    file.close()
    print("...saved")


def get_tokenizer(show_log=True):
    if show_log:
        print("... loading from tiktoken_trained.iso")
    import pickle
    import sys
    from typing import (
        AbstractSet,
        cast,
        Collection,
        Dict,
        Iterator,
        List,
        Literal,
        Sequence,
        Union,
    )
    import tiktoken

    is_colab = 'google.colab' in sys.modules
    p = "./my_sentiment" if is_colab else "."
    p = f"{p}/tiktoken_trained.iso"
    f = open(p, 'rb')
    mergeable_ranks = pickle.load(f)
    f.close()
    if show_log:
        print("...loaded tiktoken_trained.iso")

    pat_str = "|".join(
        [
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""\p{N}{1,3}""",
            r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
            r"""\s*[\r\n]+""",
            r"""\s+(?!\S)""",
            r"""\s+""",
        ]
    )

    class Tokenizer:
        """
        Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
        """

        special_tokens: Dict[str, int]

        num_reserved_special_tokens = 256

        def __init__(self, pat_str, mergeable_ranks):

            self.pat_str = pat_str
            self.mergeable_ranks = mergeable_ranks

            self.num_base_tokens = len(mergeable_ranks)
            special_tokens = [
                "<|begin_of_text|>",
                "<|end_of_text|>",
                "<|padding_token|>",
                "<|reserved_special_token_0|>",
                "<|reserved_special_token_1|>",
                "<|reserved_special_token_2|>",
                "<|reserved_special_token_3|>",
                "<|reserved_special_token_4|>",
                "<|start_header_id|>",
                "<|end_header_id|>",
                "<|eot_id|>",  # end of turn
            ]

            self.special_tokens = {
                token: self.num_base_tokens + i for i, token in enumerate(special_tokens)
            }
            self.model = tiktoken.Encoding(
                name="tiktoken_trained",
                pat_str=self.pat_str,
                mergeable_ranks=mergeable_ranks,
                special_tokens=self.special_tokens,
            )

            self.n_words: int = self.model.n_vocab
            # BOS / EOS token IDs
            self.bos_id: int = self.special_tokens["<|begin_of_text|>"]
            self.eos_id: int = self.special_tokens["<|end_of_text|>"]
            self.pad_id: int = self.special_tokens["<|padding_token|>"]
            # self.pad_id: int = -1
            self.stop_tokens = {
                self.special_tokens["<|end_of_text|>"],
                self.special_tokens["<|eot_id|>"],
            }
            if show_log:
                print(
                    f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
                )

        def encode(
                self,
                s: str,
                *,
                bos: bool,
                eos: bool,
                allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
                disallowed_special: Union[Literal["all"], Collection[str]] = (),
        ) -> List[int]:
            """
            Encodes a string into a list of token IDs.

            Args:
                s (str): The input string to be encoded.
                bos (bool): Whether to prepend the beginning-of-sequence token.
                eos (bool): Whether to append the end-of-sequence token.
                allowed_tokens ("all"|set[str]): allowed special tokens in string
                disallowed_tokens ("all"|set[str]): special tokens that raise an error when in string

            Returns:
                list[int]: A list of token IDs.

            By default, setting disallowed_special=() encodes a string by ignoring
            special tokens. Specifically:
            - Setting `disallowed_special` to () will cause all text corresponding
              to special tokens to be encoded as natural text (insteading of raising
              an error).
            - Setting `allowed_special` to "all" will treat all text corresponding
              to special tokens to be encoded as special tokens.
            """
            assert type(s) is str

            # The tiktoken tokenizer can handle <=400k chars without
            # pyo3_runtime.PanicException.
            TIKTOKEN_MAX_ENCODE_CHARS = 400_000

            # https://github.com/openai/tiktoken/issues/195
            # Here we iterate over subsequences and split if we exceed the limit
            # of max consecutive non-whitespace or whitespace characters.
            MAX_NO_WHITESPACES_CHARS = 25_000

            substrs = (
                substr
                for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
                for substr in self._split_whitespaces_or_nonwhitespaces(
                s[i: i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
            )
            t: List[int] = []
            for substr in substrs:
                t.extend(
                    self.model.encode(
                        substr,
                        allowed_special=allowed_special,
                        disallowed_special=disallowed_special,
                    )
                )
            if bos:
                t.insert(0, self.bos_id)
            if eos:
                t.append(self.eos_id)
            return t

        def decode(self, t: Sequence[int]) -> str:
            """
            Decodes a list of token IDs into a string.

            Args:
                t (List[int]): The list of token IDs to be decoded.

            Returns:
                str: The decoded string.
            """
            # Typecast is safe here. Tiktoken doesn't do anything list-related with the sequence.
            return self.model.decode(cast(List[int], t))

        @staticmethod
        def _split_whitespaces_or_nonwhitespaces(
                s: str, max_consecutive_slice_len: int
        ) -> Iterator[str]:
            """
            Splits the string `s` so that each substring contains no more than `max_consecutive_slice_len`
            consecutive whitespaces or consecutive non-whitespaces.
            """
            current_slice_len = 0
            current_slice_is_space = s[0].isspace() if len(s) > 0 else False
            slice_start = 0

            for i in range(len(s)):
                is_now_space = s[i].isspace()

                if current_slice_is_space ^ is_now_space:
                    current_slice_len = 1
                    current_slice_is_space = is_now_space
                else:
                    current_slice_len += 1
                    if current_slice_len > max_consecutive_slice_len:
                        yield s[slice_start:i]
                        slice_start = i
                        current_slice_len = 1
            yield s[slice_start:]

    return Tokenizer(pat_str, mergeable_ranks)


if __name__ == '__main__':
    print('....')
    # train_tokenizer('./bv_news.csv')
    # fn()
    # fn2()
    # fn3()
    get_tokenizer()
