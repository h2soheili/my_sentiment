import json
import os
import time

import requests

from data.bv_news_crawler import clean_text0


def get_result(text):
    import replicate
    os.environ['REPLICATE_API_TOKEN'] = "r8_eNKLF27FBXIOkjvrkNMWtU2aLM7L5YI0OlLBB"
    text = text.replace('\n', "")
    text = text.strip()
    # The meta/meta-llama-3-70b-instruct model can stream output as it's running.
    rep_stream = replicate.stream(
        "meta/meta-llama-3-70b-instruct",
        input={
            "top_k": 50,
            "top_p": 0.9,
            "prompt": text,
            "max_tokens": 512,
            "min_tokens": 0,
            "temperature": 0.6,
            "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            "presence_penalty": 1.15,
            "frequency_penalty": 0.2
        },
    )
    for event in rep_stream:
        print(str(event), end="")


def get_result_from_jabirproject(_text):
    text = "برای سهامدار بازار بورس این خبر به کدام حالت از حالت های { مثبت  خنثی منفی } اشاره دارد ؟"
    # text += "\n فقط حالت را بنویس و توضیح ننویس \n"
    text += "\n اول حالت رو بنویس و بعد خلاصه دلیل را بنویس \n"
    text += _text
    # print(text)
    url = "https://api.jabirproject.org/generate"
    headers = {
        "apikey": "e0f55e19-6cfb-43f2-a2ed-2ffde84df0f8",
        "content-type": "application/json"
    }
    data = {
        "messages": [
            {
                "role": "user",
                "content": text
            }
        ]
    }

    result = None

    for i in range(100):
        res = None
        try:
            res = requests.post(url, headers=headers, data=json.dumps(data))
            if res.ok:
                res = res.json()
                print(res)
                result = res.get("result", dict()).get("content", None)
                break
        except Exception as e:
            print(" error for ", e, res, _text)
            time.sleep(0.2)

    return result


def fn(row):
    _id = row["id"]
    try:
        text = row["text"]
        if text is None or text == "":
            print("skip ", _id)
            return None
        text = clean_text0(text)
        if text is None or text == "":
            print("skip ", _id)
            return None
        row['jabirproject'] = get_result_from_jabirproject(text)
        return row
    except Exception as e:
        print("error ", e, row)
        return None


def get_result_from_gpt_mofid(_text):
    """""
    ***********************************
    update Authorization an Cookie

    """
    text = "برای سهامدار بازار بورس این خبر به کدام حالت از حالت های { مثبت  خنثی منفی } اشاره دارد ؟"
    # text += "\n فقط حالت را بنویس و توضیح ننویس \n"
    text += "\n فقط حالت رو بنویس \n"
    text += _text
    url = "https://gpt.emofid.com/api/chat/completions"
    headers = {
        'Accept': '*/*',
        'Content-Type': 'en-US,en;q=0.9,fa;q=0.8',
        'Authorization': 'Bearer ',
        'Connection': 'keep-alive',
        'Content-Type': 'application/json',
        'Cookie': '',
        'Origin': 'https://gpt.emofid.com',
        'Referer': 'https://gpt.emofid.com/',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36',
        'sec-ch-ua': '"Not)A;Brand";v="99", "Google Chrome";v="127", "Chromium";v="127"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"'
    }

    data = {
        "model": "gpt-4o",
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": text
            }
        ],
        "chat_id": "2f84d6c1-7160-481f-ab0c-f720dd514e33"
    }

    result = None

    for i in range(100):
        res = None
        try:
            res = requests.post(url, headers=headers, data=json.dumps(data))
            if res.ok:
                res = res.json()
                result = res.get("choices", [])[0].get("message", dict()).get("content")
                break
        except Exception as e:
            print(" error for ", e, res, _text)
            time.sleep(0.2)

    return result


def fn2(row):
    _id = row["id"]
    try:
        text = row["text"]
        if text is None or text == "":
            print("skip ", _id)
            return None
        text = clean_text0(text)
        if text is None or text == "":
            print("skip ", _id)
            return None
        row['gpt_mofid'] = get_result_from_gpt_mofid(text)
        return row
    except Exception as e:
        print("error ", e, row)
        return None


if __name__ == "__main__":
    t = """
    نرخ سود همانطور که مشاهده می شود با کاهش 0.66 % در نرخ بهره بازار بین بانکی شاهد کاهش 0.95 % در نرخ قرارداد ریپو و کاهش 2.5 % در میانگین نرخ بازده اوراق با درآمد ثابت هستیم. بررسی منحنی بازده استخراج شده از نماد های اخزا نشان می دهد که میانگین نرخ های بازده در کلیه سررسیدها کاهشی بودند و به 22.59 % رسیده اند.
    """

    # get_result(t)

    # get_result_from_jabirproject(t)

    import polars as pl

    df = pl.read_csv('./bv_news.csv')
    print("df", len(df))

    # df = df.drop(["text", "sentiment", "count"], )
    # df = df.with_columns(pl.col("text2").alias("content"))
    # df = df.drop(["text2",], )
    # df.write_csv('./bv_news_2.csv')
    # exit()

    import multiprocessing as mp

    with mp.Pool(10) as pool:
        results = pool.map(fn2, df.rows(named=True)[:100])
        print("results ", len(results))
        results = list(filter(lambda x: x is not None, results))
        print("results ", len(results))
        df2 = pl.from_records(results)
        df2.write_csv('./result2.csv')
