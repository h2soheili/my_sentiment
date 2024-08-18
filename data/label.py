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
        text = row["content"]
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


if __name__ == "__main__":
    t = """
    نرخ سود همانطور که مشاهده می شود با کاهش 0.66 % در نرخ بهره بازار بین بانکی شاهد کاهش 0.95 % در نرخ قرارداد ریپو و کاهش 2.5 % در میانگین نرخ بازده اوراق با درآمد ثابت هستیم. بررسی منحنی بازده استخراج شده از نماد های اخزا نشان می دهد که میانگین نرخ های بازده در کلیه سررسیدها کاهشی بودند و به 22.59 % رسیده اند.
    """

    # get_result(t)

    # get_result_from_jabirproject(t)

    import polars as pl

    df = pl.read_csv('./bv_news_2.csv')
    print("df", len(df))

    # df = df.drop(["text", "sentiment", "count"], )
    # df = df.with_columns(pl.col("text2").alias("content"))
    # df = df.drop(["text2",], )
    # df.write_csv('./bv_news_2.csv')
    # exit()

    import multiprocessing as mp

    with mp.Pool(10) as pool:
        results = pool.map(fn, df.rows(named=True)[:100])
        print("results ", len(results))
        results = list(filter(lambda x: x is not None, results))
        print("results ", len(results))
        df2 = pl.from_records(results)
        df2.write_csv('./result2.csv')
