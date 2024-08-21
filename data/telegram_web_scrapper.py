import dataclasses
import json
import multiprocessing as mp
import os
from functools import partial
from multiprocessing import Process
from typing import List

import bs4
import jsonlines
import polars as pl
import requests
from bs4 import BeautifulSoup

from data.bv_news_crawler import clean_text0


@dataclasses.dataclass
class ChannelMeta:
    latest_id: int | None
    channel_name: str

    def dict(self):
        return {k: str(v) for k, v in dataclasses.asdict(self).items()}


@dataclasses.dataclass
class CrawledItem:
    id: int | str
    text: str
    date: str
    channel_name: str

    def dict(self):
        return {k: str(v) for k, v in dataclasses.asdict(self).items()}


def extract_row(channel_name, content) -> List[CrawledItem]:
    soup = BeautifulSoup(content, "html.parser")
    result: List[CrawledItem] = []
    rows = soup.findAll('div', {"class": "tgme_widget_message_wrap"})
    for row in rows:
        if type(row) == bs4.Tag:
            try:
                _id = row.find('div', attrs={'data-post': True})
                if _id is not None:
                    _id = _id.attrs["data-post"].replace(f"{channel_name}/", "")
                _date = row.find('time', attrs={'datetime': True}).attrs["datetime"]
                _text = row.find('div', {"class": "tgme_widget_message_text"})
                if type(_text) == bs4.Tag:
                    _text = _text.get_text().strip()
                    result.append(CrawledItem(id=_id, text=_text, date=_date, channel_name=channel_name))
            except Exception as e:
                print("extract_row error ", e, row)
    return result


def fetch(channel_name, params):
    cookies = {
        # 'stel_dt': '-210',
        # 'stel_ssid': '1664e831fce0d4d226_16981272495983999534',
    }

    headers = {
        'accept': 'application/json, text/javascript, */*; q=0.01',
        'accept-language': 'fa-IR,fa;q=0.9,en-GB;q=0.8,en;q=0.7,en-US;q=0.6',
        # 'content-length': '0',
        # 'cookie': 'stel_dt=-210; stel_ssid=1664e831fce0d4d226_16981272495983999534',
        'origin': 'https://t.me',
        'priority': 'u=1, i',
        'referer': f"https://t.me/s/{channel_name}",
        'sec-ch-ua': '"Not)A;Brand";v="99", "Google Chrome";v="127", "Chromium";v="127"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36',
        'x-requested-with': 'XMLHttpRequest',
    }

    return requests.post(f"https://t.me/s/{channel_name}", params=params, headers=headers, cookies=cookies)


def get_results(q: mp.Queue, row: ChannelMeta):
    params = {
        'after': row.latest_id,
    }
    while True:
        try:

            res = fetch(row.channel_name, params)

            if res.ok:
                text = ((res.text or "")[1:-1] or "").replace("\\n", " ")
                text = text.replace("\\", "")
                items = extract_row(row.channel_name, text)
                print("get_results ... ", row, "   items:", len(items))
                if len(items) > 0:
                    for item in items:
                        q.put(item)
                else:
                    print("len(items) > 0  ", row.channel_name)

                break
        except Exception as e:
            print(f"error in channel_name:{row.channel_name} latest_id:{row.latest_id} error:{e}")


def file_writer(q: mp.Queue):
    cache = dict()
    '''listens for messages on the q, writes to file. '''
    with open("./telegram/news.jsonl", 'a+', encoding='utf8') as f:
        while True:
            try:
                msg = q.get()
                if msg == None:
                    print("file_writer msg == None")
                    continue
                if msg == 'kill':
                    print("breaking msg == 'kill'")
                    break
                if type(msg) == CrawledItem:
                    item: CrawledItem = msg
                    key = f"{item.channel_name}_{item.id}"
                    if key not in cache:
                        f.write(json.dumps(item.dict(), ensure_ascii=False) + "\n")
                        f.flush()
                        cache[key] = None
                    else:
                        print("file_writer error  row is in cache ", key)
            except Exception as e:
                print(f"file_writer e:{e}")
        f.flush()
        f.close()
        print("end of file_writer")


def process1():

    channels = [
        'donyaye_eghtesad_com',
        # 'tejaratnews',
        # 'eghtesadonline'
    ]

    all_reqs: List[ChannelMeta] = []

    for channel_name in channels:
        ids = list([i for i in range(0, 2000, 20)])
        all_reqs = all_reqs + [ChannelMeta(channel_name=channel_name, latest_id=i) for i in ids]

    count = os.cpu_count()
    manager = mp.Manager()
    q = manager.Queue()

    print("cpu count", count)
    print("total crawls ", len(all_reqs))

    with mp.Pool(count - 1) as pool:
        writer = Process(target=file_writer, args=(q,))
        writer.start()
        func = partial(get_results, q)
        pool.map(func, all_reqs)
        pool.close()
        pool.join()
        q.put('kill')
        writer.join()
        print("end...")


def telegram_news_cleaner(obj):
    _text = obj["text"] or ""
    _text = clean_text0(_text)
    if _text is None:
        return None
    result = dict()
    result["id"] = obj['id']
    result["date"] = obj['time']
    result["text"] = _text
    result["channel_name"] = None
    result["label"] = None
    result["gpt_mofid"] = None
    return result


def process2():
    with jsonlines.open('./telegram/news.jsonl') as reader:
        rows = [obj for obj in reader]

        # rows = []
        # for obj in reader:
        #     rows.append(obj)
        #     if len(rows) > 100:
        #         break

        with mp.Pool(os.cpu_count() - 1) as pool:
            results = pool.map(telegram_news_cleaner, rows)
            results = list(filter(lambda x: x is not None, results))
            # print("results", len(results))
            df = pl.from_records(results)
            df.write_csv('./telegram_news_by_label_p1.csv')


if __name__ == "__main__":
    process1()
    # process2()
