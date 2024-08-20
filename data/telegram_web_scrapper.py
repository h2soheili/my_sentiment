import json
import multiprocessing as mp
import os
from functools import partial

import bs4
import requests
from bs4 import BeautifulSoup


def extract_row(channel_name, content):
    soup = BeautifulSoup(content, "html.parser")
    result = []
    rows = soup.findAll('div', {"class": "tgme_widget_message_wrap"})
    for row in rows:
        if type(row) == bs4.Tag:
            try:
                _id = row.find('div', attrs={'data-post': True})
                if _id is not None:
                    _id = _id.attrs["data-post"].replace(f"{channel_name}/", "")
                _time = row.find('time', attrs={'datetime': True}).attrs["datetime"]
                _content = row.find('div', {"class": "tgme_widget_message_text"})
                _content = _content.get_text().strip()
                result.append(dict(id=_id, content=_content, time=_time))
            except Exception as e:
                print(e)
    return result


def get_client(channel_name, params):

    cookies = {
        'stel_dt': '-210',
        'stel_ssid': '1664e831fce0d4d226_16981272495983999534',
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


def get_latest_page(channel_name):
    latest_id = None
    retry = 0
    params = {
        'before': 99999999999999,
    }

    while True:
        try:
            res = get_client(channel_name, params)
            if res.ok:
                text = ((res.text or "")[1:-1] or "").replace("\\n", " ")
                text = text.replace("\\", "")
                items = extract_row(channel_name, text)
                if len(items) > 0:
                    latest_id = int(items[-1]["id"])
                break
        except Exception as e:
            print(f"get_latest_page retry:{retry} channel_name:{channel_name} e:{e}")
            retry += 1

    return dict(channel_name=channel_name, latest_id=latest_id)


def get_results(q, row):
    channel_name = row["channel_name"]
    _id = row["id"]
    params = {
        'before': _id,
    }
    while True:
        try:
            res = get_client(channel_name, params)
            if res.ok:
                text = ((res.text or "")[1:-1] or "").replace("\\n", " ")
                text = text.replace("\\", "")
                items = extract_row(channel_name, text)
                if len(items) > 0:
                    q.put(items)
                else:
                    print("len(items) > 0  ", channel_name)
                    break
        except Exception as e:
            print("error in ", channel_name, "    ", e)


def file_writer(q):
    '''listens for messages on the q, writes to file. '''
    with open("./telegram/news.jsonl", 'a+', encoding='utf8') as f:
        while True:
            try:
                m = q.get()
                if m == 'kill':
                    break
                for i in (m or []):
                    f.write(json.dumps(i, ensure_ascii=False) + "\n")
                f.flush()
            except Exception as e:
                print(f"file_writer e:{e}")


if __name__ == "__main__":

    channels = [
        # 'donyaye_eghtesad_com',
        'tejaratnews',
        'eghtesadonline'
    ]

    latest_pages = [get_latest_page(c) for c in channels]
    latest_pages = list(filter(lambda t: t["latest_id"] is not None, latest_pages))

    print("latest_pages ", latest_pages)

    all_reqs = []
    for row in latest_pages:
        ids = list([i for i in range(20, row["latest_id"], 20)])
        all_reqs = all_reqs + [dict(channel_name=row["channel_name"], id=i) for i in ids]

    count = os.cpu_count()
    manager = mp.Manager()
    q = manager.Queue()

    print("cpu count", count)
    print("total crawls ", len(all_reqs))

    with mp.Pool(count) as pool:
        watcher = pool.apply_async(file_writer, (q,))
        func = partial(get_results, q)
        pool.map(func, all_reqs)
        q.put('kill')
        pool.close()
        pool.join()
        watcher.close()
        watcher.join()
