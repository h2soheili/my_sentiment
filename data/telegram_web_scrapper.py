import itertools

import bs4
from bs4 import BeautifulSoup
import requests
import polars as pl
import multiprocessing as mp

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


def get_results(channel_name):
    result = []

    _id = 99999999999999

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

    while True:
        try:
            params = {
                'before': _id,
            }
            res = requests.post(f"https://t.me/s/{channel_name}", params=params, headers=headers, cookies=cookies)
            if res.ok:
                text = ((res.text or "")[1:-1] or "").replace("\\n", " ")
                text = text.replace("\\", "")
                items = extract_row(channel_name, text)
                if len(items) > 0:
                    _id = int(items[0]["id"])
                    print(_id)
                    result = result + items
                    if _id <= 0:
                        print("_id <= 0  ", channel_name)
                        break
                else:
                    print("len(items) > 0  ", channel_name)
                    break
        except Exception as e:
            print("error in ", channel_name, "    ", e)

    print(channel_name, " ", len(result))

    return result


if __name__ == "__main__":

    channels = [
        'tejaratnews',
        'eghtesadonline'
    ]

    with mp.Pool(8) as pool:
        results = pool.map(get_results, channels)
        results = list(itertools.chain(*results))
        pl.from_records(results).write_csv('./telegram_p2.csv')
