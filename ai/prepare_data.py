import multiprocessing as mp
import os
from datetime import datetime
from functools import partial
from typing import Any

import polars as pl
from datasets import Dataset
from transformers import PreTrainedTokenizerFast

from ai.constants import map_label_str_to_class_idx


def tokenize_text(tokenizer: PreTrainedTokenizerFast,
                  max_seq_len: int,
                  text: str):
    return tokenizer.encode(text,
                            max_length=max_seq_len,
                            padding='max_length',
                            truncation=True)


def get_datasets(csv_path: str,
                 batch_len: int,
                 tokenizer: PreTrainedTokenizerFast,
                 device: Any, max_seq_len: int = 512):
    print("...creating data")

    t1 = datetime.now()

    df = pl.read_csv(csv_path)

    # df = df[:100]

    data_x = df["text"].to_list()
    data_y = df["gpt_mofid"].to_list()

    n = len(data_x)
    till = int(n * 0.7)

    print("data ", n)
    print("train ", till)
    print("val   ", n - till)

    count = os.cpu_count() - 1
    print("num of process to tokenize ", count)

    with mp.Pool(count) as pool:
        func = partial(tokenize_text, tokenizer, max_seq_len)

        prompt_tokens = pool.map(func, data_x)

        target = [map_label_str_to_class_idx(s) for s in data_y]

        train = Dataset.from_dict({"input_ids": prompt_tokens[:till], "labels": target[:till]})
        val = Dataset.from_dict({"input_ids": prompt_tokens[till:], "labels": target[till:]})
        print("... data created")

        t2 = datetime.now()
        print(f"finished in {(t2 - t1).total_seconds()} second")

        return train, val


if __name__ == '__main__':
    my_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="./tokenizer_trained.json",
        padding_side='right',
        truncation_side='right',
        bos_token='<|begin_of_text|>',
        eos_token='<|end_of_text|>',
        pad_token='<|end_of_text|>',
    )
    train, val = get_datasets(csv_path='../data/bv_news_by_label.csv',
                              tokenizer=my_tokenizer,
                              batch_len=1,
                              device=None,
                              max_seq_len=512)
