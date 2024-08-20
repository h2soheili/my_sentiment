import polars as pl
import torch
from datasets import Dataset
from transformers import PreTrainedTokenizerFast

from ai.constants import map_label_str_to_class_idx

from typing import Any


def get_datasets(csv_path: str,
                 batch_len: int,
                 tokenizer: PreTrainedTokenizerFast,
                 device: Any, max_seq_len: int = 512):
    print("...creating data")

    df = pl.read_csv(csv_path)

    # df = df[:100]

    data_x = df["text"].to_list()
    data_y = df["gpt_mofid"].to_list()

    n = len(data_x)

    till = int(n * 0.7)

    print("data ", n)
    print("train ", till)
    print("val   ", n - till)

    prompt_tokens = [tokenizer.encode(i,
                                      max_length=max_seq_len,
                                      padding='max_length',
                                      truncation=True)
                     for i in data_x]

    prompt_tokens = [t[:max_seq_len] for t in prompt_tokens]
    target = [map_label_str_to_class_idx(s) for s in data_y]
    # target = torch.tensor(target,
    #                                 dtype=torch.long,
    #                                 device=device)
    # bsz = len(prompt_tokens)
    # max_gen_len = 1
    # min_prompt_len = min(len(t) for t in prompt_tokens)
    # max_prompt_len = max(len(t) for t in prompt_tokens)
    # total_len = min(max_seq_len, max_gen_len + max_prompt_len)
    # pad_id = tokenizer.pad_token_id
    # tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=device)
    # for k, t in enumerate(prompt_tokens):
    #     tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)
    # train = DataLoader(TensorDataset(tokens[:till], target[:till]),
    #                    batch_size=batch_len, shuffle=True, drop_last=True)
    # val = DataLoader(TensorDataset(tokens[till:], target[till:]),
    #                  batch_size=batch_len, shuffle=True, drop_last=True)
    train = Dataset.from_dict({"input_ids": prompt_tokens[:till], "labels": target[:till]})
    val = Dataset.from_dict({"input_ids": prompt_tokens[till:], "labels": target[till:]})
    print("... data created")
    return train, val
