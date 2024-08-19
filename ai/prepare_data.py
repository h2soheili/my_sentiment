import polars as pl
import torch
from datasets import Dataset
from transformers import PreTrainedTokenizerFast

from ai.constants import map_label_str_to_class_idx


def get_datasets(csv_path, batch_len, device, max_seq_len=512):
    my_tokenizer = PreTrainedTokenizerFast(tokenizer_file='./tokenizer_trained.json',
                                           padding_side='right',
                                           truncation_side='right',
                                           bos_token='<|begin_of_text|>',
                                           eos_token='<|end_of_text|>',
                                           pad_token='<|end_of_text|>',
                                           )
    print("...creating data")

    df = pl.read_csv(csv_path)

    data_x = df["text"].to_list()
    data_y = df["gpt_mofid"].to_list()

    n = len(data_x)

    till = int(n * 0.7)

    max_gen_len = 1

    prompt_tokens = [my_tokenizer.encode(i, max_length=max_seq_len, padding='max_length', truncation=True) for i in data_x]
    prompt_tokens = [t[:max_seq_len] for t in prompt_tokens]
    sentiment_tokens = torch.tensor([map_label_str_to_class_idx(s) for s in data_y],
                                    dtype=torch.long,
                                    device=device)

    bsz = len(prompt_tokens)
    # min_prompt_len = min(len(t) for t in prompt_tokens)
    max_prompt_len = max(len(t) for t in prompt_tokens)

    total_len = min(max_seq_len, max_gen_len + max_prompt_len)

    pad_id = my_tokenizer.pad_token_id
    tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=device)
    for k, t in enumerate(prompt_tokens):
        tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)

    # train = DataLoader(TensorDataset(tokens[:till], sentiment_tokens[:till]),
    #                    batch_size=batch_len, shuffle=True, drop_last=True)
    # val = DataLoader(TensorDataset(tokens[till:], sentiment_tokens[till:]),
    #                  batch_size=batch_len, shuffle=True, drop_last=True)

    train = Dataset.from_dict({"input_ids": tokens[:till].tolist(), "labels": sentiment_tokens[:till].tolist()})
    val = Dataset.from_dict({"input_ids": tokens[till:].tolist(), "labels": sentiment_tokens[till:].tolist()})
    print("... data created")
    return train, val
