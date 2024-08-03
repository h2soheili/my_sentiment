import polars as pl
import torch
from torch.utils.data import TensorDataset, DataLoader

from tiktoken_trained import get_tokenizer


def get_datasets(csv_path, batch_len, device, max_seq_len=512):
    my_tokenizer = get_tokenizer()

    df = pl.read_csv(csv_path)

    data_x = df["text"].to_list()
    data_y = df["target"].to_list()

    n = len(data_x)

    till = int(n * 0.7)

    max_gen_len = 1

    prompt_tokens = [my_tokenizer.encode(i, bos=True, eos=True) for i in data_x]
    prompt_tokens = [t[:max_seq_len] for t in prompt_tokens]
    sentiment_tokens = torch.tensor(data_y,
                                    dtype=torch.long,
                                    device=device)

    bsz = len(prompt_tokens)
    # min_prompt_len = min(len(t) for t in prompt_tokens)
    max_prompt_len = max(len(t) for t in prompt_tokens)

    total_len = min(max_seq_len, max_gen_len + max_prompt_len)

    pad_id = my_tokenizer.pad_id
    tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=device)
    for k, t in enumerate(prompt_tokens):
        tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)

    train = DataLoader(TensorDataset(tokens[:till], sentiment_tokens[:till]),
                       batch_size=batch_len, shuffle=True, drop_last=True)
    val = DataLoader(TensorDataset(tokens[till:], sentiment_tokens[till:]),
                     batch_size=batch_len, shuffle=True, drop_last=True)
    return train, val


def manipulate_data(csv_path):
    df = pl.read_csv(csv_path)
    import random
    arr = [random.randint(0, 4) for _ in range(len(df))]
    df = df.with_columns(pl.Series("target", arr))
    print(df.head())
    df.write_csv(csv_path)


if __name__ == '__main__':
    print(1)
    # train, val = get_datasets('./bv_news2.csv', 4, 'cpu')
    manipulate_data('./bv_news2.csv')
    print(2)
