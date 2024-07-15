import os

import numpy as np
import polars as pl

from extended_tiktoken import extended_encoding

df = pl.read_csv('../../../bv_news2.csv')

data_x = df["text"].to_list()
data_y = df["sentiment"].to_list()
n = len(data_x)

block_size = 128
def add_padding(arr, max_token_len=block_size):
    return [(i[:max_token_len] + ([0] * (max_token_len - len(i)))) for i in arr]


till = int(n * 0.6)

print("rows", len(df), "train ", till)

train_data_x = data_x[:till]
train_data_y = data_y[:till]

val_data_x = data_x[till:]
val_data_y = data_y[till:]

# encode with tiktoken
train_ids_x = list(map(extended_encoding.encode_ordinary, train_data_x))
train_ids_y = list(map(extended_encoding.encode_ordinary, train_data_y))

val_ids_x = list(map(extended_encoding.encode_ordinary, val_data_x))
val_ids_y = list(map(extended_encoding.encode_ordinary, val_data_y))

train_ids_x = add_padding(train_ids_x)
train_ids_y = add_padding(train_ids_y, 1)

val_ids_x = add_padding(val_ids_x)
val_ids_y = add_padding(val_ids_y, 1)

# export to bin files
train_ids_x_np = np.array(train_ids_x, dtype=np.uint32)
train_ids_y_np = np.array(train_ids_y, dtype=np.uint32)

val_ids_x_np = np.array(val_ids_x, dtype=np.uint32)
val_ids_y_np = np.array(val_ids_y, dtype=np.uint32)

base_dir = os.path.dirname(__file__)

np.save(os.path.join(base_dir, "train_x"), train_ids_x_np)
np.save(os.path.join(base_dir, "train_y"), train_ids_y_np)

np.save(os.path.join(base_dir, "val_x"), val_ids_x_np)
np.save(os.path.join(base_dir, "val_y"), val_ids_y_np)
