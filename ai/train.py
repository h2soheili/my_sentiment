import sys

from torcheval.metrics.functional import multiclass_accuracy, multiclass_recall, multiclass_f1_score

is_colab = 'google.colab' in sys.modules
if is_colab and "/content/my_sentiment" not in sys.path:
    sys.path += ['/content/my_sentiment', 'my_sentiment']

from transformers import Trainer, TrainingArguments, EvalPrediction
from mode_extra import get_model_params
import os
from contextlib import nullcontext
import random
import numpy as np
import torch
from model import ModelConfig, MyTransformer
from prepare_data import get_datasets
from tiktoken_trained import get_tokenizer
from torch.optim import lr_scheduler

my_tokenizer = get_tokenizer()

print("extended_encoding.max_token_value", my_tokenizer.n_words)
vocab_size = my_tokenizer.n_words + 1
pad_id = my_tokenizer.pad_id
print("vocab_size", vocab_size)
print("pad_id", pad_id)

# -----------------------------------------------------------------------------
# default config values
# I/O
base_path = '/content/gdrive/MyDrive/mycolab/my_sentiment' if is_colab else '.'
data_dir = os.path.join(base_path, "data")
model_dir = os.path.join(base_path, "out", 'model_01')
batch_size = 18  # if gradient_accumulation_steps > 1, this is the micro-batch size
block_dim = 128
max_seq_len = block_dim
# model
n_layer = 32
n_head = 16
num_classes = 5
dropout = 0.1  # for pretraining 0 is good, for finetuning try 0.1+
bias = True  # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4  # max learning rate
max_iters = 100_000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 500  # how many steps to warm up for
lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
# backend = 'nccl'  # 'nccl', 'gloo', etc.
# device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'mps'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
should_compile_model = False  # use PyTorch 2.0 to compile the model to be faster
seed = 1024
seed_offset = 0
eval_interval = 20
log_interval = 20
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


if __name__ == "__main__":

    os.environ["PYTHONHASHSEED"] = str(seed + seed_offset)

    random.seed(seed + seed_offset)
    np.random.seed(seed + seed_offset)
    torch.cuda.manual_seed(seed + seed_offset)
    torch.manual_seed(seed + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
    device = torch.device(device)

    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    train_data_loader, val_data_loader = get_datasets(os.path.join(data_dir, "bv_news2.csv"),
                                                      batch_len=batch_size,
                                                      device=device, max_seq_len=block_dim)

    training_args = TrainingArguments(
        output_dir=model_dir,
        num_train_epochs=max_iters,
        evaluation_strategy="steps",
        eval_steps=100,
        logging_strategy="steps",
        logging_steps=20,
    )

    model_conf = ModelConfig(
        dim=block_dim,
        n_layers=n_layer,
        n_heads=n_head,
        vocab_size=vocab_size,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        pad_id=pad_id,
        dropout=dropout,
        num_classes=num_classes,
        bias=bias
    )

    model = MyTransformer(model_conf, device=device)

    if os.path.exists(model_dir):
        model_conf = ModelConfig.from_pretrained(model_dir)
        model = MyTransformer.from_pretrained(model_dir, config=model_conf)
        print("model loaded")
    else:
        model_conf.save_pretrained(model_dir)
        model.save_pretrained(model_dir)
        print("model saved ")

    print("device                ", device)
    print(model_conf)
    print("params                ", f'{get_model_params(model, False):,}')
    print("params only_trainable ", f'{get_model_params(model, True):,}')
    print(model)
    model = model.to(device=device)

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2))

    optimizer_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters)

    compiled_model = model

    # compile the model
    if should_compile_model and sys.platform != 'win32':
        print("compiling the model... (takes a ~minute)")
        compiled_model = torch.compile(model,
                                       backend='aot_eager' if device == 'mps' else 'aot_eager')  # requires PyTorch 2.0
        print('... compiled')


    def compute_metrics(eval_pred: EvalPrediction):
        predictions = torch.from_numpy(eval_pred.predictions)
        label_ids = torch.from_numpy(eval_pred.label_ids)
        acc = multiclass_accuracy(predictions, label_ids, num_classes=num_classes).item()
        rec = multiclass_recall(predictions, label_ids, num_classes=num_classes).item()
        f1 = multiclass_f1_score(predictions, label_ids, num_classes=num_classes).item()
        return {"acc": acc, "rec": rec, "f1": f1}


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data_loader,
        eval_dataset=val_data_loader,
        optimizers=(optimizer, optimizer_scheduler),
        tokenizer=None,
        compute_metrics=compute_metrics
    )

    trainer.train()
