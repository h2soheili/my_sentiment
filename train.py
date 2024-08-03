import sys

is_colab = 'google.colab' in sys.modules
if is_colab and "/content/my_sentiment" not in sys.path:
    sys.path += ['/content/my_sentiment', 'my_sentiment']

import os
import time
from tqdm import tqdm
from contextlib import nullcontext
import random
from torcheval.metrics.functional import multiclass_recall, multiclass_accuracy, multiclass_f1_score
import numpy as np
import torch
from model import ModelArgs, Transformer
from prepare_data import get_datasets
from tiktoken_trained import get_tokenizer
from dataclasses import asdict
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

out_dir = os.path.join(base_path, "out")
eval_interval = 20
log_interval = 1
eval_iters = 10
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = 'scratch'  # 'scratch' or 'resume'
data_dir = os.path.join(base_path, "data")
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
if os.path.exists(ckpt_path):
    init_from = 'resume'
# wandb logging
wandb_log = False  # disabled by default
wandb_project = 'bv'
wandb_run_name = 'bv_gpt'  # 'run' + str(time.time())

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
max_iters = 10000000  # total number of training iterations
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
backend = 'gloo'  # 'nccl', 'gloo', etc.
# backend = 'nccl'  # 'nccl', 'gloo', etc.
model_parallel_size = None
# system
# device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'mps'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile_model = False  # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# exec(open('./configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------


# os.environ['RANK'] = "0"
# os.environ['LOCAL_RANK'] = "0"
# os.environ['WORLD_SIZE'] = "1"
# os.environ['MASTER_ADDR '] = "127.0.0.1"
# os.environ['MASTER_PORT  '] = "2000"

print("torch.cuda.device_count()", torch.cuda.device_count())

# dist.init_process_group(backend='nccl', init_method='env://', rank = torch.cuda.device_count(), world_size = 1)


# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
ddp_local_rank = None
print("ddp ", ddp)
# if not ddp, we are running on a single gpu, and one process
master_process = True
seed = 1024
seed_offset = 0

os.environ["PYTHONHASHSEED"] = str(seed + seed_offset)

ddp_world_size = 1
tokens_per_iter = ddp_world_size * batch_size * block_dim
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

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

train_data_loader, val_data_loader = get_datasets(os.path.join(base_path, "bv_news2.csv"),
                                                  batch_len=batch_size,
                                                  device=device, max_seq_len=block_dim)


def get_model_params(m, only_trainable=False):
    if only_trainable:
        return sum(p.numel() for p in m.parameters() if p.requires_grad)
    return sum(p.numel() for p in m.parameters())


def calculate_metrics(t, p, classes=num_classes):
    acc = multiclass_accuracy(p, t, num_classes=classes).item()
    rec = multiclass_recall(p, t, num_classes=classes).item()
    f1 = multiclass_f1_score(p, t, num_classes=classes).item()
    return acc, rec, f1


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# model init
model_conf = ModelArgs(
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

model = None
checkpoint = None
print("init_from ", init_from)
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    model = Transformer(model_conf, device=device)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    try:
        print("loading saved model ...")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model_conf = ModelArgs(**checkpoint['model_args'])
        model = Transformer(model_conf, device=device)
        state_dict = checkpoint['model']
        model.load_state_dict(state_dict)
        model.eval()
        print("model loaded.")
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
        print("iter_num ", iter_num)
        print("best_val_loss ", best_val_loss)

    except Exception as e:
        print("error on load model ", e)

print("device                ", device)
print(model_conf)
print("params                ", f'{get_model_params(model, False):,}')
print("params only_trainable ", f'{get_model_params(model, True):,}')
print(model)

model = model.to(device)

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2))

optimizer_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters)

if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])

checkpoint = None  # free up memory

unoptimized_model = model

# compile the model
if compile_model and sys.platform != 'win32':
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model, backend='aot_eager' if device == 'mps' else 'aot_eager')  # requires PyTorch 2.0
    # model = torch.compile(model, backend='aot_eager' if device == 'mps' else 'inductor')  # requires PyTorch 2.0
    print('... compiled')

train_data_loader_iterator = iter(train_data_loader)
val_data_loader_iterator = iter(val_data_loader)


def get_random_data(split):
    global train_data_loader_iterator
    global val_data_loader_iterator
    try:
        iterator = train_data_loader_iterator if split == 'train' else val_data_loader_iterator
        return next(iterator)
    except Exception as e:
        print("get_data error ", split, e)
        if split == 'train':
            train_data_loader_iterator = iter(train_data_loader)
        else:
            val_data_loader_iterator = iter(val_data_loader)
        return get_random_data(split)


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    y = None
    logits = None
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_random_data(split)
            with ctx:
                logits, loss = model.forward(x, y)
                losses[k] = loss.item()

        logits = logits.argmax(dim=-1)
        acc, rec, f1 = calculate_metrics(y, logits)
        out[split] = dict(
            loss=losses.mean(),
            acc=acc,
            rec=rec,
            f1=f1,
        )
    return out


# logging
if wandb_log and master_process:
    import wandb

    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
t0 = time.time()
raw_model = unoptimized_model.module if ddp else unoptimized_model  # unwrap DDP container if needed
running_mfu = -1.0

model_save_path = os.path.join(out_dir, 'ckpt.pt')


def save_model():
    global checkpoint
    checkpoint = {
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': asdict(model_conf),
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'config': config,
    }
    print(f"saving checkpoint to {out_dir}")
    torch.save(checkpoint, model_save_path)
    print("... saved")


if not os.path.exists(model_save_path):
    save_model()

for i in tqdm(range(iter_num, max_iters)):
    lr = optimizer_scheduler.get_last_lr()
    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        summary = estimate_loss()
        print('\n')
        for key in summary.keys():
            print(
                f"step {iter_num} -> {key} loss:{summary[key]['loss']:.4f} \t"
                f" acc:{summary[key]['acc']:.2f}, rec:{summary[key]['rec']:.2f} f1:{summary[key]['f1']:.2f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "summary": summary,
                "lr": lr,
                "mfu": running_mfu * 100,  # convert to percentage
            })
        if summary['val']['loss'] < best_val_loss or always_save_checkpoint:
            best_val_loss = summary['val']['loss']
            if iter_num > 0:
                save_model()

    loss = 0.0
    with ctx:
        model.train()
        for j, (x, y) in enumerate(train_data_loader):
            optimizer.zero_grad()
            _, loss = model.forward(x, y)
            loss.backward()
            optimizer.step()

    optimizer_scheduler.step()

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item()
        if iter_num >= 5:  # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")
    iter_num += 1

if always_save_checkpoint:
    save_model()
