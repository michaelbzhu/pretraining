"""
torchrun --standalone --nproc_per_node=8 fsdp.py
"""

import os
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
import tqdm
from pathlib import Path
from torch.nn.attention import sdpa_kernel, SDPBackend
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard, FSDPModule, MixedPrecisionPolicy
import wandb
import time
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)

# remove this to log to wandb
os.environ["WANDB_MODE"] = "disabled"

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

data_dir = "/home/ec2-user/data/fineweb100B"


# ----- CHECKPOINTER -----
# most of this logic is borrowed from
# https://github.com/pytorch/examples/blob/70922969e70218458d2a945bf86fd8cc967fc6ea/distributed/FSDP2/checkpoint.py
def get_latest_checkpoint_folder(path):
    max_num = None
    if not os.path.exists(path):
        return max_num
    for name in os.listdir(path):
        folder_path = os.path.join(path, name)
        if os.path.isdir(folder_path):
            try:
                num = int(name)
                if max_num is None or num > max_num:
                    max_num = num
            except ValueError:
                pass  # Skip non-numeric folder names
    return max_num


MODEL_CHECKPOINT = "model_state_dict.pt"
OPTIM_CHECKPOINT = "optim_state_dict.pt"


class Checkpointer:
    def __init__(self, folder: str):
        self.folder = folder
        self.last_training_time = get_latest_checkpoint_folder(f"{folder}")

    def is_empty(self):
        return self.last_training_time is None

    def load_model(self, model: FSDPModule):
        last_model_checkpoint = (
            f"{self.folder}/{self.last_training_time}/{MODEL_CHECKPOINT}"
        )
        full_sd = torch.load(
            last_model_checkpoint, mmap=True, weights_only=True, map_location="cpu"
        )
        set_model_state_dict(
            model=model,
            model_state_dict=full_sd,
            options=StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=True,
            ),
        )

    def load_optim(self, model: FSDPModule, opt: torch.optim.Optimizer):
        last_optim_checkpoint = (
            f"{self.folder}/{self.last_training_time}/{OPTIM_CHECKPOINT}"
        )
        full_sd = torch.load(
            last_optim_checkpoint, mmap=True, weights_only=True, map_location="cpu"
        )
        set_optimizer_state_dict(
            model=model,
            optimizers=opt,
            optim_state_dict=full_sd,
            options=StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=True,
            ),
        )

    def _get_full_model_state_dict(self, model: FSDPModule):
        return get_model_state_dict(
            model=model,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
            ),
        )

    def _get_full_optimizer_state_dict(
        self,
        model: FSDPModule,
        opt: torch.optim.Optimizer,
    ):
        return get_optimizer_state_dict(
            model=model,
            optimizers=opt,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
            ),
        )

    def save(self, model: FSDPModule, optim: torch.optim.Optimizer, step: int):
        model_state_dict = self._get_full_model_state_dict(model)
        optim_state_dict = self._get_full_optimizer_state_dict(model, optim)
        if torch.distributed.get_rank() == 0:
            new_training_time = int(time.time() * 1000)
            new_checkpoint_folder = f"{self.folder}/{new_training_time}"
            new_model_checkpoint = f"{new_checkpoint_folder}/{MODEL_CHECKPOINT}"
            new_optim_checkpoint = f"{new_checkpoint_folder}/{OPTIM_CHECKPOINT}"
            new_metadata_checkpoint = f"{new_checkpoint_folder}/metadata.pt"
            step_name_checkpoint = f"{new_checkpoint_folder}/step-{step}.pt"
            os.makedirs(new_checkpoint_folder, exist_ok=True)
            torch.save(model_state_dict, new_model_checkpoint)
            torch.save(optim_state_dict, new_optim_checkpoint)
            torch.save({"step": step}, new_metadata_checkpoint)
            torch.save({"step": step}, step_name_checkpoint)


# ----- MODEL ARCHITECTURE -----
class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_head, max_seq_len=8192, base=10000.0):
        super().__init__()
        self.d_head = d_head
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos and sin for maximum sequence length
        self._precompute_freqs(max_seq_len)

    def _precompute_freqs(self, seq_len):
        """Precompute cos and sin values for positions"""
        t = torch.arange(
            seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device
        )
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, d_head/2)

        # Create cos and sin embeddings
        freqs_cos = torch.cos(freqs)
        freqs_sin = torch.sin(freqs)

        # Interleave to match the dimension (seq_len, d_head)
        self.register_buffer(
            "freqs_cos", freqs_cos.repeat_interleave(2, dim=-1), persistent=False
        )
        self.register_buffer(
            "freqs_sin", freqs_sin.repeat_interleave(2, dim=-1), persistent=False
        )

    def rotate_half(self, x):
        """Rotate half the hidden dims of the input"""
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack([-x2, x1], dim=-1).flatten(-2)

    def forward(self, q, k, start_pos=0):
        """
        Apply rotary embeddings to query and key tensors
        Args:
            q: (batch_size, n_heads, seq_len, d_head)
            k: (batch_size, n_heads, seq_len, d_head)
            start_pos: starting position for caching scenarios
        Returns:
            q_rot, k_rot with rotary embeddings applied
        """
        seq_len = q.shape[2]

        # Get the precomputed frequencies for this sequence length
        freqs_cos = self.freqs_cos[start_pos : start_pos + seq_len]
        freqs_sin = self.freqs_sin[start_pos : start_pos + seq_len]

        # Apply rotary embeddings
        q_rot = q * freqs_cos + self.rotate_half(q) * freqs_sin
        k_rot = k * freqs_cos + self.rotate_half(k) * freqs_sin

        return q_rot, k_rot


class Attention(nn.Module):
    def __init__(self, d_model, n_heads, d_head):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head

        self.Wq = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.Wk = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.Wv = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.Wo = nn.Linear(n_heads * d_head, d_model, bias=False)

        # Initialize RoPE
        self.rope = RotaryPositionalEncoding(d_head)

    def forward(self, x):
        # x is shape batch_size, seq_len, d_model
        batch_size, seq_len, d_model = x.shape
        q = self.Wq(x)  # q is shape batch_size, seq_len, n_heads * d_head
        k = self.Wk(x)
        v = self.Wv(x)

        # reshape to batch_size, n_heads, seq_len, d_head
        q = q.reshape(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        q, k = self.rope(q, k)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):  # ensure use flash attention
            a = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, is_causal=True
            )  # a is (batch_size, n_heads, seq_len, d_head)
        a = a.transpose(1, 2)  # change a to (batch_size, seq_len, n_heads, d_head)
        a = a.reshape(batch_size, seq_len, self.n_heads * self.d_head)
        out = self.Wo(a)  # out is (batch_size, seq_len, d_model)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_head):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head

        self.attn = Attention(d_model, n_heads, d_head)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.ReLU(), nn.Linear(4 * d_model, d_model)
        )

        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)

    def forward(self, x):
        x = self.attn(self.norm1(x)) + x
        x = self.mlp(self.norm2(x)) + x
        return x


class GPT(nn.Module):
    def __init__(self, d_model, n_heads, d_head, n_vocab, n_layers):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.n_vocab = n_vocab

        self.embed = nn.Embedding(n_vocab, d_model)

        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_head) for _ in range(n_layers)]
        )

        self.norm = nn.RMSNorm(d_model)
        self.out_head = nn.Linear(d_model, n_vocab)

    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_head(self.norm(x))
        return x


# ----- DATA -----
def load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch in the data.bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])
    with file.open("rb", buffering=0) as f:
        # pin_memory = true so that it avoids copy when you later transfer to GPU mem
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)  # jump past header
        nbytes = f.readinto(tokens.numpy())  # read rest of file into tokens
        assert nbytes == 2 * num_tokens
    return tokens


def distributed_data_loader(batch_size, seq_len):
    filename_pattern = f"{data_dir}/fineweb_train_*.bin"
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    file_iter = iter(files)
    # i manually updated this when i wanted to restart from a checkpoint
    for _ in range(0):
        next(file_iter)
    tokens = load_data_shard(next(file_iter))
    pos = 0

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    total_batch_size = world_size * batch_size
    while True:
        if pos + total_batch_size * seq_len + 1 > len(
            tokens
        ):  # +1 for last target token
            # get next shard if we run out on current shard
            pos = 0
            next_shard = next(file_iter)
            print(f"[data_loader] now using shard {next_shard}")
            tokens = load_data_shard(next_shard)
        inputs = []
        targets = []

        for i in range(batch_size):
            # handle rank and local batch size
            start_idx = pos + (i * seq_len) + (rank * batch_size * seq_len)
            seq = tokens[start_idx : start_idx + seq_len + 1]
            inputs.append(seq[:-1])
            targets.append(seq[1:])

        inputs = torch.stack(inputs).to(
            device="cuda", dtype=torch.int32, non_blocking=True
        )
        targets = torch.stack(targets).to(
            device="cuda", dtype=torch.int64, non_blocking=True
        )  # cross entropy requires int64

        pos += total_batch_size * seq_len
        yield inputs, targets


# ----- VALIDATION -----
def distributed_val_data_loader(batch_size, seq_len):
    tokens = load_data_shard(Path(f"{data_dir}/fineweb_val_000000.bin"))
    pos = 0

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    total_batch_size = world_size * batch_size
    while True:
        if pos + total_batch_size * seq_len + 1 > len(
            tokens
        ):  # +1 for last target token
            # loop back
            pos = 0
        inputs = []
        targets = []

        for i in range(batch_size):
            start_idx = pos + (i * seq_len) + (rank * batch_size * seq_len)
            seq = tokens[start_idx : start_idx + seq_len + 1]
            inputs.append(seq[:-1])
            targets.append(seq[1:])

        inputs = torch.stack(inputs).to(
            device="cuda", dtype=torch.int32, non_blocking=True
        )
        targets = torch.stack(targets).to(
            device="cuda", dtype=torch.int64, non_blocking=True
        )  # cross entropy requires int64

        pos += total_batch_size * seq_len
        yield inputs, targets


def validation(model):
    with torch.no_grad():
        total_val_loss = 0
        dl = distributed_val_data_loader(batch_size=1, seq_len=2048)
        steps = 2
        for step in range(steps):
            i, t = next(dl)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(i)
                logits_f = logits.view(-1, logits.size(-1))
                t_f = t.view(-1)
                val_loss = F.cross_entropy(logits_f, t_f)
            total_val_loss += val_loss
        val_loss = total_val_loss / steps
        dist.all_reduce(val_loss, dist.ReduceOp.AVG)
        return val_loss


# ----- DISTRIBUTED -----
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
assert world_size == 2
assert torch.cuda.is_available()
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(
    device
)  # doing .to('cuda') after this will go to the correct rank
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
master_process = rank == 0  # this process will do logging, checkpointing etc.

_print = print


def print(*args, **kwargs):
    if master_process:
        _print(*args, **kwargs)


# ----- HYPERPARAMS -----
@dataclass
class Hyperparams:
    d_model: int = 4096
    n_vocab: int = 50257  # n_vocab of gpt2 tiktoken encoder
    n_heads: int = 32
    d_head: int = 128
    n_layers: int = 36
    steps: int = 100000
    batch_size: int = 2
    seq_len: int = 2048
    lr: float = 3e-4
    accum_steps: int = 32
    warmup_steps: int = 1000
    seed: int = 0


args = Hyperparams()
torch.manual_seed(args.seed)


def get_lr(
    step,
    warmup_steps=args.warmup_steps,
    decay_steps=0,
    total_steps=args.steps,
    base_lr=args.lr,
):
    """Learning rate with linear warmup"""
    decay_start = total_steps - decay_steps
    if step < warmup_steps:
        # Linear warmup from 0 to base_lr
        return base_lr * (step + 1) / warmup_steps
    if step > decay_start:
        # Linear decay from base_lr to 0
        return base_lr * (1 - ((step - decay_start) / decay_steps))
    else:
        # Constant learning rate after warmup
        return base_lr


# ----- INITIALIZATION -----
model = GPT(args.d_model, args.n_heads, args.d_head, args.n_vocab, args.n_layers)

if master_process:
    run = wandb.init(
        project="Pretraining a 7B model",
        config=args,
    )
    run.define_metric("*", step_metric="step")
    run.summary["model_parameters"] = sum(p.numel() for p in model.parameters())

fsdp_kwargs = {
    "mp_policy": MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
}
for layer in model.blocks:
    fully_shard(layer, **fsdp_kwargs)
fully_shard(model, **fsdp_kwargs)

assert isinstance(model, GPT)
assert isinstance(model, FSDPModule)
if master_process:
    print(model)

checkpointer = Checkpointer("fsdp7-6_checkpoints")
if checkpointer.last_training_time is not None:
    print("loading model")
    checkpointer.load_model(model)

opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
if checkpointer.last_training_time is not None:
    checkpointer.load_optim(model, opt)

start_step = 0
if checkpointer.last_training_time is not None:
    checkpoint = torch.load(
        f"{checkpointer.folder}/{checkpointer.last_training_time}/metadata.pt",
        map_location=device,
    )
    start_step = checkpoint["step"] + 1


# ----- TRAINING LOOP -----
dl = distributed_data_loader(args.batch_size, args.seq_len)

pbar = tqdm.tqdm(
    range(start_step, args.steps),
    disable=not master_process,
    initial=start_step,
    total=args.steps,
)
for step in pbar:
    current_lr = get_lr(step)
    for param_group in opt.param_groups:
        param_group["lr"] = current_lr

    total_loss = 0
    for k in range(args.accum_steps):
        i, t = next(dl)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(i)
            if master_process:
                if step % 20 == 0:
                    min_logits = logits.min().item()
                    max_logits = logits.max().item()
                    std_logits = logits.std().item()
                    run.log(
                        {
                            "step": step,
                            "logits_min": min_logits,
                            "logits_max": max_logits,
                            "logits_std": std_logits,
                        }
                    )
            logits_flattened = logits.view(-1, logits.size(-1))
            t_flattened = t.view(-1)
            loss = F.cross_entropy(logits_flattened, t_flattened)
            loss = loss / args.accum_steps
        loss.backward()
        total_loss += loss.item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()
    opt.zero_grad()
    pbar.set_postfix({"loss": total_loss})

    tokens_constant = 0  # updated this manually when restarting from a checkpoint
    tokens_processed = (
        (step - start_step + 1)
        * args.batch_size
        * args.accum_steps
        * args.seq_len
        * world_size
    ) + tokens_constant
    if master_process:
        run.log(
            {
                "step": step,
                "train_loss": total_loss,
                "tokens_processed": tokens_processed,
            }
        )

    if step % 1000 == 0 or step == args.steps - 1:
        val_loss = validation(model)
        print(f"Step {step}, Loss: {total_loss:.4f}, Val_Loss: {val_loss:.4f}")
        if master_process:
            run.log(
                {
                    "step": step,
                    "val_loss": val_loss.item(),
                    "tokens_processed": tokens_processed,
                }
            )
        pbar.set_postfix({"val_loss": val_loss.item()})

    if (step % 10000 == 0 or step == args.steps - 1) and step != 0:
        checkpointer.save(model, opt, step)

dist.destroy_process_group()
