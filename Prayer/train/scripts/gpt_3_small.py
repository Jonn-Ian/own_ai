# llm.py
from __future__ import annotations
import os, json, math, shutil, argparse, hashlib, random
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Optional libs
try:
    from sentencepiece import SentencePieceProcessor, SentencePieceTrainer
except Exception:
    SentencePieceProcessor = None
    SentencePieceTrainer = None
try:
    import bitsandbytes as bnb
except Exception:
    bnb = None
try:
    import wandb
except Exception:
    wandb = None
try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

RAW_DATA_DIR   = "assets/raw"
PROCESSED_DIR  = "assets/processed"
TOKENS_DIR     = "assets/tokens"
SPM_PREFIX     = "assets/tokens/spm"
SAVE_DIR       = "assets/weights"

DEFAULT_VOCAB_SIZE   = 32000
DEFAULT_MAX_SEQ_LEN  = 2048
DEFAULT_STRIDE       = 512
MODEL_FILENAME       = "model.pt"
METRICS_FILENAME     = "metrics.json"

DEFAULT = {
    "n_layers": 16,
    "n_heads": 16,
    "d_model": 1024,
    "d_ff": 4096,
    "dropout": 0.1,
    "max_seq_len": 2048,
    "batch_size": 2,
    "grad_accum_steps": 32,
    "lr": 2e-4,
    "weight_decay": 0.01,
    "betas": (0.9, 0.95),
    "warmup_steps": 2000,
    "total_steps": 60000,
    "save_every": 2000,
    "vocab_size": DEFAULT_VOCAB_SIZE,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "use_checkpoint": True,
    "seed": 42,
    "log_every": 100,
    "eval_every": 1000,
    "clip_grad": 1.0,
    "use_wandb": False,
    "wandb_project": "neurosama-style-train",
    "wandb_run_name": None,
    # RLHF defaults
    "rlhf_enabled": False,
    "rm_lr": 1e-4,
    "ppo_lr": 1e-5,
    "ppo_epochs": 1,
    "ppo_batch_size": 2,
    "ppo_kl_coef": 0.02,
    "ppo_clip": 0.2,
    "rm_train_steps": 10000,
}

# -------------------------
# Dataset type detection
# -------------------------
def detect_dataset_type(jsonl_path: str) -> str:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        try:
            first = json.loads(next(f))
        except StopIteration:
            return "Unknown"
    if "conversation" in first:
        return "SFT"
    elif {"prompt", "pos", "neg"} <= first.keys():
        return "RLHF"
    elif {"instruction", "input", "output"} <= first.keys():
        return "INSTRUCT"
    else:
        return "Unknown"

# -------------------------
# Utilities
# -------------------------
def setup_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def is_main_process() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)

def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

def atomic_overwrite(obj: Any, path: str):
    tmp = f"{path}.tmp"; torch.save(obj, tmp); os.replace(tmp, path)

def atomic_write_json(obj: Any, path: str):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

# -------------------------
# Hugging Face → JSONL conversion
# -------------------------
def convert_hf_dataset(dataset_name="daily_dialog", out_dir=RAW_DATA_DIR):
    if load_dataset is None:
        raise RuntimeError("datasets library required: pip install datasets")
    os.makedirs(out_dir, exist_ok=True)
    ds = load_dataset(dataset_name)
    for split in ds.keys():
        out_file = Path(out_dir) / f"{split}.jsonl"
        with out_file.open("w", encoding="utf-8") as f:
            for ex in ds[split]:
                dialog = ex["dialog"]
                parts = []
                for i, turn in enumerate(dialog):
                    role = "[USER]" if i % 2 == 0 else "[ASSISTANT]"
                    parts.append(f"{role} {turn}")
                text = "\n".join(parts)
                f.write(json.dumps({"conversation": text}, ensure_ascii=False) + "\n")
    print(f"[INFO] Converted {dataset_name} to JSONL in {out_dir}")

# -------------------------
# Instruction → Conversation conversion
# -------------------------
def convert_instruct_to_conversation(jsonl_in: str, jsonl_out: str):
    with open(jsonl_in, "r", encoding="utf-8") as fin, \
         open(jsonl_out, "w", encoding="utf-8") as fout:
        for ln in fin:
            obj = json.loads(ln)
            instr = obj.get("instruction", "")
            inp = obj.get("input", "")
            out = obj.get("output", "")
            parts = []
            if instr:
                parts.append(f"[USER] {instr}")
            if inp:
                parts.append(f"[USER] {inp}")
            parts.append(f"[ASSISTANT] {out}")
            text = "\n".join(parts)
            fout.write(json.dumps({"conversation": text}, ensure_ascii=False) + "\n")
    print(f"[INFO] Converted instruction dataset {jsonl_in} → {jsonl_out}")

# -------------------------
# Preprocess: dedup + split + shard
# -------------------------
def preprocess(input_dir: str, out_dir: str, val_ratio: float = 0.02, shard_size: int = 200_000):
    input_dir = Path(input_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cleaned = out_dir / "cleaned.jsonl"
    seen = set()
    total = kept = 0
    with cleaned.open("w", encoding="utf-8") as cf:
        for fn in sorted(os.listdir(input_dir)):
            p = input_dir / fn
            if not p.is_file() or not fn.endswith(".jsonl"):
                continue
            dtype = detect_dataset_type(str(p))
            if dtype == "INSTRUCT":
                tmp_out = out_dir / f"{fn}.conv.jsonl"
                convert_instruct_to_conversation(str(p), str(tmp_out))
                p = tmp_out
            with p.open("r", encoding="utf-8") as f:
                for ln in f:
                    total += 1
                    s = ln.strip()
                    if not s:
                        continue
                    h = sha256(s)
                    if h in seen:
                        continue
                    seen.add(h)
                    cf.write(s + "\n")
                    kept += 1
    print(f"[INFO] Preprocess done: read {total}, kept {kept}")

    train_lines, val_lines = [], []
    with cleaned.open("r", encoding="utf-8") as inf:
        for ln in inf:
            h = sha256(ln)
            v = int(h[:6], 16)
            if v % 10000 < int(val_ratio * 10000):
                val_lines.append(ln)
            else:
                train_lines.append(ln)

    def write_shards(lines: List[str], prefix: str):
        if not lines:
            return
        shard_idx = 0
        for i in range(0, len(lines), shard_size):
            shard = lines[i:i + shard_size]
            out = out_dir / f"{prefix}_shard_{shard_idx:05d}.jsonl"
            with out.open("w", encoding="utf-8") as f:
                for s in shard:
                    f.write(s)
            shard_idx += 1
        print(f"[INFO] Wrote {shard_idx} shards for {prefix}")

    write_shards(train_lines, "train")
    val_out = out_dir / "val.jsonl"
    with val_out.open("w", encoding="utf-8") as vf:
        for s in val_lines:
            vf.write(s)
    print(f"[INFO] train shards + val written in {out_dir}")
# -------------------------
# (Parts 2–8: build_tokens, datasets, model, training, RLHF, driver, CLI)
# -------------------------

# -------------------------
# Build tokens: SentencePiece + flat ids/offsets per shard
# -------------------------
def build_tokens(data_dir: str, save_dir: str, sp_prefix: str,
                 vocab_size: int, max_seq_len: int, stride: int):
    if SentencePieceTrainer is None or SentencePieceProcessor is None:
        raise RuntimeError("sentencepiece required: pip install sentencepiece")
    data_dir = Path(data_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    sp_model = Path(sp_prefix).with_suffix(".model")
    if not sp_model.exists():
        formatted = save_dir / "spm_input.txt"
        with formatted.open("w", encoding="utf-8") as fout:
            for fn in sorted(data_dir.glob("train_shard_*.jsonl")):
                with fn.open("r", encoding="utf-8") as f:
                    for ln in f:
                        obj = json.loads(ln)
                        fout.write(obj["conversation"] + "\n")
            val_path = data_dir / "val.jsonl"
            if val_path.exists():
                with val_path.open("r", encoding="utf-8") as f:
                    for ln in f:
                        obj = json.loads(ln)
                        fout.write(obj["conversation"] + "\n")
        SentencePieceTrainer.train(
            input=str(formatted),
            model_prefix=str(sp_prefix),
            vocab_size=vocab_size,
            model_type="bpe",
            user_defined_symbols="[USER],[ASSISTANT]",
            character_coverage=1.0,
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            hard_vocab_limit=False
        )

    sp = SentencePieceProcessor(model_file=str(sp_model))

    def process_file(jsonl_path: Path):
        all_ids, offs, cur_pos = [], [], 0
        with jsonl_path.open("r", encoding="utf-8") as f:
            for ln in f:
                obj = json.loads(ln)
                text = obj["conversation"]
                ids = sp.encode(text, out_type=int)
                ids_full = [sp.bos_id()] + ids + [sp.eos_id()]
                L = len(ids_full)
                if L <= max_seq_len:
                    arr = np.array(ids_full, dtype=np.int32)
                    all_ids.append(arr)
                    offs.append((cur_pos, arr.shape[0]))
                    cur_pos += arr.shape[0]
                else:
                    start = 0
                    while start < L:
                        end = min(start + max_seq_len, L)
                        chunk = ids_full[start:end]
                        if len(chunk) < 2: break
                        arr = np.array(chunk, dtype=np.int32)
                        all_ids.append(arr)
                        offs.append((cur_pos, arr.shape[0]))
                        cur_pos += arr.shape[0]
                        if end == L: break
                        start += stride
        if not all_ids:
            return None, None
        lens = [a.shape[0] for a in all_ids]
        flat = np.empty(sum(lens), dtype=np.int32)
        pos = 0
        for a in all_ids:
            flat[pos:pos + a.shape[0]] = a
            pos += a.shape[0]
        offs_arr = np.array(offs, dtype=np.int64)
        return flat, offs_arr

    shard_paths = sorted(data_dir.glob("train_shard_*.jsonl"))
    for i, spth in enumerate(shard_paths):
        flat, offs = process_file(spth)
        if flat is None: continue
        np.save(save_dir / f"train_ids_{i:05d}.npy", flat)
        np.save(save_dir / f"train_offs_{i:05d}.npy", offs)
    val_path = data_dir / "val.jsonl"
    if val_path.exists():
        val_flat, val_offs = process_file(val_path)
        if val_flat is not None:
            np.save(save_dir / "val_ids.npy", val_flat)
            np.save(save_dir / "val_offs.npy", val_offs)

    shutil.copy(str(sp_model), str(Path(save_dir) / sp_model.name))
    meta = {"vocab_size": vocab_size, "max_seq_len": max_seq_len, "stride": stride}
    atomic_write_json(meta, str(Path(save_dir) / "meta.json"))
    print("[INFO] build_tokens complete.")

# -------------------------
# Dataset classes
# -------------------------
class TokenWindowDataset(Dataset):
    def __init__(self, ids_path: str, offs_path: str, vocab_size: int,
                 special_resp_piece: Optional[int], pad_id: int = 0):
        self.ids = np.load(ids_path, mmap_mode="r")
        self.offs = np.load(offs_path, mmap_mode="r")
        self.vocab_size = vocab_size
        self.resp_piece = special_resp_piece
        self.pad_id = pad_id

    def __len__(self): return self.offs.shape[0]

    def __getitem__(self, idx):
        start, length = int(self.offs[idx, 0]), int(self.offs[idx, 1])
        arr = self.ids[start:start + length].astype(np.int64)
        if arr.shape[0] < 2:
            x = np.array([self.pad_id], dtype=np.int64)
            y = np.array([-100], dtype=np.int64)
        else:
            x = arr[:-1]; y = arr[1:].copy()
        if self.resp_piece is not None:
            resp_pos = None
            for i, tid in enumerate(arr):
                if int(tid) == int(self.resp_piece):
                    resp_pos = i; break
            if resp_pos is not None:
                cutoff = resp_pos
                for i in range(min(cutoff, y.shape[0])):
                    y[i] = -100
        return torch.from_numpy(x), torch.from_numpy(y)

class ShardedTokenDataset(IterableDataset):
    def __init__(self, tokens_dir: str, vocab_size: int, resp_piece: Optional[int], pad_id: int = 0,
                 shard_indices: List[int] | None = None):
        super().__init__()
        self.tokens_dir = Path(tokens_dir)
        self.vocab_size = vocab_size
        self.resp_piece = resp_piece
        self.pad_id = pad_id
        self.shards = sorted(self.tokens_dir.glob("train_ids_*.npy"))
        if shard_indices is not None:
            self.shards = [self.shards[i] for i in shard_indices if i < len(self.shards)]

    def __iter__(self) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        shards = [p for i, p in enumerate(self.shards) if i % num_workers == worker_id]
        for ids_path in shards:
            offs_path = Path(str(ids_path).replace("train_ids_", "train_offs_"))
            ids = np.load(ids_path, mmap_mode="r")
            offs = np.load(offs_path, mmap_mode="r")
            for k in range(offs.shape[0]):
                start, length = int(offs[k, 0]), int(offs[k, 1])
                arr = ids[start:start + length].astype(np.int64)
                if arr.shape[0] < 2:
                    x = np.array([self.pad_id], dtype=np.int64)
                    y = np.array([-100], dtype=np.int64)
                else:
                    x = arr[:-1]; y = arr[1:].copy()
                if self.resp_piece is not None:
                    resp_pos = None
                    for i, tid in enumerate(arr):
                        if int(tid) == int(self.resp_piece):
                            resp_pos = i; break
                    if resp_pos is not None:
                        cutoff = resp_pos
                        for i in range(min(cutoff, y.shape[0])):
                            y[i] = -100
                yield torch.from_numpy(x), torch.from_numpy(y)

def collate_tokens(batch: List[Tuple[torch.Tensor, torch.Tensor]], pad_id: int):
    xs, ys = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    Xs, Ys = [], []
    for x, y in zip(xs, ys):
        if x.size(0) < max_len:
            pad_len = max_len - x.size(0)
            Xs.append(torch.cat([x, torch.full((pad_len,), pad_id, dtype=torch.long)], dim=0))
            Ys.append(torch.cat([y, torch.full((pad_len,), -100, dtype=torch.long)], dim=0))
        else:
            Xs.append(x); Ys.append(y)
    return torch.stack(Xs), torch.stack(Ys)
# -------------------------
# Model, schedulers, EMA, loss, eval
# -------------------------
def maybe_checkpoint(fn, x, use_checkpoint: bool):
    if use_checkpoint and torch.is_grad_enabled():
        return torch.utils.checkpoint.checkpoint(fn, x, use_reentrant=False)
    else:
        return fn(x)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout, max_seq_len):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()
        q = self.q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            is_causal=True,
            dropout_p=self.attn_drop.p if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out(y))

class MLP(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))

class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, max_seq_len, use_checkpoint=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, eps=1e-5)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout, max_seq_len)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-5)
        self.mlp = MLP(d_model, d_ff, dropout)
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        def attn_fn(z): return self.attn(self.ln1(z))
        def mlp_fn(z):  return self.mlp(self.ln2(z))
        x = x + maybe_checkpoint(attn_fn, x, self.use_checkpoint)
        x = x + maybe_checkpoint(mlp_fn, x, self.use_checkpoint)
        return x

class GPTSmall(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_seq_len, dropout, use_checkpoint=False):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, d_ff, dropout, max_seq_len, use_checkpoint) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model, eps=1e-5)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init_weights)
        # Tie weights
        self.head.weight = self.tok_emb.weight

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
            nn.init.normal_((m.weight), mean=0.0, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx: torch.Tensor):
        B, T = idx.size()
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.head(x)

class CosineWithWarmup:
    def __init__(self, optimizer, warmup_steps, total_steps, base_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.step_num = 0

    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup_steps and self.warmup_steps > 0:
            lr_mult = self.step_num / float(max(1, self.warmup_steps))
        else:
            progress = (self.step_num - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr_mult = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = self.base_lr * lr_mult
        for g in self.optimizer.param_groups:
            g["lr"] = lr

class EMA:
    def __init__(self, model: nn.Module, decay: float, device: torch.device):
        self.decay = decay
        self.device = device
        self.shadow: Dict[str, torch.Tensor] = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.detach().clone().to(self.device)
        self.backup: Dict[str, torch.Tensor] = {}

    def update(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad:
                new = (1.0 - self.decay) * p.data.detach().to(self.device) + self.decay * self.shadow[n]
                self.shadow[n] = new.clone()

    def apply(self, model: nn.Module):
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.data.detach().clone()
                p.data = self.shadow[n].to(p.data.device)

    def restore(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data = self.backup[n].to(p.data.device)
        self.backup = {}

def compute_loss(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100):
    B, T, V = logits.size()
    return F.cross_entropy(logits.view(B * T, V), targets.view(B * T), ignore_index=ignore_index)

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, pad_id: int):
    model.eval()
    tot_loss = 0.0
    n = 0
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        logits = model(X)
        loss = compute_loss(logits, Y, ignore_index=-100)
        tot_loss += float(loss.item())
        n += 1
    avg = tot_loss / max(1, n)
    ppl = math.exp(min(20.0, avg))
    return {"loss": avg, "ppl": ppl}

# -------------------------
# DDP utilities
# -------------------------
def ddp_init(local_rank: Optional[int] = None):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", local_rank or 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
        return rank, world_size, local_rank
    else:
        return 0, 1, local_rank or 0

def ddp_wrap(model: nn.Module):
    if dist.is_available() and dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()] if torch.cuda.is_available() else None,
            find_unused_parameters=False
        )
    return model

# -------------------------
# RLHF components
# -------------------------
class RewardModel(nn.Module):
    """
    Simple reward head trained on preference pairs.
    """
    def __init__(self, base_model: GPTSmall, d_model: int):
        super().__init__()
        self.base = base_model
        self.value_head = nn.Linear(d_model, 1)

    def forward(self, input_ids: torch.Tensor, pad_id: Optional[int] = None):
        B, T = input_ids.size()
        pos = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
        x = self.base.tok_emb(input_ids) + self.base.pos_emb(pos)
        x = self.base.drop(x)
        for blk in self.base.blocks:
            x = blk(x)
        x = self.base.ln_f(x)
        if pad_id is not None:
            mask = (input_ids != pad_id).unsqueeze(-1).float()
            denom = mask.sum(dim=1).clamp(min=1.0)
            pooled = (x * mask).sum(dim=1) / denom
        else:
            pooled = x.mean(dim=1)
        return self.value_head(pooled).squeeze(-1)

def preference_loss(r_pos: torch.Tensor, r_neg: torch.Tensor):
    return -F.logsigmoid(r_pos - r_neg).mean()

class PreferencePairDataset(Dataset):
    """
    JSONL with {"prompt": "...", "pos": "...", "neg": "..."}.
    """
    def __init__(self, jsonl_path: str, sp: SentencePieceProcessor, max_len: int):
        self.items = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for ln in f:
                obj = json.loads(ln)
                prompt = obj["prompt"]
                pos = obj["pos"]
                neg = obj["neg"]
                prompt_ids = sp.encode(prompt, out_type=int)[:max_len]
                pos_ids = sp.encode(prompt + "\n[ASSISTANT] " + pos, out_type=int)[:max_len]
                neg_ids = sp.encode(prompt + "\n[ASSISTANT] " + neg, out_type=int)[:max_len]
                self.items.append((prompt_ids, pos_ids, neg_ids))
        self.pad_id = sp.pad_id() if hasattr(sp, "pad_id") else 0

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        prompt_ids, pos_ids, neg_ids = self.items[idx]
        return (
            torch.tensor(prompt_ids, dtype=torch.long),
            torch.tensor(pos_ids, dtype=torch.long),
            torch.tensor(neg_ids, dtype=torch.long),
        )

def collate_pref(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], pad_id: int):
    prompts, pos_list, neg_list = zip(*batch)
    max_len_prompt = max(x.size(0) for x in prompts)
    max_len_resp = max(max(x.size(0), y.size(0)) for x, y in zip(pos_list, neg_list))

    def pad(seq, L):
        if seq.size(0) < L:
            return torch.cat([seq, torch.full((L - seq.size(0),), pad_id, dtype=torch.long)], dim=0)
        return seq

    prompts_padded = torch.stack([pad(x, max_len_prompt) for x in prompts])
    pos_padded = torch.stack([pad(x, max_len_resp) for x in pos_list])
    neg_padded = torch.stack([pad(x, max_len_resp) for x in neg_list])
    return prompts_padded, pos_padded, neg_padded

@torch.no_grad()
def sample_next_token_greedy(model: nn.Module, input_ids: torch.Tensor):
    logits = model(input_ids)
    next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    return next_id

@torch.no_grad()
def generate_greedy(model: nn.Module, queries: torch.Tensor, max_new_tokens: int = 64, eos_id: Optional[int] = None):
    seq = queries.clone()
    for _ in range(max_new_tokens):
        next_id = sample_next_token_greedy(model, seq)
        seq = torch.cat([seq, next_id], dim=1)
        if eos_id is not None and (next_id == eos_id).all():
            break
    return seq[:, -max_new_tokens:]  # return response segment only

def ppo_step(policy: GPTSmall, old_policy: GPTSmall, reward_model: RewardModel,
             queries: torch.Tensor, responses: torch.Tensor, optimizer: torch.optim.Optimizer,
             kl_coef: float, clip: float, pad_id: int):
    policy.train()
    old_policy.eval()

    with torch.no_grad():
        old_logits = old_policy(torch.cat([queries, responses], dim=1))
        old_logprobs = F.log_softmax(old_logits, dim=-1)

    logits = policy(torch.cat([queries, responses], dim=1))
    logprobs = F.log_softmax(logits, dim=-1)

    rewards = reward_model(torch.cat([queries, responses], dim=1), pad_id=pad_id)

    # Token-level KL on taken actions
    resp_logp_new = logprobs[:, -responses.size(1):, :]
    resp_logp_old = old_logprobs[:, -responses.size(1):, :]
    taken = responses.unsqueeze(-1)
    lp_new = resp_logp_new.gather(-1, taken).squeeze(-1)  # [B, T_resp]
    lp_old = resp_logp_old.gather(-1, taken).squeeze(-1)
    kl = (lp_old - lp_new).mean()

    resp_lp = lp_new.mean(dim=1)

    objective = (rewards - kl_coef * kl + resp_lp).mean()
    loss = -objective

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()

    return {
        "ppo_loss": float(loss.item()),
        "reward": float(rewards.mean().item()),
        "kl": float(kl.item()),
        "resp_lp": float(resp_lp.mean().item())
    }

# -------------------------
# Supervised training loop
# -------------------------
def sft_train_loop(model: nn.Module, train_dl: DataLoader, val_dl: Optional[DataLoader],
                   device: torch.device, config: Dict[str, Any], save_dir: str,
                   ema: EMA, optim: torch.optim.Optimizer, sched: CosineWithWarmup,
                   scaler: GradScaler, sp: SentencePieceProcessor):
    vocab_size = sp.get_piece_size()
    pad_id = sp.pad_id() if hasattr(sp, "pad_id") else 0
    step = 0
    best_val = float("inf")
    model_path = os.path.join(save_dir, MODEL_FILENAME)
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device)
        (model.module if hasattr(model, "module") else model).load_state_dict(ckpt["model"])
        if "optimizer" in ckpt and ckpt["optimizer"] is not None:
            try:
                optim.load_state_dict(ckpt["optimizer"])
            except Exception:
                pass
        if "ema" in ckpt and ckpt["ema"] is not None:
            for k, v in ckpt["ema"].items():
                ema.shadow[k] = v.to(device)
        step = ckpt.get("step", 0)
        best_val = ckpt.get("best_val", best_val)

    writer = SummaryWriter(log_dir=str(Path(save_dir) / "tb")) if is_main_process() else None
    if wandb and config.get("use_wandb", False) and is_main_process():
        wandb.init(project=config.get("wandb_project", "neurosama-style-train"),
                   name=config.get("wandb_run_name"), config=config)

    total_steps = config["total_steps"]
    grad_accum = config["grad_accum_steps"]
    progress = tqdm(total=total_steps, desc="sft-train", unit="step", initial=step) if is_main_process() else None
    accum = 0
    (model.module if hasattr(model, "module") else model).train()

    try:
        while step < total_steps:
            if isinstance(train_dl.sampler, DistributedSampler):
                train_dl.sampler.set_epoch(step)
            for X, Y in train_dl:
                X = torch.clamp(X.to(device), min=0, max=vocab_size - 1)
                Y = Y.to(device)
                Y_clone = Y.clone()
                Y_clone[X == pad_id] = -100
                with autocast(enabled=(device.type == "cuda")):
                    logits = model(X)
                    loss = compute_loss(logits, Y_clone, ignore_index=-100) / float(grad_accum)
                if not torch.isfinite(loss):
                    optim.zero_grad(set_to_none=True)
                    continue
                scaler.scale(loss).backward()
                accum += 1
                if accum >= grad_accum:
                    scaler.unscale_(optim)
                    if config["clip_grad"] and config["clip_grad"] > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad"])
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad(set_to_none=True)
                    sched.step()
                    ema.update(model.module if hasattr(model, "module") else model)
                    step += 1
                    accum = 0

                    if is_main_process():
                        if progress:
                            progress.update(1)
                            progress.set_postfix({"loss": float(loss.item() * grad_accum), "lr": optim.param_groups[0]["lr"]})
                        if writer and step % config["log_every"] == 0:
                            writer.add_scalar("train/loss", float(loss.item() * grad_accum), step)
                            writer.add_scalar("train/lr", optim.param_groups[0]["lr"], step)
                        if wandb and config.get("use_wandb", False) and step % config["log_every"] == 0:
                            wandb.log({"train/loss": float(loss.item() * grad_accum), "train/lr": optim.param_groups[0]["lr"], "step": step})

                    if val_dl is not None and step % config["eval_every"] == 0:
                        ema.apply(model.module if hasattr(model, "module") else model)
                        val_metrics = evaluate(model.module if hasattr(model, "module") else model, val_dl, device, pad_id)
                        ema.restore(model.module if hasattr(model, "module") else model)
                        if is_main_process():
                            if writer:
                                writer.add_scalar("val/loss", val_metrics["loss"], step)
                                writer.add_scalar("val/ppl", val_metrics["ppl"], step)
                            if wandb and config.get("use_wandb", False):
                                wandb.log({"val/loss": val_metrics["loss"], "val/ppl": val_metrics["ppl"], "step": step})
                            best_val = min(best_val, val_metrics["loss"])

                    if is_main_process() and (step % config["save_every"] == 0 or step >= total_steps):
                        ckpt = {
                            "model": (model.module if hasattr(model, "module") else model).state_dict(),
                            "optimizer": optim.state_dict(),
                            "ema": {k: v.cpu() for k, v in ema.shadow.items()},
                            "step": step,
                            "best_val": best_val
                        }
                        atomic_overwrite(ckpt, os.path.join(save_dir, MODEL_FILENAME))
                        atomic_write_json({"step": step, "best_val": best_val}, os.path.join(save_dir, METRICS_FILENAME))
                        print(f"\n[INFO] SFT checkpoint saved at step {step}. best_val={best_val}")
                    if step >= total_steps:
                        break
            if step >= total_steps:
                break
    finally:
        if is_main_process():
            if writer: writer.close()
            if wandb and config.get("use_wandb", False): wandb.finish()
            if progress: progress.close()

# -------------------------
# RLHF training loop (Reward Model + PPO fine-tuning)
# -------------------------
def rlhf_train(tokens_dir: str, save_dir: str, config: Dict[str, Any], local_rank: Optional[int] = None):
    rank, world_size, local_rank = ddp_init(local_rank)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    setup_seed(config.get("seed", 42))
    os.makedirs(save_dir, exist_ok=True)

    # tokenizer
    spm_model = None
    try:
        spm_model = next(Path(tokens_dir).glob("*.model"))
    except StopIteration:
        spm_model = None
    if spm_model is None or SentencePieceProcessor is None:
        raise RuntimeError("spm.model not found or sentencepiece missing; run build_tokens and install sentencepiece")
    sp = SentencePieceProcessor(model_file=str(spm_model))
    vocab_size = sp.get_piece_size()
    pad_id = sp.pad_id() if hasattr(sp, "pad_id") else 0

    # load SFT policy
    base_ckpt = torch.load(os.path.join(save_dir, MODEL_FILENAME), map_location=device)
    policy = GPTSmall(
        vocab_size, config["d_model"], config["n_layers"], config["n_heads"],
        config["d_ff"], config["max_seq_len"], config["dropout"],
        use_checkpoint=config.get("use_checkpoint", False)
    ).to(device)
    policy.load_state_dict(base_ckpt["model"])
    old_policy = GPTSmall(
        vocab_size, config["d_model"], config["n_layers"], config["n_heads"],
        config["d_ff"], config["max_seq_len"], config["dropout"], use_checkpoint=False
    ).to(device)
    old_policy.load_state_dict(base_ckpt["model"])
    old_policy.eval()

    rm_backbone = GPTSmall(
        vocab_size, config["d_model"], config["n_layers"], config["n_heads"],
        config["d_ff"], config["max_seq_len"], config["dropout"], use_checkpoint=False
    ).to(device)
    rm = RewardModel(rm_backbone, d_model=config["d_model"]).to(device)

    # preference dataset
    pref_path = Path(tokens_dir) / "preferences.jsonl"
    if not pref_path.exists():
        if is_main_process():
            print("[WARN] preferences.jsonl not found. RLHF requires preference data.")
        return
    dtype = detect_dataset_type(str(pref_path))
    if dtype != "RLHF":
        raise RuntimeError(f"Expected RLHF preference dataset in {pref_path}, got {dtype}")

    pref_ds = PreferencePairDataset(str(pref_path), sp, max_len=config["max_seq_len"])
    sampler = DistributedSampler(pref_ds, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    pref_dl = DataLoader(pref_ds, batch_size=config["ppo_batch_size"], shuffle=(sampler is None),
                         sampler=sampler, collate_fn=lambda b: collate_pref(b, pad_id=pad_id),
                         num_workers=2, pin_memory=True)

    # train reward model
    rm_optim = torch.optim.AdamW(rm.parameters(), lr=config["rm_lr"], betas=config["betas"], weight_decay=config["weight_decay"])
    rm_steps = 0
    rm_progress = tqdm(total=config["rm_train_steps"], desc="rm-train", unit="step") if is_main_process() else None
    rm.train()
    while rm_steps < config["rm_train_steps"]:
        if isinstance(pref_dl.sampler, DistributedSampler):
            pref_dl.sampler.set_epoch(rm_steps)
        for prompts, pos_ids, neg_ids in pref_dl:
            pos_ids = pos_ids.to(device); neg_ids = neg_ids.to(device)
            r_pos = rm(pos_ids, pad_id=pad_id); r_neg = rm(neg_ids, pad_id=pad_id)
            loss = preference_loss(r_pos, r_neg)
            rm_optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
            rm_optim.step()
            rm_steps += 1
            if is_main_process() and rm_progress:
                rm_progress.update(1)
                rm_progress.set_postfix({"rm_loss": float(loss.item())})
            if rm_steps >= config["rm_train_steps"]:
                break
    if rm_progress: rm_progress.close()
    barrier()

    # PPO fine-tuning
    ppo_optim = torch.optim.AdamW(policy.parameters(), lr=config["ppo_lr"], betas=config["betas"], weight_decay=config["weight_decay"])
    ppo_progress = tqdm(total=config["total_steps"], desc="ppo-train", unit="step") if is_main_process() else None
    step = 0
    policy.train()
    while step < config["total_steps"]:
        if isinstance(pref_dl.sampler, DistributedSampler):
            pref_dl.sampler.set_epoch(step)
        for prompts, pos_ids, neg_ids in pref_dl:
            queries = prompts.to(device)
            with torch.no_grad():
                responses = generate_greedy(policy, queries, max_new_tokens=64, eos_id=sp.eos_id())
            stats = ppo_step(policy, old_policy, rm, queries, responses, ppo_optim,
                             kl_coef=config["ppo_kl_coef"], clip=config["ppo_clip"], pad_id=pad_id)
            step += 1

            # periodic sync of old_policy to current policy
            if step % 100 == 0:
                old_policy.load_state_dict(policy.state_dict())
                old_policy.eval()

            if is_main_process() and ppo_progress:
                ppo_progress.update(1)
                ppo_progress.set_postfix(stats)
            if step % config["save_every"] == 0 or step >= config["total_steps"]:
                if is_main_process():
                    ckpt = {"model": policy.state_dict(), "step": step}
                    atomic_overwrite(ckpt, os.path.join(save_dir, "model_ppo.pt"))
                    atomic_write_json({"ppo_step": step, "stats": stats}, os.path.join(save_dir, "metrics_ppo.json"))
            if step >= config["total_steps"]:
                break
        if step >= config["total_steps"]:
            break
    if ppo_progress: ppo_progress.close()
    barrier()
    if is_main_process():
        print("[INFO] RLHF finished. Saved policy to model_ppo.pt")

# -------------------------
# Training driver (SFT + optional RLHF)
# -------------------------
def train(tokens_dir: str, save_dir: str, config: Dict[str, Any], local_rank: Optional[int] = None):
    setup_seed(config.get("seed", 42))
    rank, world_size, local_rank = ddp_init(local_rank)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs(save_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=str(Path(save_dir) / "tb")) if is_main_process() else None
    if config.get("use_wandb", False) and wandb is not None and is_main_process():
        wandb.init(project=config.get("wandb_project", "neurosama-style-train"),
                   name=config.get("wandb_run_name"), config=config)

    # tokenizer
    spm_model = None
    try:
        spm_model = next(Path(tokens_dir).glob("*.model"))
    except StopIteration:
        spm_model = None
    if spm_model is None or SentencePieceProcessor is None:
        raise RuntimeError("spm.model not found in tokens_dir or sentencepiece missing; run build_tokens and install sentencepiece")
    sp = SentencePieceProcessor(model_file=str(spm_model))
    resp_piece = sp.piece_to_id("[ASSISTANT]") if sp is not None else None
    if resp_piece == sp.unk_id():
        resp_piece = None
    vocab_size = sp.get_piece_size()
    pad_id = sp.pad_id() if hasattr(sp, "pad_id") else 0

    # dataset validation (optional files)
    train_jsonl = Path(tokens_dir) / "train.jsonl"
    pref_jsonl = Path(tokens_dir) / "preferences.jsonl"
    if train_jsonl.exists():
        dtype = detect_dataset_type(str(train_jsonl))
        if dtype not in ("SFT", "INSTRUCT"):
            raise RuntimeError(f"Expected SFT/INSTRUCT dataset in {train_jsonl}, got {dtype}")
    if config.get("rlhf_enabled", False) and pref_jsonl.exists():
        dtype = detect_dataset_type(str(pref_jsonl))
        if dtype != "RLHF":
            raise RuntimeError(f"Expected RLHF preference dataset in {pref_jsonl}, got {dtype}")

    # dataset loaders
    shard_ids_files = sorted(Path(tokens_dir).glob("train_ids_*.npy"))
    if shard_ids_files:
        shard_indices = [i for i in range(len(shard_ids_files)) if i % world_size == rank]
        train_ds = ShardedTokenDataset(tokens_dir, vocab_size, resp_piece, pad_id, shard_indices=shard_indices)
        train_dl = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=False,
                              collate_fn=lambda b: collate_tokens(b, pad_id),
                              num_workers=4, pin_memory=True)
    else:
        train_ids = Path(tokens_dir) / "train_ids.npy"
        train_offs = Path(tokens_dir) / "train_offs.npy"
        if not train_ids.exists() or not train_offs.exists():
            raise RuntimeError("train ids/off not found; run build_tokens")
        train_ds = TokenWindowDataset(str(train_ids), str(train_offs), vocab_size, resp_piece, pad_id)
        sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
        train_dl = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=(sampler is None),
                              sampler=sampler, collate_fn=lambda b: collate_tokens(b, pad_id),
                              num_workers=2, pin_memory=True)

    val_ids = Path(tokens_dir) / "val_ids.npy"
    val_offs = Path(tokens_dir) / "val_offs.npy"
    val_dl = None
    if val_ids.exists() and val_offs.exists():
        val_ds = TokenWindowDataset(str(val_ids), str(val_offs), vocab_size, resp_piece, pad_id)
        val_dl = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False,
                            collate_fn=lambda b: collate_tokens(b, pad_id),
                            num_workers=2, pin_memory=True)

    # model + DDP
    model = GPTSmall(
        vocab_size, config["d_model"], config["n_layers"], config["n_heads"],
        config["d_ff"], config["max_seq_len"], config["dropout"],
        use_checkpoint=config.get("use_checkpoint", False)
    ).to(device)
    model = ddp_wrap(model)

    # optimizer
    if bnb is not None:
        try:
            optim = bnb.optim.AdamW8bit(
                model.parameters(), lr=config["lr"],
                betas=config["betas"], weight_decay=config["weight_decay"]
            )
            if is_main_process(): print("[INFO] Using bitsandbytes AdamW8bit")
        except Exception:
            optim = torch.optim.AdamW(
                model.parameters(), lr=config["lr"],
                betas=config["betas"], weight_decay=config["weight_decay"]
            )
            if is_main_process(): print("[WARN] Falling back to AdamW")
    else:
        optim = torch.optim.AdamW(
            model.parameters(), lr=config["lr"],
            betas=config["betas"], weight_decay=config["weight_decay"]
        )

    sched = CosineWithWarmup(optim, config["warmup_steps"], config["total_steps"], base_lr=config["lr"])
    scaler = GradScaler(enabled=(device.type == "cuda"))
    ema = EMA(model.module if hasattr(model, "module") else model, decay=0.9999, device=device)

    # SFT loop
    sft_train_loop(model, train_dl, val_dl, device, config, save_dir, ema, optim, sched, scaler, sp)

    barrier()
    # Optional RLHF stage
    if config.get("rlhf_enabled", False):
        local_rank = int(os.environ.get("LOCAL_RANK", 0)) if torch.cuda.is_available() else None
        rlhf_train(tokens_dir, save_dir, config, local_rank)

    if writer and is_main_process():
        writer.close()
    if wandb and config.get("use_wandb", False) and is_main_process():
        wandb.finish()

# -------------------------
# CLI wiring
# -------------------------
def main():
    p = argparse.ArgumentParser(description="Neuro-sama style multi-turn GPT training (SFT + optional RLHF)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sc = sub.add_parser("convert_hf", help="Download & convert a Hugging Face dataset to JSONL")
    sc.add_argument("--dataset", default="daily_dialog", help="HF dataset name (e.g., daily_dialog, blended_skill_talk)")
    sc.add_argument("--out_dir", default=RAW_DATA_DIR)

    sp_ = sub.add_parser("preprocess", help="Deduplicate, split, and shard JSONL")
    sp_.add_argument("--input_dir", default=RAW_DATA_DIR)
    sp_.add_argument("--out_dir", default=PROCESSED_DIR)
    sp_.add_argument("--val_ratio", type=float, default=0.02)
    sp_.add_argument("--shard_size", type=int, default=200_000)

    sb = sub.add_parser("build_tokens", help="Tokenize to flat ids/offsets and train SentencePiece")
    sb.add_argument("--data_dir", default=PROCESSED_DIR, help="processed dir with train_shard_*.jsonl and val.jsonl")
    sb.add_argument("--save_dir", default=TOKENS_DIR, help="where to write tokens (.npy) and spm.model")
    sb.add_argument("--sp_prefix", default=SPM_PREFIX, help="prefix for spm (writes spm.model here)")
    sb.add_argument("--vocab_size", type=int, default=DEFAULT_VOCAB_SIZE)
    sb.add_argument("--max_seq_len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    sb.add_argument("--stride", type=int, default=DEFAULT_STRIDE)

    st = sub.add_parser("train", help="Train GPTSmall; use torchrun for DDP. Optional RLHF stage.")
    st.add_argument("--tokens_dir", default=TOKENS_DIR, help="dir containing train shards or single train_ids.npy and spm.model")
    st.add_argument("--save_dir", default=SAVE_DIR, help="where model.pt and metrics.json go")
    st.add_argument("--total_steps", type=int, default=DEFAULT["total_steps"])
    st.add_argument("--batch_size", type=int, default=DEFAULT["batch_size"])
    st.add_argument("--grad_accum_steps", type=int, default=DEFAULT["grad_accum_steps"])
    st.add_argument("--lr", type=float, default=DEFAULT["lr"])
    st.add_argument("--save_every", type=int, default=DEFAULT["save_every"])
    st.add_argument("--use_checkpoint", action="store_true")
    st.add_argument("--clip_grad", type=float, default=DEFAULT["clip_grad"])
    st.add_argument("--log_every", type=int, default=DEFAULT["log_every"])
    st.add_argument("--eval_every", type=int, default=DEFAULT["eval_every"])
    st.add_argument("--use_wandb", action="store_true")
    st.add_argument("--wandb_project", type=str, default=DEFAULT["wandb_project"])
    st.add_argument("--wandb_run_name", type=str, default=None)
    st.add_argument("--seed", type=int, default=DEFAULT["seed"])
    # RLHF flags
    st.add_argument("--rlhf_enabled", action="store_true")
    st.add_argument("--rm_lr", type=float, default=DEFAULT["rm_lr"])
    st.add_argument("--ppo_lr", type=float, default=DEFAULT["ppo_lr"])
    st.add_argument("--ppo_epochs", type=int, default=DEFAULT["ppo_epochs"])
    st.add_argument("--ppo_batch_size", type=int, default=DEFAULT["ppo_batch_size"])
    st.add_argument("--ppo_kl_coef", type=float, default=DEFAULT["ppo_kl_coef"])
    st.add_argument("--ppo_clip", type=float, default=DEFAULT["ppo_clip"])
    st.add_argument("--rm_train_steps", type=int, default=DEFAULT["rm_train_steps"])

    args = p.parse_args()
    if args.cmd == "convert_hf":
        convert_hf_dataset(args.dataset, args.out_dir)
    elif args.cmd == "preprocess":
        preprocess(args.input_dir, args.out_dir, val_ratio=args.val_ratio, shard_size=args.shard_size)
    elif args.cmd == "build_tokens":
        build_tokens(
            args.data_dir, args.save_dir, args.sp_prefix,
            args.vocab_size, args.max_seq_len, args.stride
        )
    elif args.cmd == "train":
        cfg = {
            "n_layers": DEFAULT["n_layers"],
            "n_heads": DEFAULT["n_heads"],
            "d_model": DEFAULT["d_model"],
            "d_ff": DEFAULT["d_ff"],
            "dropout": DEFAULT["dropout"],
            "max_seq_len": DEFAULT["max_seq_len"],
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "lr": args.lr,
            "weight_decay": DEFAULT["weight_decay"],
            "betas": DEFAULT["betas"],
            "warmup_steps": DEFAULT["warmup_steps"],
            "total_steps": args.total_steps,
            "save_every": args.save_every,
            "device": DEFAULT["device"],
            "vocab_size": DEFAULT["vocab_size"],
            "use_checkpoint": args.use_checkpoint,
            "seed": args.seed,
            "log_every": args.log_every,
            "eval_every": args.eval_every,
            "clip_grad": args.clip_grad,
            "use_wandb": args.use_wandb,
            "wandb_project": args.wandb_project,
            "wandb_run_name": args.wandb_run_name,
            # RLHF
            "rlhf_enabled": args.rlhf_enabled,
            "rm_lr": args.rm_lr,
            "ppo_lr": args.ppo_lr,
            "ppo_epochs": args.ppo_epochs,
            "ppo_batch_size": args.ppo_batch_size,
            "ppo_kl_coef": args.ppo_kl_coef,
            "ppo_clip": args.ppo_clip,
            "rm_train_steps": args.rm_train_steps,
        }
        local_rank = int(os.environ.get("LOCAL_RANK", 0)) if torch.cuda.is_available() else None
        train(args.tokens_dir, args.save_dir, cfg, local_rank)

if __name__ == "__main__":
    main()