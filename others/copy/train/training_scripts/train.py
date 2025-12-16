import os
import math
import json
import random
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sentencepiece as spm
from torch.amp import GradScaler, autocast   # AMP API
# =========================================================
# Small GPT-3 style (~600M params) for Colab T4
# =========================================================
class Config:
    # Paths
    data_dir = "/content/drive/MyDrive/Colab/assets/txt"   # folder with .jsonl or .txt
    save_dir = "/content/drive/MyDrive/Colab/assets/weights"   # checkpoints + tokenizer
    sp_model_prefix = "/content/drive/MyDrive/Colab/assets/weights/spm"  # SentencePiece files

    # Checkpointing
    save_every = 2000      # validation + checkpoint interval
    latest_name = "latest.pt"
    best_name = "best.pt"

    # Tokenizer
    special_tokens = {"INST": "[INST]", "CTX": "[CTX]", "RESP": "[RESP]"}
    vocab_size = 32000
    character_coverage = 1.0
    model_type = "bpe"

    # Model (small GPT-3 style)
    n_layers = 16          # depth
    n_heads  = 16          # attention heads
    d_model  = 1024        # hidden size
    d_ff     = 4 * d_model
    dropout  = 0.1
    max_seq_len = 1024     # reduced context window for Colab T4
    batch_size = 1         # safe for 15GB GPU
    grad_accum_steps = 16  # effective batch size = 16
    grad_clip = 1.0

    # Optimizer
    lr = 2e-4              # slightly lower LR for stability
    weight_decay = 0.01
    betas = (0.9, 0.95)    # optimizer betas
    warmup_steps = 4000
    total_steps = 200000   # training horizon

    # Extras
    use_amp = True         # mixed precision
    use_ema = True
    ema_decay = 0.999
    train_ratio = 0.98
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(Config.seed)

# =========================================================
# SentencePiece Encoder
# =========================================================
class SubwordEncoder:
    def __init__(self, model_prefix: str, vocab_size: int, special_tokens: Dict[str, str]):
        self.model_prefix = model_prefix
        self.model_path = f"{model_prefix}.model"
        self.vocab_size_requested = vocab_size
        self.special_tokens = special_tokens
        self.sp = None
        if os.path.exists(self.model_path):
            self.sp = spm.SentencePieceProcessor(model_file=self.model_path)

    def train(self, input_files: List[str]):
        os.makedirs(os.path.dirname(self.model_prefix), exist_ok=True)
        input_arg = ",".join(input_files)
        user_defined = ",".join(self.special_tokens.values())
        spm.SentencePieceTrainer.train(
            input=input_arg,
            model_prefix=self.model_prefix,
            vocab_size=self.vocab_size_requested,
            character_coverage=Config.character_coverage,
            model_type=Config.model_type,
            user_defined_symbols=user_defined,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            hard_vocab_limit=False
        )
        self.sp = spm.SentencePieceProcessor(model_file=self.model_path)

    def encode(self, text: str, add_bos_eos: bool = True) -> List[int]:
        ids = self.sp.encode(text, out_type=int)
        if add_bos_eos:
            ids = [self.sp.bos_id()] + ids + [self.sp.eos_id()]
        return ids

    def decode(self, ids: List[int]) -> str:
        return self.sp.decode(ids)

    @property
    def pad_id(self):
        return self.sp.pad_id()

    @property
    def bos_id(self):
        return self.sp.bos_id()

    @property
    def eos_id(self):
        return self.sp.eos_id()

    @property
    def vocab_size(self):
        return self.sp.get_piece_size()

# =========================================================
# Dataset
# =========================================================
def format_example(example: Dict[str, str], special_tokens: Dict[str, str]) -> str:
    inst = example.get("instruction", "").strip()
    inp = example.get("input", "").strip()
    out = example.get("output", "").strip()
    parts = [special_tokens["INST"] + " " + inst]
    if inp:
        parts.append(special_tokens["CTX"] + " " + inp)
    parts.append(special_tokens["RESP"] + " " + out)
    return "\n".join(parts)


class InstructionDataset(Dataset):
    def __init__(self, files: List[Path], encoder: SubwordEncoder, max_seq_len: int, special_tokens: Dict[str, str]):
        self.encoder = encoder
        self.max_seq_len = max_seq_len
        self.samples = []
        if encoder.sp is None:
            encoder.train([str(fp) for fp in files])
        for fp in files:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if fp.suffix == ".jsonl":
                        ex = json.loads(line)
                        text = format_example(ex, special_tokens)
                    else:
                        # If TXT, treat each line as an output-only sample
                        text = f"{Config.special_tokens['INST']} dialogue\n{Config.special_tokens['RESP']} {line}"
                    ids = self.encoder.encode(text, add_bos_eos=True)
                    # Chunk into max_seq_len windows for training
                    for i in range(0, len(ids) - 1, max_seq_len):
                        chunk = ids[i:i + max_seq_len]
                        if len(chunk) >= 2:
                            self.samples.append(chunk)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        seq = self.samples[idx]
        return torch.tensor(seq[:-1], dtype=torch.long), torch.tensor(seq[1:], dtype=torch.long)


def collate_batch(batch, pad_id: int):
    if pad_id is None or pad_id < 0:
        pad_id = 0
    xs, ys = zip(*batch)
    max_len = max(x.size(0) for x in xs)

    def pad_to(t, length):
        if t.size(0) < length:
            pad = torch.full((length - t.size(0),), pad_id, dtype=t.dtype)
            return torch.cat([t, pad])
        return t

    X = torch.stack([pad_to(x, max_len) for x in xs])
    Y = torch.stack([pad_to(y, max_len) for y in ys])
    return X, Y

# =========================================================
# GPT Model
# =========================================================
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout, max_seq_len):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("mask", mask[None, None, :, :])

    def forward(self, x):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(y))


class MLP(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, max_seq_len):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout, max_seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_seq_len, dropout):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, d_ff, dropout, max_seq_len) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        self.apply(self._init_weights)
        # Weight tying
        self.head.weight = self.tok_emb.weight

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.head(x)

# =========================================================
# Optimizer, scheduler, EMA
# =========================================================
class CosineWithWarmup:
    def __init__(self, optimizer, warmup_steps, total_steps, base_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.step_num = 0

    def step(self):
        self.step_num += 1
        if self.step_num < self.warmup_steps:
            lr_mult = self.step_num / max(1, self.warmup_steps)
        else:
            progress = (self.step_num - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr_mult = 0.5 * (1 + math.cos(math.pi * progress))
        lr = self.base_lr * lr_mult
        for group in self.optimizer.param_groups:
            group['lr'] = lr


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply(self, model: nn.Module):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

# =========================================================
# Loss and evaluation
# =========================================================
def compute_loss(logits, targets, pad_id: int):
    B, T, V = logits.size()
    ignore_idx = pad_id if pad_id is not None and pad_id >= 0 else 0
    return F.cross_entropy(logits.view(B * T, V), targets.view(B * T), ignore_index=ignore_idx)


@torch.no_grad()
def evaluate(model, loader, pad_id: int, device: str) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        logits = model(X)
        loss = compute_loss(logits, Y, pad_id)
        total_loss += loss.item()
        total_batches += 1
    avg_loss = total_loss / max(1, total_batches)
    ppl = math.exp(min(20, avg_loss))  # guard overflow
    return {"loss": avg_loss, "perplexity": ppl}

# =========================================================
# Checkpoint utilities
# =========================================================
def safe_save(path, state):
    torch.save(state, path)

def save_checkpoint(model, optim, scaler, ema, step, best_val_loss, sched):
    os.makedirs(Config.save_dir, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optim.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "ema": ema.shadow if ema is not None else None,
        "step": step,
        "best_val_loss": best_val_loss,
        "sched_step": sched.step_num if sched is not None else 0,
    }
    latest_path = os.path.join(Config.save_dir, Config.latest_name)
    safe_save(latest_path, state)
    print(f"[INFO] Saved checkpoint: {latest_path}")
    # Save best if improved (best_val_loss passed in reflects improvement)
    best_path = os.path.join(Config.save_dir, Config.best_name)
    safe_save(best_path, state)  # keep best in sync whenever best_val_loss changes

def load_checkpoint(path, model, optim, scaler, ema, sched, device: str):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if "optimizer" in ckpt and ckpt["optimizer"] is not None:
        optim.load_state_dict(ckpt["optimizer"])
    if "scaler" in ckpt and ckpt["scaler"] is not None and scaler is not None:
        scaler.load_state_dict(ckpt["scaler"])
    if ema is not None and "ema" in ckpt and ckpt["ema"] is not None:
        ema.shadow = ckpt["ema"]
    if sched is not None and "sched_step" in ckpt:
        sched.step_num = ckpt["sched_step"]
    step = ckpt.get("step", 0)
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    return step, best_val_loss

def latest_checkpoint_path(save_dir: str, latest_name: str):
    latest_path = os.path.join(save_dir, latest_name)
    if os.path.exists(latest_path):
        return latest_path
    return None

# =========================================================
# Input sanitization
# =========================================================
def sanitize_indices(t: torch.Tensor, vocab_size: int, pad_id: int):
    t = t.clone()
    safe_pad = pad_id if pad_id is not None and pad_id >= 0 else 0
    t[t < 0] = safe_pad
    t[t >= vocab_size] = safe_pad
    return t

# =========================================================
# Training loop
# =========================================================
def train():
    os.makedirs(Config.save_dir, exist_ok=True)

    files = [Path(Config.data_dir) / fn for fn in os.listdir(Config.data_dir) if fn.endswith(".jsonl")]
    if len(files) == 0:
        files = [Path(Config.data_dir) / fn for fn in os.listdir(Config.data_dir) if fn.endswith(".txt")]
    assert len(files) > 0, "No .jsonl or .txt files found in data_dir."

    random.shuffle(files)
    split = int(len(files) * Config.train_ratio)
    train_files = files[:split]
    val_files = files[split:] if split < len(files) else files[:1]

    encoder = SubwordEncoder(Config.sp_model_prefix, Config.vocab_size, Config.special_tokens)
    train_ds = InstructionDataset(train_files, encoder, Config.max_seq_len, Config.special_tokens)
    val_ds = InstructionDataset(val_files, encoder, Config.max_seq_len, Config.special_tokens)

    vocab_size = encoder.vocab_size
    pad_id = encoder.pad_id
    if pad_id is None or pad_id < 0:
        print(f"[WARN] Tokenizer pad_id={pad_id}. Using 0 as pad_id fallback.")
        pad_id = 0

    print(f"Vocab size: {vocab_size}, PAD id: {pad_id}")

    train_dl = DataLoader(
        train_ds,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=lambda b: collate_batch(b, pad_id)
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=lambda b: collate_batch(b, pad_id)
    )

    model = GPT(
        vocab_size,
        Config.d_model,
        Config.n_layers,
        Config.n_heads,
        Config.d_ff,
        Config.max_seq_len,
        Config.dropout
    ).to(Config.device)

    optim = torch.optim.AdamW(model.parameters(), lr=Config.lr, betas=Config.betas, weight_decay=Config.weight_decay)
    sched = CosineWithWarmup(optim, Config.warmup_steps, Config.total_steps, base_lr=Config.lr)

    scaler = GradScaler("cuda", enabled=Config.use_amp and Config.device == "cuda")
    ema = EMA(model, decay=Config.ema_decay) if Config.use_ema else None

    step = 0
    best_val_loss = float("inf")
    ckpt_path = latest_checkpoint_path(Config.save_dir, Config.latest_name)

    progress = tqdm(total=Config.total_steps, desc="Training", unit="step")
    if ckpt_path:
        print(f"[INFO] Resuming from checkpoint: {ckpt_path}")
        step, best_val_loss = load_checkpoint(ckpt_path, model, optim, scaler, ema, sched, Config.device)
        progress.update(step)

    try:
        while step < Config.total_steps:
            model.train()
            for batch_idx, (X, Y) in enumerate(train_dl):
                X, Y = X.to(Config.device), Y.to(Config.device)

                X = sanitize_indices(X, vocab_size, pad_id)
                Y = sanitize_indices(Y, vocab_size, pad_id)

                with autocast("cuda", enabled=Config.use_amp and Config.device == "cuda"):
                    logits = model(X)
                    loss = compute_loss(logits, Y, pad_id)

                scaler.scale(loss).backward()

                if (batch_idx + 1) % Config.grad_accum_steps == 0:
                    scaler.unscale_(optim)
                    nn.utils.clip_grad_norm_(model.parameters(), Config.grad_clip)
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad(set_to_none=True)
                    sched.step()

                    if ema is not None:
                        ema.update(model)

                    step += 1
                    progress.set_postfix(loss=loss.item(), lr=optim.param_groups[0]['lr'])
                    progress.update(1)

                    # Validation + checkpoint saving
                    if step % Config.save_every == 0:
                        val_metrics = evaluate(model, val_dl, pad_id, Config.device)
                        print(f"\nStep {step}: train_loss={loss.item():.4f}, "
                              f"val_loss={val_metrics['loss']:.4f}, "
                              f"val_ppl={val_metrics['perplexity']:.2f}")

                        # Track best and save both latest and best
                        if val_metrics["loss"] < best_val_loss:
                            best_val_loss = val_metrics["loss"]
                            # Save EMA weights as best if enabled
                            if ema is not None:
                                ema.apply(model)
                                save_checkpoint(model, optim, scaler, ema, step, best_val_loss, sched)
                                ema.restore(model)
                            else:
                                save_checkpoint(model, optim, scaler, ema, step, best_val_loss, sched)
                        else:
                            # Save latest only
                            if ema is not None:
                                ema.apply(model)
                                save_checkpoint(model, optim, scaler, ema, step, best_val_loss, sched)
                                ema.restore(model)
                            else:
                                save_checkpoint(model, optim, scaler, ema, step, best_val_loss, sched)

                    if step >= Config.total_steps:
                        break

    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted. Last saved checkpoint remains valid.")

    progress.close()
    print("Training complete.")

# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    train()
