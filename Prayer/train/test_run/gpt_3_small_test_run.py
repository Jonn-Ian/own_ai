import torch
import torch.nn.functional as F
import sentencepiece as spm
import argparse
from pathlib import Path
import json

from train_gpt3_small import GPTSmall  # import your model class

def load_checkpoint(model_path, vocab_size, d_model, n_layers, n_heads, d_ff, max_seq_len, dropout):
    ckpt = torch.load(model_path, map_location="cpu")
    model = GPTSmall(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=dropout,
        use_checkpoint=False
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model

def sample(model, sp, prompt, max_new_tokens=100, temperature=1.0, top_k=50, device="cpu"):
    ids = sp.encode(prompt, out_type=int)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        logits = model(x)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, ix = torch.topk(logits, top_k)
            mask = torch.ones_like(logits) * float("-inf")
            mask.scatter_(1, ix, v)
            logits = mask
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)
        if next_id.item() == sp.eos_id():
            break
    return sp.decode(x[0].tolist())

def main():
    parser = argparse.ArgumentParser(description="Inference with GPT-3-small-ish model")
    parser.add_argument("--tokens_dir", default="drive/MyDrive/Colab/assets/weights/spm")
    parser.add_argument("--model_path", default="drive/MyDrive/Colab/assets/weights/model.pt")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    # load tokenizer
    sp_model = next(Path(args.tokens_dir).glob("*.model"))
    sp = spm.SentencePieceProcessor(model_file=str(sp_model))

    # load meta
    meta_path = Path(args.tokens_dir) / "meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # load model
    model = load_checkpoint(
        args.model_path,
        vocab_size=meta["vocab_size"],
        d_model=1024,
        n_layers=24,
        n_heads=16,
        d_ff=4096,
        max_seq_len=meta["max_seq_len"],
        dropout=0.1
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # generate
    output = sample(
        model, sp, args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device
    )
    print("=== Prompt ===")
    print(args.prompt)
    print("=== Output ===")
    print(output)

if __name__ == "__main__":
    main()