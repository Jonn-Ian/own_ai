#!/usr/bin/env python3
import os, argparse, yaml, torch
from pathlib import Path
from torch.nn import functional as F
from sentencepiece import SentencePieceProcessor

# Import your GPTSmall model definition from training script
from train.scripts.gpt_3_small import GPTSmall

# -------------------------
# File paths (defaults)
# -------------------------
TOKENS_DIR   = r"H:\OWN_AI\Prayer\assets\tokens"
WEIGHTS_PATH = r"H:\OWN_AI\Prayer\assets\weights\model.pt"
PERSONA_PATH = r"H:\OWN_AI\Prayer\personality\persona.yml"

# -------------------------
# Persona loader
# -------------------------
def load_persona(path: str):
    with open(path, "r", encoding="utf-8") as f:
        persona = yaml.safe_load(f)
    name = persona.get("name", "Unknown")
    preamble = persona.get("preamble", "")
    style_rules = persona.get("style_rules", [])
    return name, preamble, style_rules

# -------------------------
# Sampling utilities
# -------------------------
def sample_next_token(logits, k=40, p=0.9, temperature=0.8):
    """Top-k + nucleus (top-p) sampling with temperature scaling."""
    logits = logits / max(temperature, 1e-6)
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Apply nucleus cutoff
    cutoff = cumulative_probs > p
    if cutoff.any():
        cutoff_index = cutoff.nonzero()[0, 1]
        sorted_logits = sorted_logits[:cutoff_index+1]
        sorted_indices = sorted_indices[:cutoff_index+1]

    probs = F.softmax(sorted_logits, dim=-1)
    idx = torch.multinomial(probs, num_samples=1)
    return sorted_indices[idx]

# -------------------------
# Model loader
# -------------------------
def load_model(weights_path, sp, device):
    ckpt = torch.load(weights_path, map_location=device)
    model = GPTSmall(
        vocab_size=sp.get_piece_size(),
        d_model=1024, n_layers=16, n_heads=16, d_ff=4096,
        max_seq_len=2048, dropout=0.1, use_checkpoint=False
    ).to(device)
    model.load_state_dict(ckpt["model"])
    return model

# -------------------------
# Response generator
# -------------------------
def generate_response(model, sp: SentencePieceProcessor, persona_preamble: str,
                      user_input: str, style_rules=None, max_len=200, device="cuda",
                      persona_name="Prayer", k=40, p=0.9, temperature=0.8):
    model.eval()
    with torch.no_grad():
        # Persona conditioning with style rules
        style_text = f"\nStyle: {', '.join(style_rules)}" if style_rules else ""
        prompt = f"{persona_preamble}{style_text}\n[USER] {user_input}\n[ASSISTANT]"
        ids = sp.encode(prompt, out_type=int)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)

        for _ in range(max_len):
            logits = model(input_ids)
            next_id = sample_next_token(logits[:, -1, :], k=k, p=p, temperature=temperature)
            input_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim=1)
            if next_id.item() == sp.eos_id() or input_ids.size(1) >= max_len:
                break

        out_ids = input_ids[0].tolist()
        text = sp.decode(out_ids)

        # Extract assistant response
        if "[ASSISTANT]" in text:
            resp = text.split("[ASSISTANT]", 1)[1].strip()
        else:
            resp = text

        # Optional debug logging
        if os.getenv("DEBUG_GEN") == "1":
            print("[DEBUG] Prompt:", prompt)
            print("[DEBUG] Tokens:", out_ids)

        # Speaker indicator toggle
        if os.getenv("SHOW_INDICATOR") == "1":
            return f"{persona_name}: {resp}"
        else:
            return resp

# -------------------------
# CLI loop
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Persona-conditioned generation loop")
    parser.add_argument("--tokens_dir", default=TOKENS_DIR)
    parser.add_argument("--weights", default=WEIGHTS_PATH)
    parser.add_argument("--persona", default=PERSONA_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    # Load persona
    name, preamble, style_rules = load_persona(args.persona)
    print(f"[INFO] Loaded persona: {name}")
    print(f"[INFO] Style rules: {style_rules}")

    # Load tokenizer
    try:
        spm_model = next(Path(args.tokens_dir).glob("*.model"))
    except StopIteration:
        raise FileNotFoundError(f"No SentencePiece model found in {args.tokens_dir}")
    sp = SentencePieceProcessor(model_file=str(spm_model))

    # Load model
    model = load_model(args.weights, sp, args.device)
    print("[INFO] Model loaded.")

    # Loop for interactive use
    print(f"[INFO] Starting generation loop with persona '{name}'...")
    while True:
        try:
            user_input = input("User: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"quit", "exit"}:
                break
            response = generate_response(model, sp, preamble, user_input,
                                         style_rules=style_rules, device=args.device,
                                         persona_name=name, max_len=args.max_len,
                                         k=args.top_k, p=args.top_p, temperature=args.temperature)
            print(response)
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()