import json
import random
from pathlib import Path

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_jsonl(items, path):
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def mix_datasets(config, out_file, total_size=100000):
    """
    config: list of dicts with {"path": "...", "weight": float}
    out_file: output JSONL path
    total_size: total number of samples to draw
    """
    datasets = []
    for entry in config:
        path = Path(entry["path"])
        weight = entry["weight"]
        data = load_jsonl(path)
        datasets.append({"data": data, "weight": weight})

    # Normalize weights
    total_weight = sum(d["weight"] for d in datasets)
    for d in datasets:
        d["prob"] = d["weight"] / total_weight

    mixed = []
    for _ in range(total_size):
        # pick dataset by weight
        choice = random.choices(datasets, weights=[d["prob"] for d in datasets])[0]
        sample = random.choice(choice["data"])
        mixed.append(sample)

    save_jsonl(mixed, out_file)
    print(f"[INFO] Mixed dataset saved to {out_file} with {len(mixed)} samples.")

if __name__ == "__main__":
    # Example config
    config = [
        {"path": "assets/processed/daily_dialog/train.jsonl", "weight": 0.5},
        {"path": "assets/processed/blended_skill_talk/train.jsonl", "weight": 0.3},
        {"path": "assets/processed/custom_chat/train.jsonl", "weight": 0.2},
    ]
    mix_datasets(config, "assets/processed/mixed_train.jsonl", total_size=200000)