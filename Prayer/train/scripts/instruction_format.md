# Supervised Fine‑Tuning (SFT) Dataset
``` python

    train.jsonl:
    {"conversation": "[USER] Hi!\n[ASSISTANT] Hello there, how are you?\n[USER] I'm good, just tired.\n[ASSISTANT] I hope you get some rest soon."}
    {"conversation": "[USER] What's your favorite game?\n[ASSISTANT] I really enjoy rhythm games, they keep me sharp."}
```

# 2. Reward Model Preference Dataset (for RLHF)

preferences.jsonl
``` python

    {"prompt": "Tell me a joke about cats.", "pos": "Why did the cat sit on the computer? To keep an eye on the mouse!", "neg": "Cats are animals."}
    {"prompt": "What's the capital of France?", "pos": "The capital of France is Paris.", "neg": "France is a country in Europe."}
```
