import os
import json
import datetime
from typing import List, Dict, Optional, Generator

# -----------------------------
# Configurable file paths
# -----------------------------
MEMORY_FILE = r"H:\OWN_AI\Core\Prayer\memories\temp_mem.jsonl"   # rolling short-term memory (kept to N entries)
SUMMARY_FILE = r"H:\OWN_AI\Core\Prayer\memories\summary.txt"     # long-term summary (plain text)
LOGS_FILE = r"H:\OWN_AI\Core\Prayer\memories\conversation_logs.jsonl"  # append-only conversation logs (infinite)

# -----------------------------
# Helpers
# -----------------------------
def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def _now_iso() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

# -----------------------------
# Core API
# -----------------------------
def append_to_memory(speaker: str, text: str, max_lines: int = 3) -> None:
    """
    Append one memory entry.

    Behavior:
    - Appends the entry to the append-only LOGS_FILE (jsonl).
    - Updates the short-term MEMORY_FILE (jsonl) to keep only the last `max_lines` entries.
      MEMORY_FILE is used as a lightweight recent-context store and is trimmed to `max_lines`.
    - Both files store one JSON object per line with fields: speaker, text, timestamp.

    Note: LOGS_FILE is never truncated here (infinite growth).
    """
    _ensure_dir(MEMORY_FILE)
    _ensure_dir(LOGS_FILE)

    entry = {
        "speaker": speaker,
        "text": text,
        "timestamp": _now_iso()
    }

    # 1) Append to logs (infinite append)
    try:
        with open(LOGS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        # If logs append fails, proceed (we don't want to break the bot).
        pass

    # 2) Update rolling short-term memory (keep last max_lines entries)
    entries: List[Dict] = []
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        entries.append(json.loads(ln))
                    except Exception:
                        # ignore malformed lines
                        continue
        except Exception:
            entries = []

    entries.append(entry)
    # keep only the last max_lines items
    if max_lines is None or max_lines < 0:
        trimmed = entries
    else:
        trimmed = entries[-max_lines:]

    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            for e in trimmed:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    except Exception:
        # best-effort; ignore write errors to avoid crashing caller
        pass

def get_recent_memory(max_lines: int = 3) -> str:
    """
    Return the last `max_lines` memory entries as a JSONL string (one JSON object per line).

    This function is intended so you can append the returned JSONL directly into
    your training/finetuning input or pass it as 'context' to the model.

    Example return (string):
        {"speaker":"User","text":"hi","timestamp":"2025-..Z"}
        {"speaker":"Bot","text":"hello","timestamp":"2025-..Z"}

    If MEMORY_FILE does not exist, returns an empty string.
    """
    if not os.path.exists(MEMORY_FILE):
        return ""

    out_lines: List[str] = []
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if max_lines is None or max_lines < 0:
            selected = lines
        else:
            selected = lines[-max_lines:]
        out_lines = selected
    except Exception:
        return ""
    return "\n".join(out_lines)

def get_recent_memory_as_text(max_lines: int = 3) -> str:
    """
    Return the last `max_lines` memory entries rendered as simple human-readable lines:
        Speaker: text
    This is useful when you want readable context rather than raw JSONL.
    """
    if not os.path.exists(MEMORY_FILE):
        return ""
    items: List[str] = []
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        selected = lines[-max_lines:] if max_lines is not None and max_lines >= 0 else lines
        for ln in selected:
            try:
                obj = json.loads(ln)
                sp = obj.get("speaker", "")
                tx = obj.get("text", "")
                items.append(f"{sp}: {tx}")
            except Exception:
                # fallback to raw line
                items.append(ln)
    except Exception:
        return ""
    return "\n".join(items)

def load_summary() -> str:
    """Load the long-term summary (plain text)."""
    if os.path.exists(SUMMARY_FILE):
        try:
            with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""
    return ""

def save_summary(summary: str) -> None:
    """Save/overwrite the long-term summary (plain text)."""
    _ensure_dir(SUMMARY_FILE)
    try:
        with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
            f.write(summary)
    except Exception:
        pass

def clear_memory() -> None:
    """
    Clear the short-term MEMORY_FILE (temp memory). Does NOT clear the LOGS_FILE.
    Use clear_logs() to clear the append-only logs (not recommended).
    """
    try:
        if os.path.exists(MEMORY_FILE):
            open(MEMORY_FILE, "w", encoding="utf-8").close()
    except Exception:
        pass

def clear_logs() -> None:
    """
    Clear the append-only logs file. Note: this discards historical data and
    should be used with caution.
    """
    try:
        if os.path.exists(LOGS_FILE):
            open(LOGS_FILE, "w", encoding="utf-8").close()
    except Exception:
        pass

# -----------------------------
# Utilities for exporting / iterating logs
# -----------------------------
def iter_logs() -> Generator[Dict, None, None]:
    """
    Iterate over all log entries (yields dicts). This is a streaming reader that
    can be used to build training datasets from LOGS_FILE without loading entire file.
    """
    if not os.path.exists(LOGS_FILE):
        return
        yield  # make this a generator
    try:
        with open(LOGS_FILE, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    yield json.loads(ln)
                except Exception:
                    continue
    except Exception:
        return

def load_all_logs(limit: Optional[int] = None) -> List[Dict]:
    """
    Load up to `limit` log entries into memory as a list of dicts.
    If limit is None, loads entire file (use with caution).
    """
    out = []
    for i, obj in enumerate(iter_logs()):
        out.append(obj)
        if limit is not None and i + 1 >= limit:
            break
    return out

# -----------------------------
# Backwards-compat helpers (old API)
# -----------------------------
def append(speaker: str, text: str) -> None:
    """
    Backwards-compatible alias for append_to_memory.
    """
    append_to_memory(speaker, text, max_lines=3)

# -----------------------------
# End of module
# -----------------------------