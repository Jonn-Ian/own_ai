import wave
import json
from vosk import Model, KaldiRecognizer

# Download a Vosk model and set the path.
# Example (English small): https://alphacephei.com/vosk/models
# Place it at: models/vosk-model-small-en-us-0.15
MODEL_PATH = "models/vosk-model-small-en-us-0.15"
_model = None

def _load_model():
    global _model
    if _model is None:
        _model = Model(MODEL_PATH)

def transcribe(path: str) -> str:
    """
    Transcribe an audio WAV file to text.
    Expects mono PCM WAV, but Vosk is tolerant to common formats.
    """
    _load_model()
    wf = wave.open(path, "rb")
    rec = KaldiRecognizer(_model, wf.getframerate())
    rec.SetWords(True)
    result_text = []

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            result_text.append(res.get("text", ""))

    final_res = json.loads(rec.FinalResult())
    result_text.append(final_res.get("text", ""))
    return " ".join(t.strip() for t in result_text if t.strip())