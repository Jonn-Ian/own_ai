import discord
import asyncio
from TTS.api import TTS

# Path to save temporary audio files
TMP_PATH = "H:/OWN_AI/Core/Prayer/assets/voice/reply.wav"

# Load a Coqui TTS model once at startup
# You can change the model_name to another supported voice model
# Example: "tts_models/en/vctk/vits" (multi-speaker English)
# For voice cloning: "tts_models/multilingual/multi-dataset/yourtts"
tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False, gpu=True)

async def synthesize(text: str, out_path: str = TMP_PATH, speaker_wav: str = None):
    """
    Synthesize text to an audio file using Coqui TTS.
    If speaker_wav is provided, it will attempt voice cloning.
    """
    if speaker_wav:
        tts.tts_to_file(text=text, speaker_wav=speaker_wav, file_path=out_path)
    else:
        tts.tts_to_file(text=text, file_path=out_path)
    return out_path

async def speak_in_discord(text: str, voice_client: discord.VoiceClient,
                           tmpfile: str = TMP_PATH, speaker_wav: str = None):
    """
    Speak text in a Discord voice channel using Coqui TTS.
    Requires FFmpeg installed and available in PATH.
    """
    # Generate audio file
    await synthesize(text, out_path=tmpfile, speaker_wav=speaker_wav)

    # Stop any current playback
    if voice_client.is_playing():
        voice_client.stop()

    # Play the generated audio
    source = discord.FFmpegPCMAudio(tmpfile)
    voice_client.play(source)

    # Wait until finished
    while voice_client.is_playing():
        await asyncio.sleep(1)