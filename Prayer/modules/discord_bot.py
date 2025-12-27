#!/usr/bin/env python3
import os
import sys
import asyncio
import discord
from discord.ext import commands, tasks
from dotenv import load_dotenv
from pathlib import Path
import torch

# Local modules (same folder)
from llm import generate_response, load_persona, SentencePieceProcessor, GPTSmall
from stt import transcribe
from tts import speak_in_discord

# -------------------------
# Load environment variables
# -------------------------
load_dotenv(r"H:\OWN_AI\Prayer\assets\.env")
TOKEN = os.getenv("DISCORD_TOKEN")

# -------------------------
# Config
# -------------------------
ASSIGNED_USER_ID = 123123123123  # replace with your actual Discord user ID
IDLE_TIMEOUT = 180  # 3 minutes

# -------------------------
# Bot setup
# -------------------------
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
intents.guilds = True
intents.voice_states = True
bot = commands.Bot(command_prefix="!", intents=intents)

# -------------------------
# State
# -------------------------
current_user = None
message_queue = asyncio.Queue()
last_activity = None

# -------------------------
# Background task
# -------------------------
@tasks.loop(seconds=10)
async def idle_check():
    global last_activity
    now = asyncio.get_event_loop().time()
    if last_activity and (now - last_activity > IDLE_TIMEOUT):
        channel = current_user_channel()
        if channel:
            try:
                response = generate_response(model, sp, persona_preamble, "Tell me something random", device=device)
                await channel.send(response)
                if channel.guild.voice_client:
                    await speak_in_discord(response, channel.guild.voice_client)
            except Exception as e:
                print(f"[ERROR] Idle response failed: {e}")
        last_activity = now

def current_user_channel():
    for guild in bot.guilds:
        for channel in guild.text_channels:
            if channel.permissions_for(guild.me).send_messages:
                return channel
    return None

# -------------------------
# Events
# -------------------------
@bot.event
async def on_ready():
    print(f"[INFO] Logged in as {bot.user}")
    idle_check.start()

@bot.event
async def on_message(message):
    global current_user, last_activity

    if message.author.bot:
        return

    last_activity = asyncio.get_event_loop().time()

    # Handle audio attachments (STT)
    if message.attachments:
        for attachment in message.attachments:
            if attachment.filename.lower().endswith(".wav"):
                tmp_path = f"temp_{message.id}.wav"
                await attachment.save(tmp_path)
                try:
                    text = transcribe(tmp_path)
                    message.content = text
                finally:
                    os.remove(tmp_path)

    if current_user is None:
        if message.author.id == ASSIGNED_USER_ID:
            current_user = ASSIGNED_USER_ID
            await handle_message(message)
        else:
            await message_queue.put(message)
    else:
        if message.author.id == current_user:
            await handle_message(message)
        else:
            await message_queue.put(message)

async def handle_message(message):
    global current_user
    try:
        response = generate_response(model, sp, persona_preamble, message.content, device=device)
        await message.channel.send(response)

        if message.guild.voice_client:
            await speak_in_discord(response, message.guild.voice_client)
    except Exception as e:
        await message.channel.send(f"[ERROR] Failed to generate response: {e}")

    if not message_queue.empty():
        next_msg = await message_queue.get()
        current_user = next_msg.author.id
        await handle_message(next_msg)
    else:
        current_user = None

# -------------------------
# Commands (restricted to assigned user)
# -------------------------
@bot.command(name="quit")
async def shutdown(ctx):
    if ctx.author.id == ASSIGNED_USER_ID:
        await ctx.send("Shutting down...")
        await bot.close()

@bot.command(name="join")
async def join(ctx):
    if ctx.author.id == ASSIGNED_USER_ID and ctx.author.voice:
        channel = ctx.author.voice.channel
        await channel.connect()

@bot.command(name="leave")
async def leave(ctx):
    if ctx.author.id == ASSIGNED_USER_ID and ctx.voice_client:
        await ctx.voice_client.disconnect()

@bot.command(name="refresh")
async def refresh(ctx):
    if ctx.author.id == ASSIGNED_USER_ID:
        await ctx.send("Rebooting...")
        await bot.close()
        os.execv(sys.executable, ['python'] + sys.argv)

# -------------------------
# Model + Persona init
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
spm_model = next(Path("H:/OWN_AI/Prayer/assets/tokens").glob("*.model"))
sp = SentencePieceProcessor(model_file=str(spm_model))

ckpt = torch.load("H:/OWN_AI/Prayer/assets/weights/model.pt", map_location=device)
model = GPTSmall(
    vocab_size=sp.get_piece_size(),
    d_model=1024, n_layers=16, n_heads=16, d_ff=4096,
    max_seq_len=2048, dropout=0.1, use_checkpoint=False
).to(device)
model.load_state_dict(ckpt["model"])

_, persona_preamble, _ = load_persona(r"H:\OWN_AI\Prayer\personality\persona.yml")

# -------------------------
# Run
# -------------------------
bot.run(TOKEN)