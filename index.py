import os
import logging
import wave
from typing import Type
from threading import Thread, Event
from queue import Queue, Empty

import simpleaudio as sa
from elevenlabs import ElevenLabs

import assemblyai as aai
from assemblyai.streaming.v3 import (
    BeginEvent, StreamingClient, StreamingClientOptions, StreamingError,
    StreamingEvents, StreamingParameters, StreamingSessionParameters,
    TerminationEvent, TurnEvent,
)
from groq import Groq

from dotenv import load_dotenv
load_dotenv()  


AAI_API_KEY = os.getenv("AAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EL_API_KEY   = os.getenv("EL_API_KEY")
EL_VOICE_ID  = os.getenv("EL_VOICE_ID")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("conv-ai")

groq_client = Groq(api_key=GROQ_API_KEY)
tts_client  = ElevenLabs(api_key=EL_API_KEY)

def pcm_to_wav(pcm_bytes: bytes, wav_path: str, sr: int = 16000, ch: int = 1):
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(ch)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        wf.writeframes(pcm_bytes)

def synth_and_play(text: str, voice_id: str = EL_VOICE_ID):
    if not (text and text.strip()):
        return
    stream = tts_client.text_to_speech.convert(
        voice_id=voice_id,
        output_format="pcm_16000",   # simpleaudio needs WAV/PCM (we'll wrap)
        text=text.strip(),
        model_id="eleven_multilingual_v2",
    )
    pcm = b"".join(stream)
    wav_path = "reply.wav"
    pcm_to_wav(pcm, wav_path, sr=16000, ch=1)
    sa.WaveObject.from_wave_file(wav_path).play().wait_done()

# -------- Background TTS worker (non-blocking) --------
tts_queue: Queue[str] = Queue()
stop_event = Event()

def tts_worker():
    while not stop_event.is_set():
        try:
            msg = tts_queue.get(timeout=0.2)
        except Empty:
            continue
        try:
            synth_and_play(msg)
        except Exception as e:
            print(f"[TTS error] {e}")
        finally:
            tts_queue.task_done()

# -------- LLM --------
def ask_groq(prompt: str) -> str:
    p = (prompt or "").strip()
    if not p:
        return ""
    r = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Answer briefly and clearly."},
            {"role": "user", "content": p},
        ],
        temperature=0.2,
    )
    return r.choices[0].message.content.strip()

# -------- AAI handlers --------
def on_begin(self: Type[StreamingClient], event: BeginEvent):
    print(f"Session started: {event.id}")

def on_turn(self: Type[StreamingClient], event: TurnEvent):
    print(f"[USER] {event.transcript}")
    if event.end_of_turn:
        if not event.turn_is_formatted:
            self.set_params(StreamingSessionParameters(format_turns=True))
        try:
            answer = ask_groq(event.transcript)
            if answer:
                print(f"[GROQ] {answer}")
                # enqueue for background TTS (returns immediately)
                tts_queue.put_nowait(answer)
        except Exception as e:
            print(f"[Groq error] {e}")

def on_terminated(self: Type[StreamingClient], event: TerminationEvent):
    print(f"Session terminated: {event.audio_duration_seconds:.2f}s processed")

def on_error(self: Type[StreamingClient], error: StreamingError):
    print(f"[AssemblyAI error] {error}")

# -------- MAIN --------
def main():
    # start TTS worker thread
    worker = Thread(target=tts_worker, daemon=True)
    worker.start()

    client = StreamingClient(
        StreamingClientOptions(api_key=AAI_API_KEY, api_host="streaming.assemblyai.com")
    )
    client.on(StreamingEvents.Begin, on_begin)
    client.on(StreamingEvents.Turn, on_turn)
    client.on(StreamingEvents.Termination, on_terminated)
    client.on(StreamingEvents.Error, on_error)

    client.connect(StreamingParameters(sample_rate=16000, format_turns=True))
    try:
        # this blocks until you Ctrl+C; no blocking work inside handlers now
        client.stream(aai.extras.MicrophoneStream(sample_rate=16000))
    finally:
        # allow worker to drain the queue, then stop
        tts_queue.join()
        stop_event.set()
        client.disconnect(terminate=True)

if __name__ == "__main__":
    main()
