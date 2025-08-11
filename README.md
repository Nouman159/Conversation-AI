# Conversational AI (AssemblyAI + Groq + ElevenLabs)

A minimal, real-time **voice conversation** loop:
- **Speech-to-Text**: AssemblyAI streaming transcribes your microphone.
- **LLM Response**: Transcript goes to Groq (e.g., `llama-3.3-70b-versatile`) for a reply.
- **Text-to-Speech**: ElevenLabs synthesizes the reply to audio and plays it.

## Features
- Non-blocking audio playback (background worker) so the stream stays connected.
- Uses `.env` via `python-dotenv` to manage secrets.
- Saves synthesized audio as `reply.wav` (16k mono PCM wrapped in WAV).

## Prerequisites
- **Python 3.10+**
- **PortAudio** (for mic input / playback):
  - macOS: `brew install portaudio`
  - Debian/Ubuntu: `sudo apt install portaudio19-dev`

## Install
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
