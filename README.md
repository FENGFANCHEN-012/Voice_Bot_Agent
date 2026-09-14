# Voice_Bot_Agent — Singapore Tourist Helper Bot

A voice assistant for people visiting Singapore. You press the mic, ask something like
"will it rain in Bedok later?" or "what is there to do at Sentosa?", and it answers out loud.

Behind it is a small LangGraph agent that decides which tool to call — live weather and air
quality from data.gov.sg, Wikipedia for places and transport, and a clock for Singapore time —
then turns the reply back into speech.

## What it can do

- **Weather and environment (live, from data.gov.sg):** 2-hour forecast by area, air temperature,
  rainfall, wind direction, UV index, PSI and PM2.5.
- **Places:** looks up attractions and landmarks (Gardens by the Bay, Marina Bay Sands, etc.) on Wikipedia.
- **Transport:** background info on MRT lines, stations and bus services (also from Wikipedia — not live arrival times).
- **Time and events:** current Singapore time, public holidays and notable events.
- **Voice in, voice out:** speech is transcribed with Whisper and the reply is read back with Edge TTS.
  Chinese replies get a Chinese voice automatically; there are three voice styles (default, professional, friendly).
- **Remembers the conversation** within a session, so follow-ups like "what about tomorrow?" work.

## How a request flows

```
Browser mic (webm/opus)
   │  POST /voice/send
   ▼
PyAV → 16 kHz mono WAV → faster-whisper (large-v3)   ── speech to text
   ▼
guardrails: clean input, reject empty / too-long text, tag an intent
   ▼
LangGraph ReAct agent (Ollama chat model) ── picks and calls tools
   │        ├─ data.gov.sg real-time APIs
   │        ├─ Wikipedia search + summary
   │        └─ Singapore clock
   ▼
strip emoji / markdown → split into ≤180-char sentences → Edge TTS → join with ffmpeg
   ▼
JSON { message, audio_url }  → browser plays the MP3
```

There is also a text endpoint (`POST /chat`) that skips the speech-to-text step.

## Tech stack

| Part | What I used |
| --- | --- |
| Backend API | Python, FastAPI, Uvicorn |
| Agent | LangGraph (`create_react_agent`), LangChain tools, `MemorySaver` for per-session memory |
| LLM | Ollama via `langchain-ollama` (model set by `VOICE_AGENT_MODEL`) |
| Speech to text | faster-whisper `large-v3` (CUDA float16 if a GPU is found, otherwise CPU int8), PyAV for decoding |
| Text to speech | edge-tts, ffmpeg to join multi-part audio |
| Data sources | data.gov.sg real-time API, Wikipedia REST + search API |
| Frontend | Single HTML page with vanilla JS and the MediaRecorder API |
| Tests | pytest |

## Project layout

```
backend/
  app.py              FastAPI routes: /, /chat, /voice/send, /health
  agent_service.py    builds the LangGraph agent, runs guardrails, stores turns
  tools.py            the 12 tools the agent can call
  Model_handler.py    Whisper transcription + TTS (text cleanup, chunking, voice pick)
  guardrails.py       input cleaning, length check, keyword intent tagging
  session_store.py    thread-safe conversation history (in memory, optional JSON file)
  config.py           settings read from environment variables
  test_agent_upgrade.py  unit tests for text cleanup, chunking, guardrails, language detection
resources/
  page/home.html      the web UI
  asset/              background image
```

## Running it locally

You need Python 3.11+, [Ollama](https://ollama.com) running, and `ffmpeg` on your PATH
(only used when a reply is long enough to be split into several audio parts).

```bash
python -m venv venv
venv\Scripts\activate            # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt

cd backend
uvicorn app:app --reload --port 8000
```

Open http://localhost:8000 and allow microphone access.

The first start downloads the Whisper `large-v3` weights (about 3 GB) and is slow on CPU.
For a quicker test, change `"large-v3"` to `"small"` in `Model_handler.py`.

Settings (all optional):

| Variable | Default | Meaning |
| --- | --- | --- |
| `VOICE_AGENT_MODEL` | `minimax-m3:cloud` | Ollama model name |
| `VOICE_AGENT_TEMPERATURE` | `0.2` | LLM temperature |
| `VOICE_AGENT_MAX_HISTORY` | `8` | how many past turns to keep in context |

Run the tests from `backend/`:

```bash
pytest test_agent_upgrade.py
```

## What I learned

- **Tool descriptions matter as much as the tool code.** The agent picks a tool, and fills in its arguments,
  from the docstring alone. That's why each tool lists real example area names in its `Args` section —
  the model needs to know what a valid input looks like.
- **Public APIs are not consistent.** Every data.gov.sg endpoint shapes its JSON differently
  (`items` for forecasts and PSI, `readings` with station IDs for temperature and rainfall, `records` for UV),
  so each tool needed its own parsing. When an area isn't found, the tool returns the list of valid names
  so the model can correct itself instead of failing.
- **Browser audio isn't what Whisper wants.** The browser records WebM/Opus; Whisper works best on
  16 kHz mono, so the audio is decoded and resampled with PyAV before transcription.
- **LLM text isn't speech-ready.** Models like to answer with markdown and emoji, and TTS will try to read
  those out, so replies are cleaned first. Long replies are split by sentence, synthesised in parts and
  joined with ffmpeg.
- **Keeping memory per user.** Passing the session ID as LangGraph's `thread_id` gives each user their own
  conversation without me managing the message list by hand.
- **Know the limits of your data.** Wikipedia is fine for "what is the Circle Line" but it can't say when
  the next bus comes. Live transport would need LTA DataMall, which is the next thing I'd add.

## Known limitations

- Conversation memory is in RAM, so it's gone when the server restarts.
- Transport answers are general info, not real-time.
- Generated MP3s in `backend/audio_output/` are never cleaned up.
- CORS is open (`*`) because this is a local demo.
