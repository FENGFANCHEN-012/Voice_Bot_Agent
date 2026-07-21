import os
import sys
import uuid
import traceback
import tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(__file__))
from Model_handler import audiototext, callLLM, text_to_speech

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

HERE = os.path.dirname(__file__)
AUDIO_DIR = os.path.join(HERE, "audio_output")
os.makedirs(AUDIO_DIR, exist_ok=True)
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

ASSET_DIR = os.path.join(HERE, "..", "resources", "asset")
app.mount("/asset", StaticFiles(directory=ASSET_DIR), name="asset")


FAVICON_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' "
    "fill='none' stroke='%2364e2ff' stroke-width='2' "
    "stroke-linecap='round' stroke-linejoin='round'>"
    "<path d='M12 3v6'/><path d='M8.5 7.5a5 5 0 1 0 7 0'/>"
    "<path d='M6 15c1.5-2 3.7-3 6-3s4.5 1 6 3'/></svg>"
)


@app.get("/favicon.ico")
async def favicon():
    return Response(content=FAVICON_SVG, media_type="image/svg+xml")


@app.get("/")
async def index():
    return FileResponse(os.path.join(HERE, "..", "resources", "page", "home.html"))


@app.get("/ping")
async def ping():
    return {"status": "ok"}


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        reply = callLLM(req.message)
        audio_filename = f"{uuid.uuid4()}.mp3"
        audio_path = os.path.join(AUDIO_DIR, audio_filename)
        await text_to_speech(reply, audio_path)
        return {"response": reply or "(no response)", "audio_url": f"/audio/{audio_filename}"}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/voice/send")
async def sendVoice(audio: UploadFile = File(...)):
    tmp = None
    try:
        content = await audio.read()
        suffix = os.path.splitext(audio.filename or "audio.webm")[1] or ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(content)
            tmp = f.name
        text = audiototext(tmp)
        print(f"[User said]: {text}")
        reply = callLLM(text)
        audio_filename = f"{uuid.uuid4()}.mp3"
        audio_path = os.path.join(AUDIO_DIR, audio_filename)
        await text_to_speech(reply, audio_path)
        return {"message": reply or "(no response)", "audio_url": f"/audio/{audio_filename}"}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)
