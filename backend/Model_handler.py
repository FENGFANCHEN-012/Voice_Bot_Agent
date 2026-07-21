from faster_whisper import WhisperModel
from ollama import chat
import ctranslate2
import edge_tts
import os
import re
from agent import run_agent

def _device_config():
    try:
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def audiototext(audio_file):
    device, compute = _device_config()
    model = WhisperModel("large-v3", device=device, compute_type=compute)
    segments, info = model.transcribe(audio_file)
    text = ""
    for segment in segments:
        text += segment.text + " "
    return text.strip()




def callLLM(text):
    if not text or not text.strip():
        return "I didn't catch that. Could you say it again?"
    return run_agent(text)


# exclude emoji preventing TTS from working, and return a default message if the text is empty after cleaning
def processText(text):

  emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "]+",
  flags=re.UNICODE,
)

  text = re.sub(
        r"[\U00010000-\U0010ffff]",
        "",
        text
    )
  text = re.sub(r"[*#`_]", "", text)

  text = re.sub(r"[#￥^&*]", "", text)
  
  return text.strip() if text.strip() else "I didn't catch that. Could you say it again?"



async def text_to_speech(text, output_path):
    text = processText(text)
    if(is_chinese(text)):

     communicate = edge_tts.Communicate(text, voice="zh-CN-XiaoxiaoNeural")

    else:
        communicate = edge_tts.Communicate(text, voice="en-US-AriaNeural")

    await communicate.save(output_path)



def is_chinese(word):

    return bool(
        re.search(r'[\u4e00-\u9fff]', word)
    )

    