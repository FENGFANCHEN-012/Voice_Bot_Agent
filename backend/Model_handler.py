import av
from faster_whisper import WhisperModel
import ctranslate2
import edge_tts
import os
import re
import tempfile
from pathlib import Path
from typing import List
from agent_service import agent_service

import numpy as np


TTS_VOICES = {
    "default": {
        "zh": "zh-CN-XiaoxiaoNeural",
        "en": "en-US-AriaNeural",
    },
    "professional": {
        "zh": "zh-CN-YunxiNeural",
        "en": "en-US-GuyNeural",
    },
    "friendly": {
        "zh": "zh-CN-XiaoyiNeural",
        "en": "en-US-JennyNeural",
    },
}

TTS_SETTINGS = {
    "rate": "+0%",
    "volume": "+0%",
    "pitch": "+0Hz",
}


# convert audio to wav format for WhisperModel, which requires 16kHz mono audio
def _audio_to_wav(input_path):


 
    container = av.open(input_path)
    stream = next(s for s in container.streams if s.type == "audio")
    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
    chunks = []
    for frame in container.decode(audio=0):
        out_frames = resampler.resample(frame)
        if not isinstance(out_frames, list):
            out_frames = [out_frames]
        for f in out_frames:
            arr = f.to_ndarray().flatten()
            chunks.append(arr)
    container.close()
    if not chunks:
        raise ValueError("No audio frames decoded")
    audio = np.concatenate(chunks).astype(np.int16)
    wav_path = tempfile.mktemp(suffix=".wav")
    import wave
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(audio.tobytes())
    return wav_path


def _device_config():
    try:
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"

device, compute = _device_config()
model = WhisperModel("large-v3", device=device, compute_type=compute)


def audiototext(audio_file):
    is_wav = audio_file.lower().endswith(".wav")
    path = audio_file
    try:
        if not is_wav:
            path = _audio_to_wav(audio_file)
    except Exception as e:
        print(f"[Audio] WAV conversion failed: {e}")
        return "I didn't catch that. Could you say it again?"

    try:
        segments, info = model.transcribe(path, condition_on_previous_text=False, no_speech_threshold=0.6)
        text = " ".join(s.text for s in segments).strip()
        if not text or len(text) < 3:
            return "I didn't catch that. Could you say it again?"
        print(f"[Audio] Transcribed: '{text}' (lang={info.language}, prob={info.language_probability:.2f})")
        return text
    except Exception as e:
        print(f"[Audio] Transcription failed: {e}")
        return "I didn't catch that. Could you say it again?"
    finally:
        if not is_wav and path and os.path.exists(path):
            try:
                os.unlink(path)
            except PermissionError:
                pass




def callLLM(text, session_id: str = "default"):
    if not text or not text.strip():
        return "I didn't catch that. Could you say it again?"
    result = agent_service.run_agent(text, session_id=session_id)
    return result.get("reply") or "I didn't catch that. Could you say it again?"


def processText(text):
    if text is None:
        return "I didn't catch that. Could you say it again?"

    text = str(text)
    text = re.sub(r"[\U00010000-\U0010ffff]", "", text)
    text = re.sub(r"[\u2600-\u27BF]", "", text)
    text = re.sub(r"[*#`_~]", "", text)
    text = re.sub(r"[#￥^&*]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text if text else "I didn't catch that. Could you say it again?"


def split_tts_text(text: str, max_chars: int = 180) -> List[str]:
    text = processText(text)
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    current = ""

    for sentence in sentences:
        if not sentence:
            continue
        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(sentence) <= max_chars:
                current = sentence
            else:
                words = sentence.split()
                temp = ""
                for word in words:
                    if len((temp + " " + word).strip()) <= max_chars:
                        temp = (temp + " " + word).strip()
                    else:
                        if temp:
                            chunks.append(temp)
                        temp = word
                current = temp
    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk]


async def text_to_speech(text, output_path, voice_profile: str = "default"):
    text = processText(text)
    if not text:
        return

    voice_map = TTS_VOICES.get(voice_profile, TTS_VOICES["default"])
    language_key = "zh" if is_chinese(text) else "en"
    voice_name = voice_map.get(language_key, voice_map["en"])

    chunks = split_tts_text(text)
    if len(chunks) == 1:
        communicate = edge_tts.Communicate(
            chunks[0],
            voice=voice_name,
            rate=TTS_SETTINGS["rate"],
            volume=TTS_SETTINGS["volume"],
            pitch=TTS_SETTINGS["pitch"],
        )
        await communicate.save(output_path)
        return

    temp_files = []
    try:
        for idx, chunk in enumerate(chunks):
            chunk_path = f"{output_path}.{idx}.mp3"
            communicate = edge_tts.Communicate(
                chunk,
                voice=voice_name,
                rate=TTS_SETTINGS["rate"],
                volume=TTS_SETTINGS["volume"],
                pitch=TTS_SETTINGS["pitch"],
            )
            await communicate.save(chunk_path)
            temp_files.append(chunk_path)

        if temp_files:
            import subprocess
            import os
            final_path = Path(output_path)
            final_path.parent.mkdir(parents=True, exist_ok=True)
            if len(temp_files) == 1:
                os.replace(temp_files[0], str(final_path))
            else:
                concat_list = "|".join(f"file '{path}'" for path in temp_files)
                subprocess.run([
                    "ffmpeg", "-y", "-i", "concat:" + "|".join(temp_files), "-acodec", "libmp3lame", str(final_path)
                ], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for temp_path in temp_files:
                try:
                    os.remove(temp_path)
                except FileNotFoundError:
                    pass
    finally:
        for temp_path in temp_files:
            try:
                os.remove(temp_path)
            except FileNotFoundError:
                pass


def is_chinese(word):
    return bool(re.search(r"[\u4e00-\u9fff]", word))

    