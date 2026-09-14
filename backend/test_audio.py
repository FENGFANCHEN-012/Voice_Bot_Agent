import av, wave, tempfile, os, numpy as np

wav_path = tempfile.mktemp(suffix=".wav")
with wave.open(wav_path, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(16000)
    samples = (np.sin(np.linspace(0, 440 * 2 * 3.14159, 16000 * 3)) * 16000).astype(np.int16)
    wf.writeframes(samples.tobytes())

webm_path = tempfile.mktemp(suffix=".webm")
out = av.open(webm_path, "w")
out_stream = out.add_stream("libopus", rate=16000)
out_stream.layout = "mono"
wav_in = av.open(wav_path)
for frame in wav_in.decode(audio=0):
    for packet in out_stream.encode(frame):
        out.mux(packet)
for packet in out_stream.encode(None):
    out.mux(packet)
wav_in.close()
out.close()

print(f"webm size: {os.path.getsize(webm_path)} bytes")

container = av.open(webm_path)
stream = next(s for s in container.streams if s.type == "audio")
print(f"stream codec: {stream.codec_context}")

resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)

for frame in container.decode(audio=0):
    print(f"frame: type={type(frame)}, format={frame.format}, layout={frame.layout}, rate={frame.rate}, samples={frame.samples}")
    out_frames = resampler.resample(frame)
    print(f"out_frames type: {type(out_frames)}, len: {len(out_frames)}")
    if isinstance(out_frames, list):
        for f in out_frames:
            print(f"  resampled: format={f.format}, samples={f.samples}, planes={len(f.planes)}")
            plane = f.planes[0]
            print(f"  plane type: {type(plane)}")
            print(f"  plane dir: {[x for x in dir(plane) if not x.startswith('_')]}")
container.close()

os.unlink(wav_path)
os.unlink(webm_path)
