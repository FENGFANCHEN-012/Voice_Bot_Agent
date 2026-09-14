import av, wave, tempfile, os, struct

webm_files = [f for f in os.listdir(tempfile.gettempdir()) if f.endswith('.webm')]
print(f"Found {len(webm_files)} webm files in temp")
for wf_path in webm_files[:3]:
    full = os.path.join(tempfile.gettempdir(), wf_path)
    fsize = os.path.getsize(full)
    print(f"\n--- {wf_path} ({fsize} bytes) ---")
    try:
        container = av.open(full)
        for s in container.streams:
            if s.type == "audio":
                print(f"  codec: {s.codec_context.name}, rate: {s.rate}, ch: {s.channels}, layout: {s.layout}")
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        total_samples = 0
        max_val = 0
        for frame in container.decode(audio=0):
            out = resampler.resample(frame)
            for f in (out if isinstance(out, list) else [out]):
                arr = f.to_ndarray().flatten()
                total_samples += len(arr)
                mx = max(abs(int(x)) for x in arr)
                if mx > max_val:
                    max_val = mx
        container.close()
        duration = total_samples / 16000.0
        print(f"  duration: {duration:.2f}s, max_amplitude: {max_val}")
    except Exception as e:
        print(f"  ERROR: {e}")
