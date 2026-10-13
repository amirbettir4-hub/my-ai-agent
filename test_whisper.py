"""test_whisper.py — find which transcription provider works on this network."""
import io
import os
import wave
import base64
import requests
from dotenv import load_dotenv

load_dotenv()


def make_silent_wav(seconds=1, rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * rate * seconds)
    return buf.getvalue()


audio = make_silent_wav()
print(f"Test audio: {len(audio)} bytes\n")

results = []


def attempt(name, fn):
    print(f"=== {name} ===")
    try:
        out = fn()
        print(f"✅ WORKED: {repr(out)[:150]}\n")
        results.append((name, True, ""))
    except Exception as e:
        print(f"❌ {type(e).__name__}: {str(e)[:250]}\n")
        results.append((name, False, f"{type(e).__name__}: {str(e)[:120]}"))


# ── OpenRouter STT candidates (same key as chat) ──────────────
def or_stt(model_id):
    def _inner():
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        r = requests.post(
            "https://openrouter.ai/api/v1/audio/transcriptions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_id,
                "input_audio": {
                    "data": base64.b64encode(audio).decode(),
                    "format": "wav",
                },
            },
            timeout=60,
        )
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        return r.json().get("text", "")[:80]
    return _inner


# ── Local faster-whisper (CPU, no API) ────────────────────────
def local_faster_whisper():
    from faster_whisper import WhisperModel
    import tempfile
    # Write our silent wav to disk so faster-whisper can read it
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio)
        path = f.name
    try:
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(path, beam_size=1)
        text = " ".join(s.text for s in segments)
        return f"(local) {text[:80]!r}"
    finally:
        os.unlink(path)


# ── Legacy providers (kept for completeness) ──────────────────
def groq_test():
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return client.audio.transcriptions.create(
        file=("test.wav", audio),
        model="whisper-large-v3-turbo",
        response_format="text",
    )


# Run all tests
attempt("OpenRouter STT — x-ai/grok-stt-1.0", or_stt("x-ai/grok-stt-1.0"))
attempt("OpenRouter STT — openai/whisper-large-v3", or_stt("openai/whisper-large-v3"))
attempt("OpenRouter STT — openai/whisper-1", or_stt("openai/whisper-1"))
attempt("Local faster-whisper (tiny model)", local_faster_whisper)
attempt("Groq whisper-large-v3-turbo", groq_test)


print("=" * 62)
print("SUMMARY")
print("=" * 62)
for name, ok, err in results:
    print(f"{'✅' if ok else '❌'}  {name}")
    if err:
        print(f"      {err}")
working = [n for n, ok, _ in results if ok]
print()
if working:
    print(f"Working: {len(working)} provider(s)")
    print("Best pick → use the first one that says ✅")
else:
    print("Nothing works yet — need to install faster-whisper manually.")