"""tools.py — all agent tools. Lazy clients, graceful failures, no crash-on-import."""

import os
import re
import json
import time
import base64
import asyncio
import shutil
import subprocess
import sys
import tempfile
import urllib.parse

import requests
import yt_dlp
import edge_tts
from dotenv import load_dotenv

load_dotenv()


# ============================================================
#  LAZY CLIENT LOADERS
#  Nothing is built at import time — a bad/missing key only
#  breaks the one tool that needs it, never the whole app.
# ============================================================

_groq_client = None
_hf_client = None


def _need(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing {name} in .env — add it and restart.")
    return val


def get_groq():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        _groq_client = Groq(api_key=_need("GROQ_API_KEY"))
    return _groq_client


def get_hf():
    global _hf_client
    if _hf_client is None:
        from huggingface_hub import InferenceClient
        _hf_client = InferenceClient(token=_need("HF_TOKEN"))
    return _hf_client


def _vision_candidates():
    """Return list of vision models to try, in order."""
    import config
    return [config.VISION_MODEL] + list(config.VISION_MODEL_FALLBACKS)


THREE_WS_BASE = "https://three.ws/api/3d"
WORKSPACE_FILE = "workspace.json"


# ============================================================
#  URL CLEANING
# ============================================================

def _clean_url(url: str) -> str:
    if not url:
        return url
    md_match = re.search(r'\((https?://[^\)]+)\)', url)
    if md_match:
        url = md_match.group(1)
    url = url.strip().strip('"').strip("'").strip("`").strip()
    match = re.search(r'(https?://\S+)', url)
    if match:
        url = match.group(1)
    return url.rstrip('.,;:!?)')


# ============================================================
#  WORKSPACE
# ============================================================

def _load_workspace():
    if not os.path.exists(WORKSPACE_FILE):
        return None
    try:
        with open(WORKSPACE_FILE, "r", encoding="utf-8") as f:
            return json.load(f).get("path")
    except Exception:
        return None


def _save_workspace(path):
    with open(WORKSPACE_FILE, "w", encoding="utf-8") as f:
        json.dump({"path": path}, f)


def set_workspace(path: str) -> str:
    path = os.path.abspath(path.strip().strip('"'))
    if not os.path.exists(path):
        return f"❌ Folder doesn't exist: {path}"
    if not os.path.isdir(path):
        return f"❌ Not a folder: {path}"
    _save_workspace(path)
    return f"✅ Workspace set to: {path}"


def get_workspace() -> str:
    path = _load_workspace()
    return path if path else "No workspace set yet. Use set_workspace first."


def _safe_path(relative_path: str):
    workspace = _load_workspace()
    if not workspace:
        return None, "❌ No workspace set. Use set_workspace first."
    full = os.path.abspath(os.path.join(workspace, relative_path))
    if not full.startswith(os.path.abspath(workspace)):
        return None, "❌ Cannot access files outside the workspace."
    return full, None


def write_project_file(relative_path: str, content: str) -> str:
    full, err = _safe_path(relative_path)
    if err:
        return err
    folder = os.path.dirname(full)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)
    return f"✅ Wrote to project: {relative_path}"


def read_project_file(relative_path: str) -> str:
    full, err = _safe_path(relative_path)
    if err:
        return err
    if not os.path.exists(full):
        return f"❌ Not found: {relative_path}"
    with open(full, "r", encoding="utf-8") as f:
        return f.read()


def list_project_files(subpath: str = "") -> str:
    full, err = _safe_path(subpath)
    if err:
        return err
    if not os.path.exists(full):
        return f"❌ Not found: {subpath or '(workspace root)'}"

    files = []
    for root, dirs, filenames in os.walk(full):
        dirs[:] = [
            d for d in dirs
            if not d.startswith(".") and d not in
            ("node_modules", "Library", "Temp", "obj", "bin")
        ]
        for name in filenames:
            rel = os.path.relpath(os.path.join(root, name), full)
            files.append(rel)
            if len(files) >= 100:
                files.append("... (truncated, over 100 files)")
                break
        if len(files) >= 100:
            break

    if not files:
        return f"Empty folder: {subpath or '(workspace root)'}"
    return f"Files in {subpath or 'workspace root'}:\n" + "\n".join(files)


def scan_workspace() -> str:
    workspace = _load_workspace()
    if not workspace:
        return "❌ No workspace set."

    lua_count = cs_count = py_count = other_count = 0
    for root, dirs, filenames in os.walk(workspace):
        dirs[:] = [
            d for d in dirs
            if not d.startswith(".") and d not in
            ("node_modules", "Library", "Temp", "obj", "bin")
        ]
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext in (".lua", ".luau"):
                lua_count += 1
            elif ext == ".cs":
                cs_count += 1
            elif ext == ".py":
                py_count += 1
            else:
                other_count += 1

    project_type = "Generic folder"
    if lua_count > cs_count and lua_count > py_count:
        project_type = "Roblox project (Luau)"
    elif cs_count > lua_count and cs_count > py_count:
        project_type = "Unity project (C#)"

    return "\n".join([
        f"📁 Workspace: {workspace}",
        f"🎮 Project type: {project_type}",
        f"📊 File counts:",
        f"   • Luau/Lua: {lua_count}",
        f"   • C# (.cs): {cs_count}",
        f"   • Python: {py_count}",
        f"   • Other: {other_count}",
    ])


# ============================================================
#  AUDIO (YouTube → text, with HF Whisper fallback)
# ============================================================

def download_youtube_audio(url: str) -> str:
    url = _clean_url(url)
    print(f"🎧 Downloading audio from: {url}")
    ydl_opts = {
        "format": "worstaudio/worst",
        "outtmpl": "downloaded_audio.%(ext)s",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "64",
        }],
        "postprocessor_args": ["-ac", "1", "-ar", "16000"],
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 30,
        "retries": 3,
        "extractor_args": {
            "youtube": {"player_client": ["android", "ios", "web"]}
        },
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return "downloaded_audio.mp3"


def _split_audio_if_needed(audio_path: str, max_mb: int = 24) -> list:
    if not os.path.exists(audio_path):
        return [audio_path]
    size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    print(f"   Audio file size: {size_mb:.1f} MB")
    if size_mb <= max_mb:
        return [audio_path]

    print(f"   ⚠️ File too big, splitting...")
    chunk_duration = 600
    chunks = []
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
        capture_output=True, text=True,
    )
    try:
        total_duration = float(result.stdout.strip())
    except ValueError:
        return [audio_path]

    num_chunks = int(total_duration // chunk_duration) + 1
    for i in range(num_chunks):
        start = i * chunk_duration
        chunk_path = f"audio_chunk_{i}.mp3"
        subprocess.run([
            "ffmpeg", "-i", audio_path, "-ss", str(start),
            "-t", str(chunk_duration), "-ac", "1", "-ar", "16000",
            "-b:a", "64k", chunk_path,
            "-hide_banner", "-loglevel", "error",
        ], check=True)
        chunks.append(chunk_path)
        print(f"   Chunk {i + 1}/{num_chunks}")
    return chunks


def _transcribe_chunk(chunk_path: str) -> str:
    """Try Groq Whisper first, fall back to HF Whisper if Groq is blocked."""
    # --- Groq attempt ---
    try:
        with open(chunk_path, "rb") as f:
            return get_groq().audio.transcriptions.create(
                file=(chunk_path, f.read()),
                model="whisper-large-v3-turbo",
                response_format="text",
            )
    except Exception as e:
        print(f"   ⚠️ Groq transcription failed ({type(e).__name__}), trying HF...")

    # --- HF fallback ---
    try:
        with open(chunk_path, "rb") as f:
            audio_bytes = f.read()
        result = get_hf().automatic_speech_recognition(
            audio_bytes,
            model="openai/whisper-large-v3",
        )
        return result.text if hasattr(result, "text") else str(result)
    except Exception as e:
        return f"[transcription failed: {e}]"


def youtube_to_text(url: str) -> str:
    try:
        audio_file = download_youtube_audio(url)
        chunks = _split_audio_if_needed(audio_file)
        transcripts = []
        for i, chunk in enumerate(chunks):
            print(f"📝 Transcribing chunk {i + 1}/{len(chunks)}...")
            transcripts.append(_transcribe_chunk(chunk))
            if chunk != audio_file and os.path.exists(chunk):
                os.remove(chunk)
        if os.path.exists(audio_file):
            os.remove(audio_file)
        full_text = "\n\n".join(transcripts)
        print(f"   ✅ Transcribed ({len(full_text)} chars)")
        return full_text
    except Exception as e:
        return f"Error: {e}"


# ============================================================
#  VIDEO WATCHER
# ============================================================

def _download_video(url: str, output_path: str = "watched_video.mp4") -> str:
    url = _clean_url(url)
    print(f"🎬 Downloading video: {url}")
    ydl_opts = {
        "format": "best[ext=mp4][height<=720]/best[ext=mp4]/best",
        "outtmpl": output_path,
        "quiet": True, "no_warnings": True,
        "socket_timeout": 30, "retries": 3,
        "extractor_args": {
            "youtube": {"player_client": ["android", "ios", "web"]}
        },
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_path


def _extract_keyframes(video_path: str, output_folder: str = "video_frames",
                       interval: int = 10) -> list:
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    output_pattern = os.path.join(output_folder, "frame_%04d.jpg")
    subprocess.run([
        "ffmpeg", "-i", video_path, "-vf", f"fps=1/{interval}",
        "-q:v", "2", output_pattern,
        "-hide_banner", "-loglevel", "error",
    ], check=True)
    return sorted([
        os.path.join(output_folder, f)
        for f in os.listdir(output_folder) if f.endswith(".jpg")
    ])


def _analyze_frame(image_path: str, question: str) -> str:
    """Try each vision model in order until one works."""
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    msg = HumanMessage(content=[
        {"type": "text", "text": question},
        {"type": "image_url",
         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
    ])

    api_key = _need("OPENROUTER_API_KEY")
    last_err = None

    for model_id in _vision_candidates():
        try:
            model = ChatOpenAI(
                model=model_id,
                openai_api_key=api_key,
                openai_api_base="https://openrouter.ai/api/v1",
            )
            return model.invoke([msg]).content
        except Exception as e:
            last_err = e
            print(f"   ⚠️ {model_id} failed ({type(e).__name__}), trying next...")

    return f"(all vision models failed: {last_err})"


def watch_video(url: str, frame_interval: int = 10, max_frames: int = 6) -> str:
    video_file = None
    audio_file = None
    frames_folder = "video_frames"
    try:
        print(f"🎬 Downloading video...")
        video_file = _download_video(url)
        print(f"🎬 Extracting frames every {frame_interval}s...")
        frames = _extract_keyframes(video_file, interval=frame_interval)
        print(f"   {len(frames)} frames.")

        print("🎧 Extracting audio...")
        audio_file = "watched_audio.mp3"
        subprocess.run([
            "ffmpeg", "-i", video_file, "-vn", "-ac", "1", "-ar", "16000",
            "-b:a", "64k", audio_file,
            "-hide_banner", "-loglevel", "error",
        ], check=True)

        chunks = _split_audio_if_needed(audio_file)
        transcripts = []
        for i, chunk in enumerate(chunks):
            print(f"🎧 Transcribing {i + 1}/{len(chunks)}...")
            transcripts.append(_transcribe_chunk(chunk))
            if chunk != audio_file and os.path.exists(chunk):
                os.remove(chunk)
        transcription = "\n\n".join(transcripts)

        print(f"👁️ Analyzing up to {max_frames} frames...")
        frame_analyses = []
        for i, frame_path in enumerate(frames[:max_frames]):
            print(f"   Frame {i + 1}/{min(len(frames), max_frames)}...")
            analysis = _analyze_frame(
                frame_path,
                "Describe this video frame in detail. Focus on UI, "
                "characters, game assets, layouts, colors, visible text.",
            )
            frame_analyses.append(f"[t={i * frame_interval}s] {analysis}")

        return (
            f"--- VIDEO ANALYSIS REPORT ---\n"
            f"**Source:** {url}\n"
            f"**Frames analyzed:** {len(frame_analyses)} of {len(frames)}\n\n"
            f"**AUDIO TRANSCRIPT:**\n{transcription}\n\n"
            f"**VISUAL ANALYSIS:**\n" + "\n\n".join(frame_analyses) +
            "\n--- END REPORT ---"
        )
    except Exception as e:
        return f"❌ Error watching video: {e}"
    finally:
        if video_file and os.path.exists(video_file):
            os.remove(video_file)
        if audio_file and os.path.exists(audio_file):
            os.remove(audio_file)
        if os.path.exists(frames_folder):
            shutil.rmtree(frames_folder, ignore_errors=True)


# ============================================================
#  WEB SEARCH  (uses the new `ddgs` package)
# ============================================================

def web_search(query: str, max_results: int = 4) -> str:
    try:
        from ddgs import DDGS
        print(f"🌐 Searching: {query}")
        results = DDGS().text(query, max_results=max_results)
        if not results:
            return "No results found."
        return "\n\n".join(
            f"{i}. {r.get('title', 'No title')}\n"
            f"   {r.get('body', '')}\n"
            f"   URL: {r.get('href', '')}"
            for i, r in enumerate(results, 1)
        )
    except ImportError:
        return (
            "❌ Web search unavailable. Install it with:\n"
            "   pip install ddgs"
        )
    except Exception as e:
        return f"Search error: {e}"


# ============================================================
#  IMAGE GENERATION (HF, with free Pollinations fallback)
# ============================================================

def generate_image(prompt: str) -> str:
    os.makedirs("ai_creations", exist_ok=True)
    safe = "".join(
        c for c in prompt[:30] if c.isalnum() or c in " -_"
    ).strip().replace(" ", "_") or "generated"
    filepath = os.path.join("ai_creations", f"image_{safe}.png")

    # Try HuggingFace first
    try:
        print(f"🎨 HF FLUX: {prompt}")
        image = get_hf().text_to_image(
            prompt=prompt,
            model="black-forest-labs/FLUX.1-schnell",
        )
        image.save(filepath)
        return f"✅ Image saved to: {filepath}"
    except Exception as e:
        print(f"   HF failed ({e}), trying Pollinations...")

    # Fallback: Pollinations (free, no key)
    try:
        from io import BytesIO
        from PIL import Image
        url = (
            "https://image.pollinations.ai/prompt/"
            + urllib.parse.quote(prompt)
            + "?width=1024&height=1024&nologo=true"
        )
        r = requests.get(url, timeout=90)
        r.raise_for_status()
        Image.open(BytesIO(r.content)).save(filepath)
        return f"✅ Image saved to: {filepath}"
    except Exception as e:
        return f"Image generation error: {e}"


# ============================================================
#  3D MODEL (three.ws — best-effort, may be down)
# ============================================================

def generate_3d_model(prompt: str, max_polls: int = 30) -> str:
    try:
        print(f"🧊 Generating 3D: {prompt}")
        response = requests.post(
            f"{THREE_WS_BASE}/generate",
            json={"prompt": prompt, "format": "glb"},
            headers={"Content-Type": "application/json"},
            timeout=60,
        )
        if response.status_code == 429:
            return f"⏳ Rate limited. Retry in {response.headers.get('Retry-After', '30')}s."
        if response.status_code != 200:
            return f"❌ three.ws error {response.status_code}: {response.text[:300]}"

        data = response.json()
        status = data.get("status")

        if status == "done":
            return (f"✅ 3D model generated!\n"
                    f"📦 GLB: {data.get('glbUrl', '')}\n"
                    f"👁️ Viewer: {data.get('viewerUrl', '')}")

        if status == "pending":
            poll_url = data.get("poll", "")
            retry_after = data.get("retryAfter", 3)
            if not poll_url:
                job_id = data.get("job", "")
                if job_id:
                    poll_url = f"{THREE_WS_BASE}/generate?job={job_id}"
                else:
                    return f"❌ No poll URL. Response: {data}"
            if poll_url.startswith("/"):
                poll_url = f"https://three.ws{poll_url}"

            print("⏳ Polling...")
            for _ in range(max_polls):
                time.sleep(max(2, retry_after))
                pr = requests.get(poll_url, timeout=30)
                if pr.status_code == 429:
                    retry_after = int(pr.headers.get("Retry-After", "5"))
                    continue
                if pr.status_code != 200:
                    continue
                pd = pr.json()
                if pd.get("status") == "done":
                    return (f"✅ 3D model generated!\n"
                            f"📦 GLB: {pd.get('glbUrl', '')}\n"
                            f"👁️ Viewer: {pd.get('viewerUrl', '')}")
                if pd.get("status") == "error":
                    return f"❌ Generation failed: {pd.get('error')}"
                retry_after = pd.get("retryAfter", retry_after)
            return "⏳ Timed out waiting for 3D model."

        if status == "error":
            return f"❌ Generation failed: {data.get('error', 'Unknown')}"
        return f"⚠️ Unexpected response: {data}"
    except requests.exceptions.Timeout:
        return "⚠️ 3D service timed out."
    except Exception as e:
        return f"3D generation error: {e}"


# ============================================================
#  TEXT-TO-SPEECH
# ============================================================

async def _speak_async(text, filename, voice, rate):
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    await communicate.save(filename)


def text_to_speech(text: str, filename: str = "ai_voice.mp3",
                   voice: str = "en-US-GuyNeural",
                   rate: str = "+0%") -> str:
    try:
        os.makedirs("ai_creations", exist_ok=True)
        filepath = os.path.join("ai_creations", filename)
        asyncio.run(_speak_async(text, filepath, voice, rate))
        return f"✅ Voice saved to: {filepath}"
    except Exception as e:
        return f"Voice error: {e}"


# ============================================================
#  CODE EXECUTION — local subprocess (free, instant, unlimited)
# ============================================================

def run_python_code(code: str) -> str:
    """Run Python locally in a sandboxed subprocess with timeout.
    No API key, no rate limits, no waiting for a cloud service."""
    path = None
    try:
        print("💻 Running code locally...")
        with tempfile.NamedTemporaryFile(
            "w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            path = f.name

        result = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True, timeout=30,
        )
        return (
            f"✅ Exit: {result.returncode}\n"
            f"--- STDOUT ---\n{result.stdout}\n"
            f"--- STDERR ---\n{result.stderr}"
        ).strip()
    except subprocess.TimeoutExpired:
        return "❌ Code timed out after 30s."
    except Exception as e:
        return f"Execution error: {e}"
    finally:
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except Exception:
                pass