"""Studio_01_Brain_Voice_Audio - Kaggle (GPU T4 x2, Internet ON). Zero paid APIs.
Scripts (Qwen2.5-7B), TTS (Kokoro), voice clone (F5-TTS), music (MusicGen), SFX (AudioLDM2), splitter (Demucs)."""
import os, sys, glob
from datetime import datetime
from urllib.parse import urlparse, parse_qs

HUB_NAME = "AI_Studio_Master_Hub"
LOCAL_HUB = f"/kaggle/working/{HUB_NAME}"
SUBFOLDERS = ["00_reference_faces", "01_scripts", "02_voices", "03_images",
              "04_raw_videos", "05_music", "06_captions_hashtags", "07_final_exports"]
FOLDER_MIME = "application/vnd.google-apps.folder"

drive = None          # PyDrive2 GoogleDrive instance (None = local-only mode)
drive_ids = {}        # subfolder name -> Drive folder id


# ---------------- Google Drive (PyDrive2) ----------------
def _find_client_secrets():
    for p in ["client_secrets.json", "/kaggle/working/client_secrets.json",
              *glob.glob("/kaggle/input/**/client_secrets*.json", recursive=True)]:
        if os.path.exists(p):
            return p
    return None


def connect_drive():
    """Prints a Google sign-in link; the user pastes the code (or the full redirected URL) once."""
    global drive
    from pydrive2.auth import GoogleAuth
    from pydrive2.drive import GoogleDrive
    secrets = _find_client_secrets()
    if not secrets:
        print("No client_secrets.json found -> LOCAL-ONLY mode (images still saved in", LOCAL_HUB, ").\n"
              "See README 'Google Drive setup' to enable Drive sync.")
        return None
    creds_file = "/kaggle/working/drive_creds.json"
    gauth = GoogleAuth(settings={
        "client_config_backend": "file", "client_config_file": secrets,
        "save_credentials": True, "save_credentials_backend": "file", "save_credentials_file": creds_file,
        "get_refresh_token": True, "oauth_scope": ["https://www.googleapis.com/auth/drive"],
    })
    gauth.LoadCredentialsFile(creds_file)
    if gauth.credentials is None:
        gauth.GetFlow()
        gauth.flow.redirect_uri = "http://localhost:8080/"
        url = gauth.GetAuthUrl()
        print("1) Open this link and sign in with Google:\n\n", url, "\n")
        print("2) After allowing access the browser shows 'site can't be reached' - that is normal.\n"
              "   Copy the FULL address from the address bar (or just the code=... value) and paste it below.")
        raw = input("Paste here: ").strip()
        code = parse_qs(urlparse(raw).query).get("code", [raw])[0] if raw.startswith("http") else raw
        gauth.Auth(code)
    elif gauth.access_token_expired:
        gauth.Refresh()
    else:
        gauth.Authorize()
    gauth.SaveCredentialsFile(creds_file)
    drive = GoogleDrive(gauth)
    print("Google Drive connected.")
    return drive


def _get_or_create_folder(name, parent_id):
    """Reuses an existing folder; creates only if missing. Never deletes or renames."""
    q = f"title='{name}' and mimeType='{FOLDER_MIME}' and '{parent_id}' in parents and trashed=false"
    found = drive.ListFile({"q": q}).GetList()
    if found:
        return found[0]["id"]
    f = drive.CreateFile({"title": name, "mimeType": FOLDER_MIME, "parents": [{"id": parent_id}]})
    f.Upload()
    return f["id"]


def setup_folders():
    for s in SUBFOLDERS:
        os.makedirs(os.path.join(LOCAL_HUB, s), exist_ok=True)
    if drive is not None:
        root = _get_or_create_folder(HUB_NAME, "root")
        for s in SUBFOLDERS:
            drive_ids[s] = _get_or_create_folder(s, root)
        print(f"Drive folder My Drive/{HUB_NAME} ready with {len(SUBFOLDERS)} subfolders.")
    print("Local folder ready:", LOCAL_HUB)


def save_and_sync_to_drive(local_file_path, subfolder_name):
    """File is already saved locally under LOCAL_HUB/subfolder; uploads a copy into the matching Drive folder."""
    if drive is None or subfolder_name not in drive_ids:
        return f"Saved locally only: {local_file_path}"
    try:
        f = drive.CreateFile({"title": os.path.basename(local_file_path),
                              "parents": [{"id": drive_ids[subfolder_name]}]})
        f.SetContentFile(local_file_path)
        f.Upload()
        return f"Synced to Drive: {HUB_NAME}/{subfolder_name}/{os.path.basename(local_file_path)}"
    except Exception as e:
        return f"Saved locally, Drive upload failed: {e}"


def save_and_sync(local_path, subfolder):
    return save_and_sync_to_drive(local_path, subfolder)


def _out(subfolder, prefix, ext):
    folder = os.path.join(LOCAL_HUB, subfolder); os.makedirs(folder, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    p, n = os.path.join(folder, f"{prefix}_{stamp}.{ext}"), 1
    while os.path.exists(p):  # never overwrite
        p = os.path.join(folder, f"{prefix}_{stamp}_{n}.{ext}"); n += 1
    return p


# ---------------- VRAM auto-cleaner: only one model in memory at a time ----------------
_model = {"name": None, "obj": None}

def _load(name, loader):
    import gc, torch
    if _model["name"] != name:
        _model["obj"] = None; _model["name"] = None
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        _model["obj"] = loader(); _model["name"] = name
    return _model["obj"]


# ---------------- Tab 1: Script writer ----------------
BANNED = ["delve", "tapestry", "testament", "in conclusion", "ever-evolving", "embark", "realm",
          "unleash", "game-changer", "navigate the", "in today's fast-paced world"]
SYSTEM = ("You are a real human YouTube/Reels scriptwriter. Write like people actually talk: short sentences, "
          "contractions, natural pauses written as '...', and put camera/visual cues on their own line as "
          "[VISUAL SCENE: ...]. NEVER use these words or phrases: " + ", ".join(BANNED) +
          ". No headings, no bullet lists, no summaries at the end.")

def _qwen():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    mid = "Qwen/Qwen2.5-7B-Instruct"
    tok = AutoTokenizer.from_pretrained(mid)
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
    m = AutoModelForCausalLM.from_pretrained(mid, quantization_config=q, device_map="auto")
    return tok, m

def write_script(topic, language, length, tone):
    import re, torch
    if not topic.strip(): return "", "Enter a topic."
    tok, m = _load("qwen", _qwen)
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Topic: {topic}\nLanguage: {language}\nTone: {tone}\nLength: about {int(length)} words."}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(m.device)
    with torch.no_grad():
        out = m.generate(ids, max_new_tokens=int(length * 2.2) + 64, do_sample=True, temperature=0.85, top_p=0.9)
    text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
    for w in BANNED:  # safety net
        text = re.sub(re.escape(w), "", text, flags=re.I)
    p = _out("01_scripts", "script", "txt")
    with open(p, "w", encoding="utf-8") as f: f.write(text)
    return text, save_and_sync(p, "01_scripts")


# ---------------- Tab 2: Kokoro TTS ----------------
KOKORO_VOICES = {"Hindi female (hf_alpha)": ("h", "hf_alpha"), "Hindi male (hm_omega)": ("h", "hm_omega"),
                 "English US female (af_heart)": ("a", "af_heart"), "English US male (am_michael)": ("a", "am_michael"),
                 "English UK female (bf_emma)": ("b", "bf_emma")}

def tts(text, voice, speed):
    import numpy as np, soundfile as sf
    from kokoro import KPipeline
    if not text.strip(): return None, "Enter text."
    lang, v = KOKORO_VOICES[voice]
    pipe = _load(f"kokoro_{lang}", lambda: KPipeline(lang_code=lang))
    sr, pause = 24000, np.zeros(int(24000 * 0.25), dtype=np.float32)  # 0.25s breath
    chunks = []
    for para in [p.strip() for p in text.split("\n") if p.strip()]:
        for _, _, audio in pipe(para, voice=v, speed=float(speed)):
            chunks.append(np.asarray(audio, dtype=np.float32))
        chunks.append(pause)
    p = _out("02_voices", "voice", "wav"); sf.write(p, np.concatenate(chunks), sr)
    return p, save_and_sync(p, "02_voices")


# ---------------- Tab 3: F5-TTS voice clone ----------------
def clone(ref_audio, ref_text, new_text, speed):
    import soundfile as sf
    if not ref_audio or not new_text.strip(): return None, "Upload ~10s reference audio and enter new text."
    from f5_tts.api import F5TTS
    f5 = _load("f5", lambda: F5TTS())
    wav, sr, _ = f5.infer(ref_file=ref_audio, ref_text=ref_text or "", gen_text=new_text, speed=float(speed))
    p = _out("02_voices", "clone", "wav"); sf.write(p, wav, sr)
    return p, save_and_sync(p, "02_voices")


# ---------------- Tab 4: MusicGen ----------------
def music(prompt, seconds):
    import numpy as np, soundfile as sf
    from transformers import pipeline
    if not prompt.strip(): return None, "Describe the music."
    pipe = _load("musicgen", lambda: pipeline("text-to-audio", model="facebook/musicgen-medium", device=0))
    r = pipe(prompt, forward_params={"do_sample": True, "max_new_tokens": int(seconds * 50)})
    audio = np.squeeze(r["audio"])
    p = _out("05_music", "bgm", "wav"); sf.write(p, audio.T if audio.ndim > 1 else audio, r["sampling_rate"])
    return p, save_and_sync(p, "05_music")


# ---------------- Tab 5: AudioLDM2 SFX ----------------
def _aldm():
    import torch
    from diffusers import AudioLDM2Pipeline
    return AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2", torch_dtype=torch.float16).to("cuda")

def sfx(prompt, seconds, steps):
    import soundfile as sf
    if not prompt.strip(): return None, "Describe the sound."
    pipe = _load("audioldm2", _aldm)
    a = pipe(prompt, negative_prompt="low quality, noise, distortion", num_inference_steps=int(steps),
             audio_length_in_s=float(seconds)).audios[0]
    p = _out("05_music", "sfx", "wav"); sf.write(p, a, 16000)
    return p, save_and_sync(p, "05_music")


# ---------------- Tab 6: Demucs splitter ----------------
def split(file):
    import subprocess, shutil, tempfile
    if not file: return None, None, "Upload a song or video."
    _load(None, lambda: None)  # free VRAM
    tmp = tempfile.mkdtemp()
    src = file
    if not file.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
        src = os.path.join(tmp, "input.wav")
        subprocess.run(["ffmpeg", "-y", "-i", file, "-vn", "-ac", "2", "-ar", "44100", src], check=True, capture_output=True)
    subprocess.run([sys.executable, "-m", "demucs", "--two-stems=vocals", "-o", tmp, src], check=True)
    stem = glob.glob(os.path.join(tmp, "*", "*"))[0]
    v, b = _out("02_voices", "vocals", "wav"), _out("05_music", "instrumental", "wav")
    shutil.copy(os.path.join(stem, "vocals.wav"), v); shutil.copy(os.path.join(stem, "no_vocals.wav"), b)
    return v, b, save_and_sync(v, "02_voices") + "\n" + save_and_sync(b, "05_music")


# ---------------- UI ----------------
CSS = """
.gradio-container{background:radial-gradient(circle at top,#1a1a22 0%,#0b0b0f 70%)!important;color:#e8e2d0}
#title h1{font-family:Georgia,serif;letter-spacing:.12em;color:#d4af37;text-align:center;margin:8px 0}
#title p{text-align:center;color:#9a937f}
.block,.form{background:#131318!important;border:1px solid #3a3220!important;border-radius:14px!important}
button.primary{background:linear-gradient(90deg,#b8902c,#e6c463)!important;color:#111!important;font-weight:700}
.tab-nav button.selected{color:#d4af37!important;border-color:#d4af37!important}
"""

def build_ui():
    import gradio as gr
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="amber", neutral_hue="slate"), css=CSS,
                   title="AI Creator Studio 01") as demo:
        gr.Markdown("# AI CREATOR STUDIO 01 — BRAIN, VOICE & AUDIO SUITE\n"
                    f"Every file is saved to `{HUB_NAME}` and synced to Google Drive when connected.", elem_id="title")
        with gr.Tab("1 · Script Writer"):
            t = gr.Textbox(label="Topic", lines=2)
            with gr.Row():
                lang = gr.Dropdown(["Hinglish", "Hindi", "English"], value="Hinglish", label="Language")
                tone = gr.Dropdown(["Conversational", "Storytelling", "Documentary", "Funny"], value="Conversational", label="Tone")
                ln = gr.Slider(80, 800, 250, step=10, label="Words")
            b = gr.Button("Write script", variant="primary")
            o, s = gr.Textbox(label="Script", lines=16, show_copy_button=True), gr.Textbox(label="Status")
            b.click(write_script, [t, lang, ln, tone], [o, s])
        with gr.Tab("2 · Text to Voice"):
            tx = gr.Textbox(label="Text (new line = 0.25s breath pause)", lines=8)
            with gr.Row():
                vc = gr.Dropdown(list(KOKORO_VOICES), value=list(KOKORO_VOICES)[0], label="Voice")
                sp = gr.Slider(0.6, 1.4, 0.95, step=0.05, label="Speed")
            b = gr.Button("Generate voice", variant="primary")
            a, s = gr.Audio(label="Voice", type="filepath"), gr.Textbox(label="Status")
            b.click(tts, [tx, vc, sp], [a, s])
        with gr.Tab("3 · Voice Cloner"):
            ra = gr.Audio(label="10-second reference voice", type="filepath")
            rt = gr.Textbox(label="What is said in the reference (optional, improves quality)")
            nt = gr.Textbox(label="New text to speak", lines=6)
            sp = gr.Slider(0.6, 1.4, 1.0, step=0.05, label="Speed")
            b = gr.Button("Clone voice", variant="primary")
            a, s = gr.Audio(label="Cloned voice", type="filepath"), gr.Textbox(label="Status")
            b.click(clone, [ra, rt, nt, sp], [a, s])
        with gr.Tab("4 · Music / BGM"):
            mp = gr.Textbox(label="Describe music", value="calm Indian bansuri flute with soft acoustic guitar, documentary mood")
            sec = gr.Slider(5, 30, 15, step=1, label="Seconds")
            b = gr.Button("Generate music", variant="primary")
            a, s = gr.Audio(label="Music", type="filepath"), gr.Textbox(label="Status")
            b.click(music, [mp, sec], [a, s])
        with gr.Tab("5 · Sound Effects"):
            fp = gr.Textbox(label="Describe sound", value="heavy rain on a tin roof with distant thunder")
            with gr.Row():
                sec = gr.Slider(2, 20, 8, step=1, label="Seconds")
                st = gr.Slider(50, 200, 100, step=10, label="Quality steps")
            b = gr.Button("Generate SFX", variant="primary")
            a, s = gr.Audio(label="SFX", type="filepath"), gr.Textbox(label="Status")
            b.click(sfx, [fp, sec, st], [a, s])
        with gr.Tab("6 · Vocal Splitter"):
            f = gr.File(label="Song or video", type="filepath")
            b = gr.Button("Split vocals / music", variant="primary")
            v, m = gr.Audio(label="Vocals", type="filepath"), gr.Audio(label="Instrumental", type="filepath")
            s = gr.Textbox(label="Status")
            b.click(split, [f], [v, m, s])
    return demo


if __name__ == "__main__":
    connect_drive(); setup_folders(); build_ui().launch(share=True)
