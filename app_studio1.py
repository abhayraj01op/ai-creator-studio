"""Studio_01_Brain_Voice_Audio v2 - Kaggle (GPU T4 x2, Internet ON, Python 3.13). Zero paid APIs.
Scripts+SEO (Qwen2.5-14B 4-bit), voices (edge-tts + Kokoro), long music + songs (MusicGen + Bark), SFX (AudioLDM2), splitter (Demucs)."""
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


def _free():
    import gc, torch
    _model["obj"] = None; _model["name"] = None
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


# ---------------- Qwen 14B (Tab 1 + Tab 5) ----------------
BANNED = ["delve", "tapestry", "testament", "in conclusion", "ever-evolving", "embark", "realm",
          "unleash", "game-changer", "navigate the", "in today's fast-paced world", "furthermore", "moreover"]

def _qwen():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    mid = "Qwen/Qwen2.5-14B-Instruct"
    tok = AutoTokenizer.from_pretrained(mid)
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                           bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    m = AutoModelForCausalLM.from_pretrained(mid, quantization_config=q, device_map="auto",
                                             torch_dtype=torch.float16)
    return tok, m

def _chat(system, user, max_new):
    import torch
    tok, m = _load("qwen14b", _qwen)
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(m.device)
    with torch.no_grad():
        out = m.generate(ids, max_new_tokens=max_new, do_sample=True, temperature=0.85, top_p=0.92,
                         repetition_penalty=1.08, pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
    for b in BANNED:  # last-line safety net against AI cliches
        text = text.replace(b, "").replace(b.capitalize(), "")
    return text

STORY_SYSTEM = (
    "You are a gifted human storyteller and screenwriter, not an AI. Write with raw human psychology: "
    "inner thoughts, fear, guilt, hope, small contradictions. Fill every scene with sensory detail - smells, "
    "sounds, textures, temperature, ambient atmosphere. Use realistic Indian and global cultural nuance "
    "(family dynamics, food, festivals, street life, local speech) where it fits. Dialogue must feel imperfect "
    "and natural: interruptions, half sentences, filler words. Use dramatic pacing - short punchy lines in "
    "tense moments, slower lines in emotional ones, natural pauses as '...'. "
    "STRICTLY NEVER use these words/phrases: " + ", ".join(BANNED) + ". No headings, no bullet points, "
    "no moral summary at the end unless the format asks for it. Output only the story/script.")

FORMATS = ["Real-Life Horror Story", "Heart-Touching Emotional Story", "YouTube Documentary 10-Min",
           "Suspense Thriller", "Viral Instagram Reel 60s", "Kids Moral Story", "Podcast Monologue", "Custom Format"]
LANGS = ["Conversational Hinglish", "Natural Spoken Hindi", "Indian English", "Urdu-Hindi Mix", "US/UK English"]
MOODS = ["Spine-Chilling & Atmospheric", "Deeply Emotional & Tear-Jerking", "Intense Suspense",
         "Dark & Gritty", "Warm & Inspiring"]
LANG_RULES = {
    "Conversational Hinglish": "Hindi in Roman letters mixed naturally with English, the way young Indians text and talk.",
    "Natural Spoken Hindi": "Simple spoken Hindi in Devanagari script, not bookish/shuddh Hindi.",
    "Indian English": "English with natural Indian expressions and rhythm.",
    "Urdu-Hindi Mix": "Hindustani with natural Urdu words (Devanagari or Roman script), poetic where it fits.",
    "US/UK English": "Natural native English.",
}

def write_script(topic, fmt, language, mood, words, director):
    if not topic.strip():
        return "Please enter a topic or story idea.", None, ""
    words = int(words)
    user = (f"Story idea: {topic}\nFormat: {fmt}\nLanguage: {language} - {LANG_RULES.get(language, '')}\n"
            f"Mood: {mood}\nTarget length: about {words} words.\n")
    if fmt == "Viral Instagram Reel 60s":
        user += "Start with a 3-second scroll-stopping hook. Keep it tight.\n"
    if fmt == "YouTube Documentary 10-Min":
        user += "Narration style with [VISUAL SCENE: ...] cues on their own lines.\n"
    if fmt == "Kids Moral Story":
        user += "Gentle, simple words; end with a soft one-line moral.\n"
    if director and director.strip():
        user += f"Director's instructions (follow exactly): {director.strip()}\n"
    text = _chat(STORY_SYSTEM, user, max_new=min(4096, int(words * 2.2) + 200))
    p = _out("01_scripts", "script", "txt")
    with open(p, "w", encoding="utf-8") as f: f.write(text)
    return text, p, save_and_sync(p, "01_scripts")


# ---------------- Tab 5: SEO ----------------
SEO_SYSTEM = ("You are a viral social media strategist who writes like a real person. No cliches, no "
              "emoji spam. NEVER use: " + ", ".join(BANNED) + ".")

def seo(topic, platform, language, custom):
    if not topic.strip():
        return "Please enter the video topic or paste the script.", None, ""
    user = (f"Content: {topic}\nPlatform: {platform}\nLanguage: {language}\n"
            "Give exactly these sections:\n"
            "TITLES: 5 curiosity-driven titles (numbered)\n"
            "CAPTIONS: 3 short human storytelling captions (numbered)\n"
            "PINNED COMMENT: 1 hook that makes people reply\n"
            "HASHTAGS: 15 hashtags mixing niche and broad, on one line\n")
    if custom and custom.strip():
        user += f"Extra instructions: {custom.strip()}\n"
    text = _chat(SEO_SYSTEM, user, max_new=900)
    p = _out("06_captions_hashtags", "seo", "txt")
    with open(p, "w", encoding="utf-8") as f: f.write(text)
    return text, p, save_and_sync(p, "06_captions_hashtags")


# ---------------- Tab 2: Text to voice (edge-tts + Kokoro) ----------------
VOICES = {
    "Hindi - Swara (Female, India)": ("edge", "hi-IN-SwaraNeural"),
    "Hindi - Madhur (Male, India)": ("edge", "hi-IN-MadhurNeural"),
    "Hinglish - Swara (Female, India)": ("edge", "hi-IN-SwaraNeural"),
    "Indian English - Neerja (Female)": ("edge", "en-IN-NeerjaNeural"),
    "Indian English - Prabhat (Male)": ("edge", "en-IN-PrabhatNeural"),
    "Urdu - Uzma (Female, Pakistan)": ("edge", "ur-PK-UzmaNeural"),
    "Urdu - Asad (Male, Pakistan)": ("edge", "ur-PK-AsadNeural"),
    "Bengali - Tanishaa (Female, India)": ("edge", "bn-IN-TanishaaNeural"),
    "Bengali - Bashkar (Male, India)": ("edge", "bn-IN-BashkarNeural"),
    "Marathi - Aarohi (Female, India)": ("edge", "mr-IN-AarohiNeural"),
    "Marathi - Manohar (Male, India)": ("edge", "mr-IN-ManoharNeural"),
    "Tamil - Pallavi (Female, India)": ("edge", "ta-IN-PallaviNeural"),
    "Tamil - Valluvar (Male, India)": ("edge", "ta-IN-ValluvarNeural"),
    "US English - Aria (Female, Global)": ("edge", "en-US-AriaNeural"),
    "US English - Guy (Male, Global)": ("edge", "en-US-GuyNeural"),
    "UK English - Sonia (Female, Global)": ("edge", "en-GB-SoniaNeural"),
    "Kokoro - Heart (US Female, offline)": ("kokoro", ("a", "af_heart")),
    "Kokoro - Michael (US Male, offline)": ("kokoro", ("a", "am_michael")),
    "Kokoro - Emma (UK Female, offline)": ("kokoro", ("b", "bf_emma")),
    "Kokoro - George (UK Male, offline)": ("kokoro", ("b", "bm_george")),
    "Kokoro - Alpha (Hindi Female, offline)": ("kokoro", ("h", "hf_alpha")),
    "Kokoro - Omega (Hindi Male, offline)": ("kokoro", ("h", "hm_omega")),
}
STYLES = {  # (speed multiplier, pitch offset Hz, paragraph pause seconds)
    "Natural Storytelling": (1.00, 0, 0.6),
    "Low-Pitch Horror Whisper": (0.88, -10, 1.0),
    "High-Energy YouTuber": (1.12, 4, 0.3),
    "Calm Documentary Narrator": (0.94, -3, 0.8),
    "Emotional & Soft": (0.90, -2, 0.9),
}

def _parse_pause(notes, default):
    import re
    m = re.search(r"(\d+(?:\.\d+)?)\s*(s|sec|second)", notes or "", re.I)
    return float(m.group(1)) if m else default

def _edge_tts(text, voice, rate_pct, pitch_hz, path):
    import asyncio, edge_tts
    async def run():
        c = edge_tts.Communicate(text, voice, rate=f"{rate_pct:+d}%", pitch=f"{pitch_hz:+d}Hz")
        await c.save(path)
    try:
        asyncio.run(run())
    except RuntimeError:  # an event loop is already running
        import threading
        t = threading.Thread(target=lambda: asyncio.run(run())); t.start(); t.join()

def tts(text, voice_label, style, speed, pitch, notes):
    from pydub import AudioSegment
    if not text.strip():
        return None, None, "Please enter the script text."
    engine, voice = VOICES[voice_label]
    s_mul, s_pitch, s_pause = STYLES[style]
    spd = float(speed) * s_mul
    pit = int(pitch) + s_pitch
    pause = _parse_pause(notes, s_pause)
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    tmp = os.path.join(LOCAL_HUB, "02_voices", "_tmp"); os.makedirs(tmp, exist_ok=True)
    full = AudioSegment.silent(duration=200)
    if engine == "edge":
        for i, para in enumerate(paras):
            pth = os.path.join(tmp, f"p{i}.mp3")
            _edge_tts(para, voice, int(round((spd - 1) * 100)), pit, pth)
            full += AudioSegment.from_file(pth) + AudioSegment.silent(duration=int(pause * 1000))
    else:
        import numpy as np, soundfile as sf
        lang, vname = voice
        pipe = _load(f"kokoro_{lang}", lambda: __import__("kokoro").KPipeline(lang_code=lang))
        for i, para in enumerate(paras):
            chunks = [a for _, _, a in pipe(para, voice=vname, speed=spd)]
            pth = os.path.join(tmp, f"p{i}.wav")
            sf.write(pth, np.concatenate([np.asarray(c) for c in chunks]), 24000)
            seg = AudioSegment.from_file(pth)
            if pit:  # simple pitch shift by resampling
                f = 2 ** (pit / 120)
                seg = seg._spawn(seg.raw_data, overrides={"frame_rate": int(seg.frame_rate * f)}).set_frame_rate(24000)
            full += seg + AudioSegment.silent(duration=int(pause * 1000))
    wav = _out("02_voices", "voice", "wav")
    full.export(wav, format="wav")
    mp3 = wav[:-4] + ".mp3"
    full.export(mp3, format="mp3", bitrate="320k")
    msg = save_and_sync(wav, "02_voices") + "\n" + save_and_sync(mp3, "02_voices")
    return wav, mp3, msg


# ---------------- Tab 3: Long music (MusicGen) + songs (Bark) ----------------
def _musicgen():
    import torch
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
    proc = AutoProcessor.from_pretrained("facebook/musicgen-medium")
    m = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-medium",
                                                         torch_dtype=torch.float16).to("cuda")
    return proc, m

def _long_music(prompt, seconds):
    """Sliding-window continuation: each 30s window is conditioned on the last 10s, joined with crossfades."""
    import torch, numpy as np, soundfile as sf
    from pydub import AudioSegment
    proc, m = _load("musicgen", _musicgen)
    sr = m.config.audio_encoder.sampling_rate
    tps = m.config.audio_encoder.frame_rate  # tokens per second (50)
    WIN, CTX, XF = 30, 10, 2000
    tmp = os.path.join(LOCAL_HUB, "05_music", "_tmp"); os.makedirs(tmp, exist_ok=True)
    song, prev = None, None
    remaining = float(seconds)
    i = 0
    while remaining > 0.5:
        new_len = min(WIN if prev is None else WIN - CTX, remaining)
        with torch.no_grad():
            if prev is None:
                inp = proc(text=[prompt], padding=True, return_tensors="pt").to("cuda")
                out = m.generate(**inp, max_new_tokens=int(new_len * tps) + 4, do_sample=True, guidance_scale=3)
                audio = out[0, 0].float().cpu().numpy()
            else:
                ctx = prev[-CTX * sr:]
                inp = proc(audio=[ctx], sampling_rate=sr, text=[prompt], padding=True, return_tensors="pt").to("cuda")
                inp["input_values"] = inp["input_values"].half()
                out = m.generate(**inp, max_new_tokens=int((CTX + new_len) * tps) + 4, do_sample=True, guidance_scale=3)
                audio = out[0, 0].float().cpu().numpy()[len(ctx):]  # drop the repeated context
        prev = audio if prev is None else np.concatenate([prev, audio])
        pth = os.path.join(tmp, f"w{i}.wav"); sf.write(pth, audio, sr)
        seg = AudioSegment.from_file(pth)
        song = seg if song is None else song.append(seg, crossfade=min(XF, len(seg) // 2))
        remaining -= new_len; i += 1
    return song[: int(seconds * 1000)].fade_in(500).fade_out(1500)

def music_mode_a(prompt, seconds):
    if not prompt.strip():
        return None, "Please describe the music."
    song = _long_music(prompt, seconds)
    p = _out("05_music", "bgm", "wav"); song.export(p, format="wav")
    return p, save_and_sync(p, "05_music")

SINGERS = {
    "Male - Hindi": "v2/hi_speaker_8", "Female - Hindi": "v2/hi_speaker_9",
    "Male - English": "v2/en_speaker_6", "Female - English": "v2/en_speaker_9",
    "Male - Global": "v2/en_speaker_3", "Female - Global": "v2/de_speaker_3",
}
GENRES = {
    "Bollywood Acoustic": "bollywood acoustic guitar ballad, soft tabla, warm, emotional",
    "Lo-Fi Chill": "lo-fi chill hip hop beat, vinyl crackle, mellow keys",
    "Sad Soulful": "sad soulful piano and strings, slow, melancholic",
    "Sufi Folk": "sufi folk, harmonium, dholak, spiritual, qawwali feel",
    "Hip-Hop Rap": "hip hop rap beat, punchy 808 drums, dark synth",
    "Cinematic": "cinematic orchestral, epic drums, emotional strings",
}

def _bark():
    import torch
    from transformers import AutoProcessor, BarkModel
    proc = AutoProcessor.from_pretrained("suno/bark")
    m = BarkModel.from_pretrained("suno/bark", torch_dtype=torch.float16).to("cuda")
    return proc, m

def music_mode_b(lyrics, singer, genre, custom, seconds):
    import torch, numpy as np, soundfile as sf
    from pydub import AudioSegment
    if not lyrics.strip():
        return None, "Please write the song lyrics."
    proc, m = _load("bark", _bark)
    sr = m.generation_config.sample_rate
    tmp = os.path.join(LOCAL_HUB, "05_music", "_tmp"); os.makedirs(tmp, exist_ok=True)
    vocals = AudioSegment.silent(duration=300)
    lines = [l.strip() for l in lyrics.split("\n") if l.strip()]
    for i, line in enumerate(lines):
        if line.startswith("[") and line.endswith("]"):  # [Verse] / [Chorus] -> short breath
            vocals += AudioSegment.silent(duration=700); continue
        sung = line if "♪" in line else f"♪ {line} ♪"
        inp = proc(sung, voice_preset=SINGERS[singer]).to("cuda")
        with torch.no_grad():
            a = m.generate(**inp, do_sample=True).float().cpu().numpy().squeeze()
        pth = os.path.join(tmp, f"v{i}.wav"); sf.write(pth, a, sr)
        vocals += AudioSegment.from_file(pth) + AudioSegment.silent(duration=250)
        if len(vocals) >= seconds * 1000: break
    _free()  # unload Bark before MusicGen
    dur = max(len(vocals) / 1000 + 2, 15)
    bgm = _long_music(f"{GENRES[genre]}, {custom or ''}, instrumental backing track", min(dur, 180))
    mix = (bgm - 9).overlay(vocals + 2, position=1000)
    mix = mix[: int(max(seconds, len(vocals) / 1000 + 2) * 1000)].fade_out(1500)
    p = _out("05_music", "song", "wav"); mix.export(p, format="wav")
    return p, save_and_sync(p, "05_music")


# ---------------- Tab 4: SFX (AudioLDM2) ----------------
ENV = {
    "Horror Night": "eerie night ambience, distant howl, creaking wood, wind",
    "Rain & Thunder": "heavy rain on a tin roof, rolling thunder",
    "Busy Indian Street": "busy Indian street, auto rickshaw horns, vendors shouting, traffic",
    "Cinematic Whoosh/Impact": "cinematic whoosh followed by a deep booming impact",
    "Custom": "",
}

def _aldm():
    import torch
    from diffusers import AudioLDM2Pipeline
    return AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2", torch_dtype=torch.float16).to("cuda")

def sfx(desc, env, seconds):
    import soundfile as sf
    prompt = ", ".join(x for x in [desc.strip(), ENV.get(env, "")] if x)
    if not prompt:
        return None, "Please describe the sound or pick a preset."
    pipe = _load("audioldm2", _aldm)
    a = pipe(prompt, negative_prompt="low quality, noise, distorted", num_inference_steps=150,
             audio_length_in_s=float(seconds)).audios[0]
    p = _out("05_music", "sfx", "wav"); sf.write(p, a, 16000)
    return p, save_and_sync(p, "05_music")


# ---------------- Tab 6: Splitter + ZIP ----------------
def split(file):
    import subprocess, shutil
    if file is None:
        return None, None, "Please upload a song or video."
    src = file if isinstance(file, str) else file.name
    _free()
    work = os.path.join(LOCAL_HUB, "05_music", "_demucs")
    r = subprocess.run([sys.executable, "-m", "demucs", "--two-stems=vocals", "-o", work, src],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None, None, "Demucs failed:\n" + r.stderr[-1500:]
    base = os.path.splitext(os.path.basename(src))[0]
    found = glob.glob(os.path.join(work, "*", base, "vocals.wav"))
    if not found:
        return None, None, "Demucs finished but output was not found."
    d = os.path.dirname(found[0])
    v, b = _out("05_music", "vocals", "wav"), _out("05_music", "bgm_only", "wav")
    shutil.copy(os.path.join(d, "vocals.wav"), v); shutil.copy(os.path.join(d, "no_vocals.wav"), b)
    return v, b, save_and_sync(v, "05_music") + "\n" + save_and_sync(b, "05_music")

def make_zip():
    import shutil
    base = f"/kaggle/working/{HUB_NAME}"
    return shutil.make_archive(base, "zip", LOCAL_HUB) if os.path.isdir(LOCAL_HUB) else None


# ---------------- UI ----------------
CSS = """
.gradio-container{background:#0b0d12 !important;color:#f3f4f6}
h1,h2,h3,label,span,p{color:#f3f4f6}
.tab-nav button,button[role=tab]{color:#f3f4f6 !important;background:#1e2430 !important;
  border:1px solid #f5c54255 !important;font-weight:600 !important;opacity:1 !important}
.tab-nav button.selected,button[role=tab][aria-selected=true]{background:#f5c542 !important;
  color:#000000 !important;font-weight:800 !important}
button.primary{background:#f5c542 !important;color:#000 !important;font-weight:700 !important}
.block,.form{background:#141821 !important;border-color:#f5c54233 !important}
textarea,input{background:#1e2430 !important;color:#f3f4f6 !important}
"""

def build_ui():
    import gradio as gr
    setup_folders()
    with gr.Blocks(title="Studio 01 - Brain, Voice & Audio", css=CSS) as ui:
        gr.HTML("<h1 style='color:#f5c542;text-align:center'>Studio 01 - Brain, Voice &amp; Audio</h1>"
                f"<p style='text-align:center'>Everything auto-saves to {LOCAL_HUB}"
                f"{' and syncs to Google Drive' if drive else ' (Drive sync off)'}</p>")
        with gr.Tab("1. Story & Script Writer"):
            t = gr.Textbox(label="Video Topic / Story Idea", lines=3)
            with gr.Row():
                f = gr.Dropdown(FORMATS, value=FORMATS[0], label="Format")
                l = gr.Dropdown(LANGS, value=LANGS[0], label="Language")
                mo = gr.Dropdown(MOODS, value=MOODS[0], label="Storytelling Emotion / Mood")
            w = gr.Slider(300, 2500, value=800, step=50, label="Target Word Count")
            dr = gr.Textbox(label="Custom Director Instructions (characters, twists, scenes, rules)", lines=3)
            b = gr.Button("Write Story", variant="primary")
            o = gr.Textbox(label="Story / Script", lines=22); fo = gr.File(label="Download .txt"); st = gr.Textbox(label="Status")
            b.click(write_script, [t, f, l, mo, w, dr], [o, fo, st])
        with gr.Tab("2. Human Text-to-Voice"):
            tx = gr.Textbox(label="Script Text", lines=10)
            with gr.Row():
                vv = gr.Dropdown(list(VOICES), value=list(VOICES)[0], label="Language & Voice Actor")
                sy = gr.Dropdown(list(STYLES), value="Natural Storytelling", label="Speaking Emotion / Delivery Style")
            with gr.Row():
                sp = gr.Slider(0.70, 1.30, value=0.95, step=0.01, label="Speed (x)")
                pi = gr.Slider(-15, 15, value=0, step=1, label="Voice Pitch (Hz)")
            nt = gr.Textbox(label="Custom Voice Direction / Pause Notes (e.g. '1.5 sec pause between paragraphs')")
            b2 = gr.Button("Generate Voice", variant="primary")
            a2 = gr.Audio(label="Voice (.wav)", type="filepath"); m2 = gr.File(label="Download .mp3"); s2 = gr.Textbox(label="Status")
            b2.click(tts, [tx, vv, sy, sp, pi, nt], [a2, m2, s2])
        with gr.Tab("3. Music & Song Studio"):
            with gr.Tab("Mode A - Long Instrumental / BGM"):
                mp = gr.Textbox(label="Custom Music Prompt", value="cinematic lo-fi piano with soft rain, emotional")
                md = gr.Slider(15, 180, value=60, step=5, label="Duration (seconds)")
                b3 = gr.Button("Generate Music", variant="primary")
                a3 = gr.Audio(type="filepath"); s3 = gr.Textbox(label="Status")
                b3.click(music_mode_a, [mp, md], [a3, s3])
            with gr.Tab("Mode B - Full Song with Vocals"):
                ly = gr.Textbox(label="Song Lyrics ([Verse], [Chorus], ♪ allowed)", lines=10,
                                value="[Verse]\n♪ Tere bina yeh raatein ♪\n♪ Kitni lambi lagti hain ♪\n[Chorus]\n♪ Laut aa, laut aa ♪")
                with gr.Row():
                    sg = gr.Dropdown(list(SINGERS), value="Female - Hindi", label="Singer Voice")
                    gn = gr.Dropdown(list(GENRES), value="Bollywood Acoustic", label="Genre")
                cp = gr.Textbox(label="Custom Music / Lyrics Prompt")
                sd = gr.Slider(15, 180, value=45, step=5, label="Duration (seconds)")
                b4 = gr.Button("Generate Song", variant="primary")
                a4 = gr.Audio(type="filepath"); s4 = gr.Textbox(label="Status")
                b4.click(music_mode_b, [ly, sg, gn, cp, sd], [a4, s4])
        with gr.Tab("4. Sound Effects"):
            de = gr.Textbox(label="SFX Description")
            en = gr.Dropdown(list(ENV), value="Horror Night", label="Environment Preset")
            du = gr.Slider(2, 15, value=6, step=1, label="Duration (seconds)")
            b5 = gr.Button("Generate SFX", variant="primary")
            a5 = gr.Audio(type="filepath"); s5 = gr.Textbox(label="Status")
            b5.click(sfx, [de, en, du], [a5, s5])
        with gr.Tab("5. Titles, Captions & Hashtags"):
            tp = gr.Textbox(label="Video Topic or Script", lines=6)
            with gr.Row():
                pf = gr.Dropdown(["YouTube", "YouTube Shorts", "Instagram Reels", "Facebook", "TikTok"], value="YouTube", label="Platform")
                lg = gr.Dropdown(LANGS, value=LANGS[0], label="Language")
            cs = gr.Textbox(label="Custom SEO Instructions")
            b6 = gr.Button("Generate SEO Pack", variant="primary")
            o6 = gr.Textbox(label="Result", lines=18); f6 = gr.File(); s6 = gr.Textbox(label="Status")
            b6.click(seo, [tp, pf, lg, cs], [o6, f6, s6])
        with gr.Tab("6. Vocal Splitter & ZIP"):
            up = gr.File(label="Upload song or video")
            b7 = gr.Button("Split Vocals & Music", variant="primary")
            v7 = gr.Audio(label="Vocals", type="filepath"); m7 = gr.Audio(label="Music only", type="filepath"); s7 = gr.Textbox(label="Status")
            b7.click(split, [up], [v7, m7, s7])
            gr.Markdown("---")
            bz = gr.Button("Download AI_Studio_Master_Hub.zip", variant="primary"); fz = gr.File()
            bz.click(make_zip, None, fz)
    return ui
