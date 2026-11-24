# Studio 01 - Brain, Voice & Audio (v2)

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/abhayraj01op/ai-creator-studio/blob/main/Studio_01_Brain_Voice_Audio.ipynb)

1-click link: https://kaggle.com/kernels/welcome?src=https://github.com/abhayraj01op/ai-creator-studio/blob/main/Studio_01_Brain_Voice_Audio.ipynb

## Steps
1. Upload `app_studio1.py` and `Studio_01_Brain_Voice_Audio.ipynb` to the GitHub repo `abhayraj01op/ai-creator-studio` (branch `main`).
2. Open the link above. In Kaggle settings choose **GPU T4 x2** and turn **Internet ON**.
3. Click **Run All**, then open the public Gradio link.

## Tools
1. Story & Script Writer - Qwen2.5-14B-Instruct (4-bit), 8 formats, 5 languages, 5 moods, 300-2500 words, director box -> `01_scripts`
2. Human Text-to-Voice - edge-tts (Hindi, Hinglish, Indian English, Urdu, Bengali, Marathi, Tamil, US/UK) + Kokoro -> `02_voices` (.wav + 320k .mp3)
3. Music & Song Studio - Mode A: MusicGen-medium up to 3 min (sliding window + crossfade). Mode B: Bark singing vocals mixed over MusicGen backing -> `05_music`
4. Sound Effects - AudioLDM2, 5 presets, 2-15 s -> `05_music`
5. Titles, Captions & Hashtags - Qwen2.5-14B -> `06_captions_hashtags`
6. Vocal Splitter (Demucs) + `AI_Studio_Master_Hub.zip` download

Only one heavy model stays in GPU memory; it is cleared (`gc.collect(); torch.cuda.empty_cache()`) when you switch tools.

## Notes
- edge-tts voices need Internet ON (free Microsoft voices, no key).
- Bark singing is experimental: results vary per run; regenerate if a line sounds off.
- First run of each tool downloads the model (14B model ~9 GB).

## Google Drive sync (optional)
Create a free OAuth "Desktop app" client in Google Cloud Console, download it as `client_secrets.json`, and add it to the notebook as a Kaggle dataset. On Run All you get a sign-in link; paste the returned address back. Without it, files are saved only in `/kaggle/working/AI_Studio_Master_Hub/`. Existing Drive folders are reused, never deleted.
