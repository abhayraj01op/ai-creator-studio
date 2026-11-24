# AI Creator Studio 01 — Brain, Voice & Audio Suite

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/abhayraj01op/ai-creator-studio/blob/main/Studio_01_Brain_Voice_Audio.ipynb)

Link: https://kaggle.com/kernels/welcome?src=https://github.com/abhayraj01op/ai-creator-studio/blob/main/Studio_01_Brain_Voice_Audio.ipynb

## 3 steps
1. Upload all 4 files to the **main** branch of `github.com/abhayraj01op/ai-creator-studio`.
2. Click the button above. In Kaggle settings: **GPU T4 x2**, **Internet ON**.
3. **Run All**, open the Google link, paste the code once, then open the public Gradio link.

## Tabs (all free, open-source)
| Tab | Model | Saves to |
|---|---|---|
| Script Writer | Qwen/Qwen2.5-7B-Instruct (4-bit) | 01_scripts |
| Text to Voice | hexgrad/Kokoro-82M (speed 0.95, 0.25s breaths) | 02_voices |
| Voice Cloner | SWivid/F5-TTS (10s reference) | 02_voices |
| Music / BGM | facebook/musicgen-medium (5–30s) | 05_music |
| Sound Effects | cvssp/audioldm2 | 05_music |
| Vocal Splitter | demucs --two-stems=vocals | 02_voices + 05_music |

Only one model stays in GPU memory; switching tabs auto-clears VRAM.

## Google Drive setup (one time, ~5 min)
Google requires your own free sign-in file:
1. console.cloud.google.com → new project → enable **Google Drive API**.
2. OAuth consent screen → External → add your email as test user.
3. Credentials → Create OAuth client ID → **Desktop app** → download JSON, rename to `client_secrets.json`.
4. In Kaggle: Add Data → upload it as a private dataset.

Without it, files are still saved in `/kaggle/working/AI_Studio_Master_Hub`. Folders are reused, never deleted; files never overwritten.

Only clone voices you have permission to use.
