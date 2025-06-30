from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import List

import torch
import torchaudio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from transformers import (
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

# ---------------------------------------------------------------------------
# Environment & model loading
# ---------------------------------------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise RuntimeError("HF_TOKEN environment variable not found. Add it to your .env file or docker build args.")

# Download location is defined in the Dockerfile, but ensure cache path exists when
# running outside Docker.
Path(os.getenv("TRANSFORMERS_CACHE", "./models")).mkdir(parents=True, exist_ok=True)

audio_sample_rate = 16_000  # Whisper expects 16 kHz mono audio

# Load Whisper Medium
print("Loading Whisper model – this might take a minute the first time…")
processor_whisper = WhisperProcessor.from_pretrained(
    "openai/whisper-medium", use_auth_token=HF_TOKEN
)
model_whisper = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-medium", use_auth_token=HF_TOKEN
)
model_whisper.eval()

# Load M2M100
print("Loading M2M100 translation model – this might take a minute the first time…")
translator_tokenizer = M2M100Tokenizer.from_pretrained(
    "facebook/m2m100_418M", use_auth_token=HF_TOKEN
)
translator_model = M2M100ForConditionalGeneration.from_pretrained(
    "facebook/m2m100_418M", use_auth_token=HF_TOKEN
)
translator_model.eval()

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(title="Local Doctor-Patient Translator")


@app.post("/transcribe_translate")
async def transcribe_translate(
    source_lang: str = Form(..., examples=["en"]),
    target_lang: str = Form(..., examples=["es"]),
    file: UploadFile = File(...),
):
    """Convert uploaded speech to text with Whisper and translate with M2M100.

    Parameters
    ----------
    source_lang: ISO-639-1 language code of the speaker.
    target_lang: ISO-639-1 language code for the translation output.
    file: Binary audio file (webm/ogg/wav) recorded in the browser.
    """
    if file.content_type not in {"audio/webm", "audio/ogg", "audio/wav"}:
        raise HTTPException(status_code=400, detail="Unsupported audio format.")

    # ---------------------------------------------------------------------
    # Save & convert audio to 16 kHz mono WAV using ffmpeg (fast & reliable)
    # ---------------------------------------------------------------------
    raw_bytes = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".webm") as tmp_in:
        tmp_in.write(raw_bytes)
        tmp_in.flush()

        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_wav:
            cmd = [
                "ffmpeg",
                "-i",
                tmp_in.name,
                "-ar",
                str(audio_sample_rate),  # resample
                "-ac",
                "1",  # mono
                "-y",  # overwrite without prompt
                "-loglevel",
                "error",  # show errors only, capture below
                tmp_wav.name,
            ]

            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                error_msg = (
                    "ffmpeg failed to convert audio. "
                    + proc.stderr.strip().split("\n")[-1]
                )
                raise HTTPException(status_code=400, detail=error_msg)

            # Load the waveform into a tensor
            waveform, sr = torchaudio.load(tmp_wav.name)

    if sr != audio_sample_rate:
        # Safety net (shouldn't happen thanks to ffmpeg)
        waveform = torchaudio.functional.resample(waveform, sr, audio_sample_rate)

    # ---------------------------------------------------------------------
    # Speech-to-text with Whisper
    # ---------------------------------------------------------------------
    inputs = processor_whisper(
        waveform.squeeze(), sampling_rate=audio_sample_rate, return_tensors="pt"
    )
    with torch.no_grad():
        predicted_ids = model_whisper.generate(inputs.input_features)
    transcription: str = processor_whisper.batch_decode(
        predicted_ids, skip_special_tokens=True
    )[0]

    # ---------------------------------------------------------------------
    # Translate with M2M100
    # ---------------------------------------------------------------------
    translator_tokenizer.src_lang = source_lang
    encoded = translator_tokenizer(transcription, return_tensors="pt")

    forced_bos_token_id = translator_tokenizer.get_lang_id(target_lang)

    with torch.no_grad():
        generated_tokens = translator_model.generate(
            **encoded, forced_bos_token_id=forced_bos_token_id
        )
    translation: str = translator_tokenizer.batch_decode(
        generated_tokens, skip_special_tokens=True
    )[0]

    return JSONResponse(
        {"transcription": transcription, "translation": translation}
    )


# ---------------------------------------------------------------------------
# Static frontend (served from /frontend directory)
# ---------------------------------------------------------------------------
frontend_dir = Path(__file__).parent.parent / "frontend"
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend") 