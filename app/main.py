from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import List

import ray
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Ray Actor for Whisper (Speech-to-Text) - Runs on CPU
# ---------------------------------------------------------------------------
@ray.remote
class WhisperActor:
    def __init__(self, hf_token: str):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        
        print("Loading Whisper model on this node...")
        self.processor = WhisperProcessor.from_pretrained(
            "openai/whisper-medium", token=hf_token
        )
        self.model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-medium", token=hf_token
        )
        self.model.eval()
        self.audio_sample_rate = 16_000
        print("Whisper model loaded successfully!")

    def transcribe(self, wav_bytes: bytes) -> str:
        import torch
        import torchaudio
        import tempfile
        
        # Save bytes to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name
        
        try:
            # Load audio
            waveform, sr = torchaudio.load(tmp_path)
            
            # Resample if needed
            if sr != self.audio_sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.audio_sample_rate)
            
            # Process
            inputs = self.processor(
                waveform.squeeze(), 
                sampling_rate=self.audio_sample_rate, 
                return_tensors="pt"
            )
            
            # Generate transcription
            with torch.no_grad():
                predicted_ids = self.model.generate(inputs.input_features)
            
            transcription = self.processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]
            
            return transcription
            
        finally:
            # Clean up temp file
            os.unlink(tmp_path)

# ---------------------------------------------------------------------------
# Ray Actor for M2M100 (Translation) - Runs on GPU if available
# ---------------------------------------------------------------------------
@ray.remote
class M2M100Actor:
    def __init__(self, hf_token: str):
        import torch
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
        
        print("Loading M2M100 translation model on this node...")
        self.tokenizer = M2M100Tokenizer.from_pretrained(
            "facebook/m2m100_418M", token=hf_token
        )
        self.model = M2M100ForConditionalGeneration.from_pretrained(
            "facebook/m2m100_418M", token=hf_token
        )
        self.model.eval()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print("M2M100 model loaded on GPU!")
        else:
            print("M2M100 model loaded on CPU!")

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        import torch
        
        self.tokenizer.src_lang = source_lang
        encoded = self.tokenizer(text, return_tensors="pt")
        
        # Move to GPU if available
        if torch.cuda.is_available():
            encoded = {k: v.cuda() for k, v in encoded.items()}
        
        forced_bos_token_id = self.tokenizer.get_lang_id(target_lang)
        
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **encoded, forced_bos_token_id=forced_bos_token_id
            )
        
        translation = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )[0]
        
        return translation

# ---------------------------------------------------------------------------
# Initialize Ray and create actors
# ---------------------------------------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise RuntimeError("HF_TOKEN environment variable not found. Add it to your .env file or export it.")

# Initialize Ray
# If running on head node, Ray will already be initialized
# If running standalone, this will start a local Ray instance
try:
    ray.init(address="auto")
    print("Connected to existing Ray cluster")
except:
    ray.init()
    print("Started local Ray instance")

# Create actors - Ray will place them on appropriate nodes
print("Creating Whisper actor (CPU)...")
whisper_actor = WhisperActor.remote(HF_TOKEN)

print("Creating M2M100 actor (GPU if available)...")
# Request GPU if available, otherwise fall back to CPU
if ray.cluster_resources().get("GPU", 0) > 0:
    m2m100_actor = M2M100Actor.options(num_gpus=1).remote(HF_TOKEN)
else:
    m2m100_actor = M2M100Actor.remote(HF_TOKEN)

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(title="Local Doctor-Patient Translator (Ray-Enabled)")

@app.post("/transcribe_translate")
async def transcribe_translate(
    source_lang: str = Form(..., examples=["en"]),
    target_lang: str = Form(..., examples=["es"]),
    file: UploadFile = File(...),
):
    """Convert uploaded speech to text with Whisper and translate with M2M100.
    
    This now uses Ray to distribute the workload across multiple machines.
    
    Parameters
    ----------
    source_lang: ISO-639-1 language code of the speaker.
    target_lang: ISO-639-1 language code for the translation output.
    file: Binary audio file (webm/ogg/wav) recorded in the browser.
    """
    if file.content_type not in {"audio/webm", "audio/ogg", "audio/wav"}:
        raise HTTPException(status_code=400, detail="Unsupported audio format.")

    # ---------------------------------------------------------------------
    # Save & convert audio to 16 kHz mono WAV using ffmpeg
    # ---------------------------------------------------------------------
    raw_bytes = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp_in:
        tmp_in.write(raw_bytes)
        tmp_in.flush()
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            cmd = [
                "ffmpeg",
                "-i", tmp_in.name,
                "-ar", "16000",  # resample to 16kHz
                "-ac", "1",      # mono
                "-y",            # overwrite without prompt
                "-loglevel", "error",
                tmp_wav.name,
            ]
            
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                os.unlink(tmp_in.name)
                error_msg = f"ffmpeg failed to convert audio. {proc.stderr.strip()}"
                raise HTTPException(status_code=400, detail=error_msg)
            
            # Read the converted WAV file
            with open(tmp_wav.name, "rb") as f:
                wav_bytes = f.read()
    
    # Clean up temp files
    os.unlink(tmp_in.name)
    os.unlink(tmp_wav.name)
    
    try:
        # ---------------------------------------------------------------------
        # Distributed inference using Ray
        # ---------------------------------------------------------------------
        # 1. Transcribe audio (will run on CPU node)
        transcription_ref = whisper_actor.transcribe.remote(wav_bytes)
        transcription = ray.get(transcription_ref)
        
        # 2. Translate text (will run on GPU node if available)
        translation_ref = m2m100_actor.translate.remote(
            transcription, source_lang, target_lang
        )
        translation = ray.get(translation_ref)
        
        return JSONResponse(
            {"transcription": transcription, "translation": translation}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/ray-status")
async def ray_status():
    """Check Ray cluster status and resource availability."""
    try:
        resources = ray.cluster_resources()
        available = ray.available_resources()
        nodes = []
        
        for node in ray.nodes():
            if node["Alive"]:
                nodes.append({
                    "node_id": node["NodeID"],
                    "address": node["NodeManagerAddress"],
                    "cpu": node["Resources"].get("CPU", 0),
                    "gpu": node["Resources"].get("GPU", 0),
                })
        
        return {
            "connected": True,
            "total_resources": dict(resources),
            "available_resources": dict(available),
            "nodes": nodes
        }
    except Exception as e:
        return {"connected": False, "error": str(e)}

# ---------------------------------------------------------------------------
# Static frontend (served from /frontend directory)
# ---------------------------------------------------------------------------
frontend_dir = Path(__file__).parent.parent / "frontend"
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend") 