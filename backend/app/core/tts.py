import os
import sys
import base64
import io
import asyncio
import torch
import soundfile as sf
from pathlib import Path

# Disable torchcodec before importing torchaudio
# This forces torchaudio to use soundfile backend
os.environ.setdefault('TORCHAUDIO_USE_BACKEND_DISPATCHER', '0')

import torchaudio

# Monkey-patch torchaudio.load to handle torchcodec errors
_original_torchaudio_load = torchaudio.load

def patched_torchaudio_load(filepath, **kwargs):
    """Patched torchaudio.load that falls back to soundfile if torchcodec fails"""
    # Force soundfile backend if not specified
    if 'backend' not in kwargs:
        kwargs['backend'] = 'soundfile'
    
    try:
        return _original_torchaudio_load(filepath, **kwargs)
    except (ImportError, RuntimeError, OSError) as e:
        error_str = str(e).lower()
        if 'torchcodec' in error_str or 'ffmpeg' in error_str or 'libtorchcodec' in error_str:
            # Fallback to direct soundfile loading
            speech, sample_rate = sf.read(filepath)
            speech = torch.from_numpy(speech).float()
            
            # soundfile returns (samples,) for mono or (samples, channels) for multi-channel
            # torchaudio expects (channels, samples)
            if len(speech.shape) == 1:
                # Mono: (samples,) -> (1, samples)
                speech = speech.unsqueeze(0)
            elif len(speech.shape) == 2:
                # Multi-channel: (samples, channels) -> (channels, samples)
                speech = speech.T
            
            return speech, sample_rate
        else:
            raise

torchaudio.load = patched_torchaudio_load

# Add CosyVoice to path
backend_path = Path(__file__).parent.parent.parent
cosyvoice_path = backend_path / "CosyVoice"
if cosyvoice_path.exists():
    sys.path.insert(0, str(cosyvoice_path))
    sys.path.insert(0, str(cosyvoice_path / "third_party" / "Matcha-TTS"))

# Patch load_wav to handle torchcodec errors before importing CosyVoice
def patch_cosyvoice_load_wav():
    """Patch CosyVoice's load_wav function to use soundfile directly if torchcodec fails"""
    try:
        from cosyvoice.utils import file_utils
        
        original_load_wav = file_utils.load_wav
        
        def patched_load_wav(wav, target_sr, min_sr=16000):
            """Patched load_wav that uses soundfile directly if torchaudio fails"""
            try:
                # Try original method first
                return original_load_wav(wav, target_sr, min_sr)
            except (ImportError, RuntimeError, OSError) as e:
                error_str = str(e).lower()
                if 'torchcodec' in error_str or 'ffmpeg' in error_str or 'libtorchcodec' in error_str:
                    # Fallback to direct soundfile loading
                    try:
                        speech, sample_rate = sf.read(wav)
                        speech = torch.from_numpy(speech).float()
                        
                        # Handle stereo to mono conversion
                        if len(speech.shape) > 1:
                            speech = speech.mean(dim=-1)
                        speech = speech.unsqueeze(0)  # Add channel dimension
                        
                        # Resample if needed
                        if sample_rate != target_sr:
                            assert sample_rate >= min_sr, f'wav sample rate {sample_rate} must be greater than {min_sr}'
                            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
                            speech = resampler(speech)
                        
                        return speech
                    except Exception as fallback_error:
                        print(f"⚠️  Soundfile fallback also failed: {fallback_error}")
                        raise e  # Re-raise original error
                else:
                    raise
        
        # Apply the patch
        file_utils.load_wav = patched_load_wav
        return True
    except Exception as e:
        print(f"⚠️  Could not patch load_wav: {e}")
        return False

try:
    from cosyvoice.cli.cosyvoice import AutoModel
    # Patch after import
    patch_cosyvoice_load_wav()
except ImportError as e:
    print(f"⚠️  CosyVoice not available: {e}")
    print("Please ensure CosyVoice is cloned and dependencies are installed.")
    AutoModel = None

# Model configuration
MODEL_DIR_ENV = os.getenv("COSYVOICE_MODEL_DIR", "pretrained_models/Fun-CosyVoice3-0.5B")
# Resolve model directory relative to backend
if not os.path.isabs(MODEL_DIR_ENV):
    MODEL_DIR = str(backend_path / MODEL_DIR_ENV)
else:
    MODEL_DIR = MODEL_DIR_ENV
SPEAKER_ID = os.getenv("COSYVOICE_SPEAKER_ID", "中文女")  # Default speaker for SFT model
USE_SFT = os.getenv("COSYVOICE_USE_SFT", "false").lower() == "true"

# Global model instance
_model = None
_model_lock = asyncio.Lock()

def _load_model():
    """Load CosyVoice model (synchronous)"""
    global _model
    if _model is None:
        if AutoModel is None:
            raise ImportError("CosyVoice AutoModel is not available. Please install CosyVoice dependencies.")
        print(f"🔄 Loading CosyVoice model from {MODEL_DIR}...")
        try:
            if USE_SFT:
                # Use SFT model (requires speaker ID)
                model_path = MODEL_DIR if "SFT" in MODEL_DIR else MODEL_DIR.replace("Fun-CosyVoice3-0.5B", "CosyVoice-300M-SFT")
                _model = AutoModel(model_dir=model_path)
                print(f"✅ CosyVoice SFT model loaded. Available speakers: {_model.list_available_spks()}")
            else:
                # Use CosyVoice3 for zero-shot (recommended)
                _model = AutoModel(model_dir=MODEL_DIR)
                print(f"✅ CosyVoice model loaded.")
        except Exception as e:
            print(f"❌ Failed to load CosyVoice model: {e}")
            raise
    return _model

async def _get_model():
    """Get or load model (async-safe)"""
    async with _model_lock:
        if _model is None:
            # Run model loading in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _load_model)
        return _model

async def text_to_speech(text: str) -> str:
    """
    Convert text to speech using CosyVoice.
    
    Args:
        text: Text to synthesize
        
    Returns:
        Base64-encoded WAV audio data, or empty string on error
    """
    if not text or not text.strip():
        return ""
    
    try:
        model = await _get_model()
        
        # Run inference in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        if USE_SFT:
            # Use SFT inference (faster, requires speaker ID)
            def _synthesize_sft():
                audio_chunks = []
                for result in model.inference_sft(text, SPEAKER_ID, stream=False):
                    audio_chunks.append(result['tts_speech'])
                if audio_chunks:
                    # Concatenate all chunks
                    audio = torch.cat(audio_chunks, dim=1)
                    return audio, model.sample_rate
                return None, None
            
            audio, sample_rate = await loop.run_in_executor(None, _synthesize_sft)
        else:
            # Use zero-shot inference (CosyVoice3 recommended)
            # For zero-shot, we need a prompt text and prompt audio
            # Using default prompt for simplicity
            prompt_text = "希望你以后能够做的比我还好呦。"
            prompt_wav = str(cosyvoice_path / "asset" / "zero_shot_prompt.wav")
            
            # Fallback if prompt file doesn't exist
            if not os.path.exists(prompt_wav):
                prompt_wav = None
            
            def _synthesize_zero_shot():
                audio_chunks = []
                # CosyVoice3 requires <|endofprompt|> token
                full_prompt = f"You are a helpful assistant.<|endofprompt|>{prompt_text}"
                
                if prompt_wav and os.path.exists(prompt_wav):
                    for result in model.inference_zero_shot(
                        text, 
                        full_prompt, 
                        prompt_wav, 
                        stream=False
                    ):
                        audio_chunks.append(result['tts_speech'])
                else:
                    # Fallback: try without prompt audio (may not work for all models)
                    try:
                        for result in model.inference_zero_shot(
                            text,
                            full_prompt,
                            "",
                            stream=False
                        ):
                            audio_chunks.append(result['tts_speech'])
                    except:
                        # Last resort: try SFT if available
                        if hasattr(model, 'inference_sft'):
                            speakers = model.list_available_spks()
                            speaker = speakers[0] if speakers else "中文女"
                            for result in model.inference_sft(text, speaker, stream=False):
                                audio_chunks.append(result['tts_speech'])
                
                if audio_chunks:
                    audio = torch.cat(audio_chunks, dim=1)
                    return audio, model.sample_rate
                return None, None
            
            audio, sample_rate = await loop.run_in_executor(None, _synthesize_zero_shot)
        
        if audio is None:
            print("❌ TTS: No audio generated")
            return ""
        
        # Convert to WAV bytes
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio, sample_rate, format="wav")
        buffer.seek(0)
        wav_bytes = buffer.read()
        
        # Encode to base64
        audio_base64 = base64.b64encode(wav_bytes).decode('utf-8')
        return audio_base64
        
    except Exception as e:
        print(f"❌ TTS Error: {e}")
        import traceback
        traceback.print_exc()
        return ""

