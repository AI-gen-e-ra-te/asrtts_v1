import os
from faster_whisper import WhisperModel

# 模型大小：base, small, medium, large-v3
# 建议先用 base 测试，速度快
MODEL_SIZE = "base" 
device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

print(f"🔄 Loading Whisper model ({MODEL_SIZE}) on {device}...")
try:
    # compute_type="int8" 可以加速且省显存
    model = WhisperModel(MODEL_SIZE, device=device, compute_type="int8")
    print("✅ Whisper model loaded.")
except Exception as e:
    print(f"❌ Failed to load Whisper: {e}")
    model = None

def transcribe_audio(file_path: str) -> str:
    if not model:
        return "Error: ASR model not loaded."
    
    segments, info = model.transcribe(file_path, beam_size=5, language="zh")
    
    text = ""
    for segment in segments:
        text += segment.text
    
    return text.strip()