"""
Setup script for CosyVoice TTS integration.
This script helps set up the CosyVoice environment and download models.
"""
import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, check=True):
    """Run a shell command"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, check=check)
    return result.returncode == 0

def main():
    print("=" * 60)
    print("CosyVoice Setup Script")
    print("=" * 60)
    
    # Check if CosyVoice is cloned
    cosyvoice_path = Path(__file__).parent / "CosyVoice"
    if not cosyvoice_path.exists():
        print("\n❌ CosyVoice repository not found!")
        print("Please run: git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git")
        return 1
    
    print(f"\n✅ CosyVoice found at: {cosyvoice_path}")
    
    # Check submodules
    print("\n📦 Checking submodules...")
    os.chdir(cosyvoice_path)
    run_command("git submodule update --init --recursive", check=False)
    
    # Install requirements
    print("\n📥 Installing CosyVoice requirements...")
    requirements_file = cosyvoice_path / "requirements.txt"
    if requirements_file.exists():
        # Use pip with Chinese mirror for faster download
        cmd = f"pip install -r {requirements_file} -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com"
        if not run_command(cmd, check=False):
            print("⚠️  Some packages may have failed to install. Continuing...")
    else:
        print("⚠️  requirements.txt not found in CosyVoice directory")
    
    print("\n✅ Setup complete!")
    print("\n📝 Next steps:")
    print("1. Run download_models.py to download pretrained models")
    print("2. Set COSYVOICE_MODEL_DIR environment variable to point to your model directory")
    print("3. Optionally set COSYVOICE_USE_SFT=true to use SFT model instead of zero-shot")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

