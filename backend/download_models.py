"""
Download script for CosyVoice pretrained models.
Supports both ModelScope (for Chinese users) and HuggingFace (for overseas users).
"""
import os
import sys
from pathlib import Path

def download_modelscope_models():
    """Download models using ModelScope SDK (recommended for Chinese users)"""
    print("=" * 60)
    print("Downloading CosyVoice models using ModelScope SDK")
    print("=" * 60)
    
    try:
        from modelscope import snapshot_download
        
        models_dir = Path(__file__).parent / "pretrained_models"
        models_dir.mkdir(exist_ok=True)
        
        models = [
            ('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', 'Fun-CosyVoice3-0.5B'),
            ('iic/CosyVoice2-0.5B', 'CosyVoice2-0.5B'),
            # ('iic/CosyVoice-300M', 'CosyVoice-300M'),
            # ('iic/CosyVoice-300M-SFT', 'CosyVoice-300M-SFT'),
            # ('iic/CosyVoice-300M-Instruct', 'CosyVoice-300M-Instruct'),
            # ('iic/CosyVoice-ttsfrd', 'CosyVoice-ttsfrd'),
        ]
        
        for model_id, local_name in models:
            local_dir = models_dir / local_name
            if local_dir.exists():
                print(f"⏭️  {local_name} already exists, skipping...")
                continue
            
            print(f"\n📥 Downloading {local_name}...")
            try:
                snapshot_download(model_id, local_dir=str(local_dir))
                print(f"✅ {local_name} downloaded successfully!")
            except Exception as e:
                print(f"❌ Failed to download {local_name}: {e}")
                continue
        
        print("\n✅ Model download complete!")
        return True
        
    except ImportError:
        print("❌ ModelScope not installed. Install with: pip install modelscope")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def download_huggingface_models():
    """Download models using HuggingFace SDK (for overseas users)"""
    print("=" * 60)
    print("Downloading CosyVoice models using HuggingFace SDK")
    print("=" * 60)
    
    try:
        from huggingface_hub import snapshot_download
        
        models_dir = Path(__file__).parent / "pretrained_models"
        models_dir.mkdir(exist_ok=True)
        
        models = [
            ('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', 'Fun-CosyVoice3-0.5B'),
            ('FunAudioLLM/CosyVoice2-0.5B', 'CosyVoice2-0.5B'),
            # ('FunAudioLLM/CosyVoice-300M', 'CosyVoice-300M'),
            # ('FunAudioLLM/CosyVoice-300M-SFT', 'CosyVoice-300M-SFT'),
            # ('FunAudioLLM/CosyVoice-300M-Instruct', 'CosyVoice-300M-Instruct'),
            # ('FunAudioLLM/CosyVoice-ttsfrd', 'CosyVoice-ttsfrd'),
        ]
        
        for model_id, local_name in models:
            local_dir = models_dir / local_name
            if local_dir.exists():
                print(f"⏭️  {local_name} already exists, skipping...")
                continue
            
            print(f"\n📥 Downloading {local_name}...")
            try:
                snapshot_download(model_id, local_dir=str(local_dir))
                print(f"✅ {local_name} downloaded successfully!")
            except Exception as e:
                print(f"❌ Failed to download {local_name}: {e}")
                continue
        
        print("\n✅ Model download complete!")
        return True
        
    except ImportError:
        print("❌ HuggingFace Hub not installed. Install with: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("\nChoose download method:")
    print("1. ModelScope (recommended for Chinese users)")
    print("2. HuggingFace (for overseas users)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        success = download_modelscope_models()
    elif choice == "2":
        success = download_huggingface_models()
    else:
        print("❌ Invalid choice")
        return 1
    
    if success:
        print("\n📝 Next steps:")
        print("1. Set COSYVOICE_MODEL_DIR environment variable:")
        print("   export COSYVOICE_MODEL_DIR=pretrained_models/Fun-CosyVoice3-0.5B")
        print("2. Or edit .env file to add:")
        print("   COSYVOICE_MODEL_DIR=pretrained_models/Fun-CosyVoice3-0.5B")
        print("3. Optionally install ttsfrd for better text normalization:")
        print("   cd pretrained_models/CosyVoice-ttsfrd/")
        print("   unzip resource.zip -d .")
        print("   pip install ttsfrd_dependency-0.1-py3-none-any.whl")
        print("   pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl")
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())

