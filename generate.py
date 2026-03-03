import os
import sys
import argparse
import time
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Add current directory to path for indextts imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

from indextts.infer_v2 import IndexTTS2

def main():
    parser = argparse.ArgumentParser(
        description="IndexTTS Batch Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--voice_ref", type=str, required=True, help="Path to the speaker reference audio (wav)")
    parser.add_argument("--input_dir", type=str, default="input", help="Directory containing .txt files")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory for generated wav files")
    parser.add_argument("--model_dir", type=str, default="./checkpoints", help="Model checkpoints directory")
    parser.add_argument("--fp16", action="store_true", default=False, help="Use FP16 for inference")
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose logging")
    
    args = parser.parse_args()

    # Create directories if they don't exist
    os.makedirs(args.input_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.model_dir):
        print(f"ERROR: Model directory {args.model_dir} does not exist.")
        sys.exit(1)

    # Initialize IndexTTS2
    print(f"Loading model from {args.model_dir}...")
    tts = IndexTTS2(
        model_dir=args.model_dir,
        cfg_path=os.path.join(args.model_dir, "config.yaml"),
        use_fp16=args.fp16,
    )

    # Scan for .txt files
    input_path = Path(args.input_dir)
    txt_files = list(input_path.glob("*.txt"))

    if not txt_files:
        print(f"No .txt files found in {args.input_dir}. Please add some text files.")
        return

    print(f"Found {len(txt_files)} file(s) to process.")

    for txt_file in txt_files:
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                text = f.read().strip()
            
            if not text:
                print(f"Skipping empty file: {txt_file}")
                continue

            output_wav = Path(args.output_dir) / f"{txt_file.stem}.wav"
            
            print(f"Generating audio for '{txt_file.name}' -> '{output_wav.name}'...")
            
            # Use default inference settings from webui.py logic
            tts.infer(
                spk_audio_prompt=args.voice_ref,
                text=text,
                output_path=str(output_wav),
                verbose=args.verbose
            )
            
        except Exception as e:
            print(f"Error processing {txt_file}: {e}")

    print("Batch processing complete.")

if __name__ == "__main__":
    main()
