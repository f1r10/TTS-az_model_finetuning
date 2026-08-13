"""
2-ci addım: Bütün audio fayllarını eyni formata gətirmək
(sample rate + mono). Fine-tuning üçün bu şərtdir.

İstifadə:
    python scripts/2_normalize.py

Nəticə: normalized/ qovluğuna eyni adla, amma standart
formatda (16kHz, mono) fayllar yazılacaq.
"""

import os
import subprocess
import sys

from config import RAW_AUDIO_DIR, NORMALIZED_DIR, TARGET_SAMPLE_RATE, TARGET_CHANNELS


def check_ffmpeg():
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ 'ffmpeg' tapılmadı. Quraşdır:")
        print("   Mac:     brew install ffmpeg")
        print("   Linux:   sudo apt install ffmpeg")
        print("   Windows: https://ffmpeg.org/download.html")
        sys.exit(1)


def main():
    check_ffmpeg()
    os.makedirs(NORMALIZED_DIR, exist_ok=True)

    files = [f for f in os.listdir(RAW_AUDIO_DIR) if f.lower().endswith(".wav")]

    if not files:
        print(f"⚠️  '{RAW_AUDIO_DIR}' qovluğunda .wav fayl tapılmadı.")
        print("    Əvvəlcə 1_download.py skriptini işə sal.")
        sys.exit(0)

    print(f"{len(files)} fayl normallaşdırılacaq...\n")

    for fname in sorted(files):
        in_path = os.path.join(RAW_AUDIO_DIR, fname)
        out_path = os.path.join(NORMALIZED_DIR, fname)

        result = subprocess.run(
            [
                "ffmpeg",
                "-i", in_path,
                "-ar", str(TARGET_SAMPLE_RATE),
                "-ac", str(TARGET_CHANNELS),
                out_path,
                "-y",
                "-loglevel", "error",
            ]
        )

        if result.returncode == 0:
            print(f"✅ {fname}")
        else:
            print(f"❌ {fname} — xəta baş verdi")

    print(f"\nBitdi. Fayllar burada: {NORMALIZED_DIR}")


if __name__ == "__main__":
    main()
