"""
1-ci addım: YouTube-dan audio endirmək.

İstifadə:
    1. Layihə kök qovluğunda "urls.txt" faylı yarat
       (yoxdursa bu skript nümunə fayl yaradacaq).
    2. Hər sətirdə bir YouTube linki yaz.
    3. Bu skripti işə sal:  python scripts/1_download.py

Nəticə: raw_audio/ qovluğuna video1.wav, video2.wav ... şəklində
audio faylları yüklənəcək.
"""

import os
import subprocess
import sys

from config import RAW_AUDIO_DIR, URLS_FILE


def ensure_urls_file():
    if not os.path.exists(URLS_FILE):
        with open(URLS_FILE, "w", encoding="utf-8") as f:
            f.write("# Hər sətirdə bir YouTube linki yaz, bu sətirləri sil\n")
            f.write("# https://www.youtube.com/watch?v=XXXXXXXXXXX\n")
        print(f"⚠️  '{URLS_FILE}' tapılmadı, nümunə fayl yaradıldı.")
        print("    Linklərini əlavə edib skripti yenidən işə sal.")
        sys.exit(0)


def read_urls():
    urls = []
    with open(URLS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    return urls


def check_yt_dlp():
    try:
        subprocess.run(
            ["yt-dlp", "--version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ 'yt-dlp' tapılmadı. Əvvəlcə quraşdır:")
        print("   pip install yt-dlp")
        sys.exit(1)


def main():
    check_yt_dlp()
    ensure_urls_file()
    urls = read_urls()

    if not urls:
        print(f"⚠️  '{URLS_FILE}' faylında link tapılmadı. Linklər əlavə et.")
        sys.exit(0)

    os.makedirs(RAW_AUDIO_DIR, exist_ok=True)
    print(f"{len(urls)} link tapıldı, endirmə başlayır...\n")

    for i, url in enumerate(urls, start=1):
        out_template = os.path.join(RAW_AUDIO_DIR, f"video{i}.%(ext)s")
        print(f"[{i}/{len(urls)}] Endirilir: {url}")
        result = subprocess.run(
            [
                "yt-dlp",
                "-x",
                "--audio-format", "wav",
                "--audio-quality", "0",
                "-o", out_template,
                url,
            ]
        )
        if result.returncode != 0:
            print(f"   ⚠️  Xəta baş verdi, bu link atlanıldı: {url}")
        else:
            print(f"   ✅ video{i}.wav yadda saxlanıldı")

    print(f"\nBitdi. Fayllar burada: {RAW_AUDIO_DIR}")


if __name__ == "__main__":
    main()
