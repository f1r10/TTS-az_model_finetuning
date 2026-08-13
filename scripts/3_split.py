"""
3-cü addım: Normallaşdırılmış audio fayllarını kiçik parçalara bölmək.
Transkripsiya ETMİR — sadəcə susqunluğa görə kəsir.

İstifadə:
    python scripts/3_split.py

Nəticə: segments/ qovluğuna clip_0001.wav, clip_0002.wav ...
şəklində kiçik audio parçaları yazılacaq. Bu fayllar sonra
metadata_template.csv-də sırayla görünəcək.

Qeyd: Əgər parçalar çox böyük çıxırsa (az bölünürsə),
config.py-da SILENCE_THRESH_DB dəyərini artır (məs. -35).
Əgər çox kiçik/çox sayda çıxırsa, azalt (məs. -45).
"""

import os
import sys

from pydub import AudioSegment
from pydub.silence import split_on_silence

from config import (
    NORMALIZED_DIR,
    SEGMENTS_DIR,
    MIN_CLIP_LEN_MS,
    MAX_CLIP_LEN_MS,
    SILENCE_THRESH_DB,
    MIN_SILENCE_LEN_MS,
    KEEP_SILENCE_MS,
)


def main():
    os.makedirs(SEGMENTS_DIR, exist_ok=True)

    files = [f for f in os.listdir(NORMALIZED_DIR) if f.lower().endswith(".wav")]

    if not files:
        print(f"⚠️  '{NORMALIZED_DIR}' qovluğunda .wav fayl tapılmadı.")
        print("    Əvvəlcə 2_normalize.py skriptini işə sal.")
        sys.exit(0)

    clip_counter = 1
    total_kept = 0
    total_rejected = 0

    for fname in sorted(files):
        print(f"\nEmal olunur: {fname}")
        audio = AudioSegment.from_wav(os.path.join(NORMALIZED_DIR, fname))

        chunks = split_on_silence(
            audio,
            min_silence_len=MIN_SILENCE_LEN_MS,
            silence_thresh=SILENCE_THRESH_DB,
            keep_silence=KEEP_SILENCE_MS,
        )

        for chunk in chunks:
            if len(chunk) < MIN_CLIP_LEN_MS or len(chunk) > MAX_CLIP_LEN_MS:
                total_rejected += 1
                continue

            out_name = f"clip_{clip_counter:04d}.wav"
            out_path = os.path.join(SEGMENTS_DIR, out_name)
            chunk.export(out_path, format="wav")

            clip_counter += 1
            total_kept += 1

        print(f"   -> {len(chunks)} xam parça tapıldı")

    print(f"\n{'=' * 50}")
    print(f"Cəmi saxlanılan parça: {total_kept}")
    print(f"Uzunluq şərtinə görə atılan: {total_rejected}")
    print(f"Fayllar burada: {SEGMENTS_DIR}")

    if total_kept < 80:
        print(
            "\n⚠️  Qeyd: 80-dən az parça var. Fine-tuning üçün minimum "
            "80-150 nümunə tövsiyə olunur. Daha çox material "
            "əlavə etməyi düşün."
        )


if __name__ == "__main__":
    main()
