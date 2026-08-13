"""
5-ci addım: Doldurulmuş metadata.csv faylını fine-tuning-dən
əvvəl yoxlamaq — boş sətirlər, tapılmayan fayllar, çox qısa/uzun
mətnlər üçün.

İstifadə:
    python scripts/5_validate.py
"""

import os
import sys
import pandas as pd
import librosa

from config import METADATA_FINAL, SEGMENTS_DIR, CSV_ENCODING


def main():
    if not os.path.exists(METADATA_FINAL):
        print(f"❌ '{METADATA_FINAL}' tapılmadı.")
        print("    metadata_template.csv-ni doldurub 'metadata.csv' adı ilə")
        print("    layihə kök qovluğuna yadda saxla.")
        sys.exit(1)

    df = pd.read_csv(METADATA_FINAL, encoding=CSV_ENCODING)

    problems = 0

    # 1. Boş mətn sətirləri
    empty = df[df["text"].isna() | (df["text"].astype(str).str.strip() == "")]
    if len(empty) > 0:
        print(f"⚠️  Boş mətn sahəsi olan sətir sayı: {len(empty)}")
        print(empty[["audio"]].to_string(index=False))
        problems += len(empty)
        print()

    # 2. Tapılmayan audio faylları
    missing = []
    for fname in df["audio"]:
        full_path = os.path.join(SEGMENTS_DIR, os.path.basename(str(fname)))
        if not os.path.exists(full_path):
            missing.append(fname)
    if missing:
        print(f"⚠️  Tapılmayan audio fayl sayı: {len(missing)}")
        for m in missing:
            print(f"   - {m}")
        problems += len(missing)
        print()

    # 3. Çox qısa mətn (1 sözdən az) — çox güman ki səhv/natamam
    df_clean = df.dropna(subset=["text"])
    too_short = df_clean[df_clean["text"].astype(str).str.split().str.len() < 2]
    if len(too_short) > 0:
        print(f"⚠️  Çox qısa mətn (1 sözdən az) olan sətir sayı: {len(too_short)}")
        print(too_short[["audio", "text"]].to_string(index=False))
        problems += len(too_short)
        print()

    # 4. Ümumi audio uzunluğu (fine-tuning üçün nə qədər material var)
    total_duration = 0.0
    valid_files = 0
    for fname in df["audio"]:
        full_path = os.path.join(SEGMENTS_DIR, os.path.basename(str(fname)))
        if os.path.exists(full_path):
            try:
                y, sr = librosa.load(full_path, sr=None)
                total_duration += len(y) / sr
                valid_files += 1
            except Exception:
                pass

    print("=" * 50)
    print(f"Cəmi sətir sayı: {len(df)}")
    print(f"Etibarlı (fayl mövcud) sətir sayı: {valid_files}")
    print(f"Ümumi audio uzunluğu: {total_duration / 60:.1f} dəqiqə")

    if valid_files < 80:
        print(
            "\n⚠️  Tövsiyə olunan minimum (80-150 nümunə) hələ ötülməyib. "
            "Daha çox material əlavə etmək faydalı olar."
        )

    if problems == 0:
        print("\n✅ Problem tapılmadı. Fine-tuning-ə keçə bilərsən.")
    else:
        print(f"\n⚠️  Cəmi {problems} problem tapıldı. Fine-tuning-dən əvvəl düzəlt.")


if __name__ == "__main__":
    main()
