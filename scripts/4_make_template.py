"""
4-cü addım: Manual transkripsiya üçün boş CSV şablonu yaratmaq.

İstifadə:
    python scripts/4_make_template.py

Nəticə: layihə kök qovluğunda "metadata_template.csv" faylı yaranacaq.
Bu faylı Excel/Google Sheets-də aç, "text" sütununu doldur,
sonra "metadata.csv" adı ilə (kök qovluqda) yadda saxla.
"""

import os
import sys
import pandas as pd

from config import SEGMENTS_DIR, METADATA_TEMPLATE, CSV_ENCODING


def main():
    if not os.path.exists(SEGMENTS_DIR):
        print(f"❌ '{SEGMENTS_DIR}' qovluğu tapılmadı.")
        sys.exit(1)

    files = sorted(f for f in os.listdir(SEGMENTS_DIR) if f.lower().endswith(".wav"))

    if not files:
        print(f"⚠️  '{SEGMENTS_DIR}' qovluğunda .wav fayl tapılmadı.")
        print("    Əvvəlcə 3_split.py skriptini işə sal.")
        sys.exit(0)

    df = pd.DataFrame({
        "audio": files,
        "text": [""] * len(files),
    })

    df.to_csv(METADATA_TEMPLATE, index=False, encoding=CSV_ENCODING)

    print(f"✅ {len(files)} sətirlik şablon yaradıldı: {METADATA_TEMPLATE}")
    print()
    print("Növbəti addımlar:")
    print("  1. Bu faylı Excel/Google Sheets-də aç")
    print("  2. Hər audio faylını dinlə (segments/ qovluğunda)")
    print("     və eşitdiyini 'text' sütununa yaz")
    print("  3. Doldurub qurtarandan sonra 'metadata.csv' adı ilə")
    print("     layihə kök qovluğuna yadda saxla")
    print("  4. Yoxlama üçün: python scripts/5_validate.py")


if __name__ == "__main__":
    main()
