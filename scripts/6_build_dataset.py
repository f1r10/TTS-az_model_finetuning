"""
6-cı addım: metadata.csv-ni HuggingFace 'datasets' formatına çevirmək.
Bu, fine-tuning skriptinin (finetune-hf-vits) birbaşa istifadə
edə biləcəyi formatdır.

İstifadə:
    python scripts/6_build_dataset.py

Nəticə: layihə kök qovluğunda "hf_dataset/" qovluğu yaranacaq.
Bunu fine-tuning əmrinə --dataset_name kimi ötürə bilərsən.

Tələb olunan paket: pip install datasets
"""

import os
import sys
import pandas as pd

from config import METADATA_FINAL, SEGMENTS_DIR, PROJECT_ROOT, CSV_ENCODING

try:
    from datasets import Dataset, Audio
except ImportError:
    print("❌ 'datasets' paketi tapılmadı. Quraşdır:")
    print("   pip install datasets")
    sys.exit(1)


def main():
    if not os.path.exists(METADATA_FINAL):
        print(f"❌ '{METADATA_FINAL}' tapılmadı. Əvvəlcə 5_validate.py ilə yoxla.")
        sys.exit(1)

    df = pd.read_csv(METADATA_FINAL, encoding=CSV_ENCODING)
    df = df.dropna(subset=["text"])
    df = df[df["text"].astype(str).str.strip() != ""]

    # audio sütununu tam yola çeviririk
    df["audio"] = df["audio"].apply(
        lambda f: os.path.join(SEGMENTS_DIR, os.path.basename(str(f)))
    )

    # yalnız mövcud faylları saxla
    df = df[df["audio"].apply(os.path.exists)]

    print(f"{len(df)} keçərli sətir tapıldı.")

    dataset = Dataset.from_pandas(df.reset_index(drop=True))
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    out_dir = os.path.join(PROJECT_ROOT, "hf_dataset")
    dataset.save_to_disk(out_dir)

    print(f"✅ Dataset hazırlandı: {out_dir}")
    print("\nFine-tuning əmrində bunu istifadə et:")
    print(f"  --dataset_name {out_dir}")


if __name__ == "__main__":
    main()
