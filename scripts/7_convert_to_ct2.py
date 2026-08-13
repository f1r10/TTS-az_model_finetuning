"""
7-ci addım: LocalDoc/azerbaijani-whisper-small modelini HuggingFace
formatından CTranslate2 formatına çevirmək (faster-whisper üçün lazımdır).

Qeyd: Bu model repo-sunda standart 'preprocessor_config.json' faylı YOXDUR,
əvəzinə qeyri-standart adlı 'processor_config.json' var. Bu skript bunu
avtomatik aşkarlayıb, çatışmayan 'preprocessor_config.json' faylını
əsas whisper-small modelindən əldə edərək düzəldir — əks halda
ct2-transformers-converter xəta verə bilər və ya faster-whisper
sonradan feature-extractor konfiqurasiyasını tapa bilməz.

İstifadə:
    python scripts/7_convert_to_ct2.py

Nəticə: ct2_model/ qovluğunda CTranslate2 formatında model
(faster-whisper birbaşa bunu istifadə edəcək).
"""

import os
import shutil
import subprocess
import sys

from config import PROJECT_ROOT

HF_MODEL_ID = "LocalDoc/azerbaijani-whisper-small"
BASE_MODEL_ID = "openai/whisper-small"  # preprocessor_config.json ehtiyat mənbəyi

LOCAL_HF_DIR = os.path.join(PROJECT_ROOT, "hf_model_raw")
CT2_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "ct2_model")


def check_dependencies():
    missing = []
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        missing.append("huggingface_hub")
    try:
        import transformers  # noqa: F401
    except ImportError:
        missing.append("transformers")
    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append("torch")

    try:
        subprocess.run(
            ["ct2-transformers-converter", "--help"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        missing.append("ctranslate2 (ct2-transformers-converter tapılmadı)")

    if missing:
        print("❌ Çatışmayan paketlər:", ", ".join(missing))
        print("   Quraşdır: pip install -r requirements.txt")
        sys.exit(1)


def download_hf_model():
    from huggingface_hub import snapshot_download

    print(f"İndirilir: {HF_MODEL_ID} ...")
    local_dir = snapshot_download(repo_id=HF_MODEL_ID, local_dir=LOCAL_HF_DIR)
    print(f"✅ Model endirildi: {local_dir}")
    return local_dir


def fix_missing_preprocessor_config(local_dir):
    """
    Bu repoda 'preprocessor_config.json' əvəzinə 'processor_config.json'
    var. CTranslate2 converter və faster-whisper standart adı gözləyir.
    Çatışmırsa, əsas whisper-small modelindən əldə edib əlavə edirik
    (feature-extractor parametrləri fine-tuning zamanı dəyişməyib).
    """
    target_path = os.path.join(local_dir, "preprocessor_config.json")
    nonstandard_path = os.path.join(local_dir, "processor_config.json")

    if os.path.exists(target_path):
        print("✅ 'preprocessor_config.json' artıq mövcuddur.")
        return

    if os.path.exists(nonstandard_path):
        print(
            "⚠️  'preprocessor_config.json' tapılmadı, əvəzinə "
            "'processor_config.json' var — bu qeyri-standartdır."
        )

    print(f"   '{BASE_MODEL_ID}' modelindən preprocessor_config.json əldə edilir...")
    from huggingface_hub import hf_hub_download

    fetched_path = hf_hub_download(
        repo_id=BASE_MODEL_ID,
        filename="preprocessor_config.json",
    )
    shutil.copy(fetched_path, target_path)
    print(f"✅ 'preprocessor_config.json' əlavə edildi: {target_path}")


def convert_to_ct2():
    if os.path.exists(CT2_OUTPUT_DIR) and os.listdir(CT2_OUTPUT_DIR):
        print(f"⚠️  '{CT2_OUTPUT_DIR}' artıq mövcuddur və boş deyil.")
        answer = input("   Yenidən konvertasiya edilsin? Üzərinə yazılacaq (b/y) [y]: ").strip().lower()
        if answer == "b":
            print("Konvertasiya ləğv edildi.")
            return
        shutil.rmtree(CT2_OUTPUT_DIR)

    print("\nCTranslate2 formatına konvertasiya edilir (bir neçə dəqiqə çəkə bilər)...")
    result = subprocess.run(
        [
            "ct2-transformers-converter",
            "--model", LOCAL_HF_DIR,
            "--output_dir", CT2_OUTPUT_DIR,
            "--copy_files", "tokenizer.json", "preprocessor_config.json",
            "--quantization", "float16",
        ]
    )

    if result.returncode != 0:
        print("\n⚠️  'float16' ilə xəta baş verdisə (GPU yoxdursa gözlənilir),")
        print("    'int8' ilə yenidən sınanılır (CPU üçün uyğundur)...")
        result = subprocess.run(
            [
                "ct2-transformers-converter",
                "--model", LOCAL_HF_DIR,
                "--output_dir", CT2_OUTPUT_DIR,
                "--copy_files", "tokenizer.json", "preprocessor_config.json",
                "--quantization", "int8",
            ]
        )

    if result.returncode == 0:
        print(f"\n✅ Model hazırdır: {CT2_OUTPUT_DIR}")
    else:
        print("\n❌ Konvertasiya uğursuz oldu. Yuxarıdakı xəta mesajına bax.")
        sys.exit(1)


def main():
    check_dependencies()
    local_dir = download_hf_model()
    fix_missing_preprocessor_config(local_dir)
    convert_to_ct2()


if __name__ == "__main__":
    main()
