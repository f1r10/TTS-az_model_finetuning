"""
8-ci addım: segments/ qovluğundakı bütün audio parçalarını
LocalDoc/azerbaijani-whisper-small (CTranslate2 formatında)
modeli ilə faster-whisper istifadə edərək avtomatik transkript
etmək və metadata_auto.csv faylını doldurmaq.

Bu skript AVTOMATIK transkripsiya edir. Sən sonra bunu özün
yoxlayıb (5_validate.py və ya əl ilə dinləyərək) düzəldəcəksən,
son versiyanı 'metadata.csv' adı ilə saxlayacaqsan.

İstifadə:
    python scripts/8_transcribe.py

Xüsusiyyətlər:
    - Nəticələr hər fayldan sonra dərhal CSV-yə yazılır (proses
      yarımçıq kəsilsə belə, məlumat itmir).
    - Artıq transkript olunmuş fayllar avtomatik ötürülür (resume
      dəstəyi) — skripti təkrar işə salsan, qaldığı yerdən davam edir.

Nəticə: layihə kök qovluğunda 'metadata_auto.csv'
"""

import os
import sys
import csv

from config import SEGMENTS_DIR, PROJECT_ROOT, CSV_ENCODING

CT2_MODEL_DIR = os.path.join(PROJECT_ROOT, "ct2_model")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "metadata_auto.csv")

# faster-whisper transkripsiya parametrləri
LANGUAGE = "az"
BEAM_SIZE = 5
VAD_FILTER = True  # seqment daxilində qalan susqunluq/küy hissələrini əlavə filtrləyir


def check_ct2_model():
    if not os.path.isdir(CT2_MODEL_DIR) or not os.listdir(CT2_MODEL_DIR):
        print(f"❌ '{CT2_MODEL_DIR}' tapılmadı və ya boşdur.")
        print("   Əvvəlcə: python scripts/7_convert_to_ct2.py")
        sys.exit(1)


def load_existing_results():
    """Əvvəlki (yarımçıq qalmış) nəticələri oxuyur ki, təkrar işlənməsin."""
    done = {}
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, "r", encoding=CSV_ENCODING, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                done[row["audio"]] = row["text"]
    return done


def main():
    check_ct2_model()

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("❌ 'faster_whisper' tapılmadı. Quraşdır:")
        print("   pip install faster-whisper")
        sys.exit(1)

    files = sorted(f for f in os.listdir(SEGMENTS_DIR) if f.lower().endswith(".wav"))
    if not files:
        print(f"⚠️  '{SEGMENTS_DIR}' qovluğunda .wav fayl tapılmadı.")
        sys.exit(0)

    already_done = load_existing_results()
    remaining = [f for f in files if f not in already_done]

    print(f"Cəmi fayl: {len(files)} | Artıq hazır: {len(already_done)} | Qalan: {len(remaining)}")

    if not remaining:
        print("✅ Bütün fayllar artıq transkript olunub.")
        return

    # GPU varsa istifadə et, yoxdursa CPU-ya keç
    import ctranslate2
    device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    print(f"Model yüklənir (device={device}, compute_type={compute_type})...")
    model = WhisperModel(CT2_MODEL_DIR, device=device, compute_type=compute_type)

    if device == "cuda":
        # cuBLAS/cuDNN DLL-ləri sistemdə olmaya bilər (Windows-da tez-tez rast gəlinir).
        # Bunu real transkripsiyaya keçmədən əvvəl bir test faylı ilə yoxlayırıq;
        # xəta olarsa CPU-ya avtomatik keçirik ki, proses dayanmasın.
        test_file = os.path.join(SEGMENTS_DIR, files[0])
        try:
            list(model.transcribe(test_file, language=LANGUAGE, beam_size=1)[0])
            print("✅ GPU testi uğurlu.")
        except RuntimeError as e:
            print(f"\n⚠️  GPU-da xəta baş verdi: {e}")
            print("   CPU rejiminə keçilir (int8)...")
            device = "cpu"
            compute_type = "int8"
            model = WhisperModel(CT2_MODEL_DIR, device=device, compute_type=compute_type)

    # CSV faylını aç (əlavə etmə rejimində, başlıq yalnız yoxdursa yazılır)
    file_exists = os.path.exists(OUTPUT_CSV)
    csv_file = open(OUTPUT_CSV, "a", encoding=CSV_ENCODING, newline="")
    writer = csv.writer(csv_file)
    if not file_exists:
        writer.writerow(["audio", "text"])

    try:
        for i, fname in enumerate(remaining, start=1):
            fpath = os.path.join(SEGMENTS_DIR, fname)
            print(f"[{i}/{len(remaining)}] {fname} ... ", end="", flush=True)

            segments, info = model.transcribe(
                fpath,
                language=LANGUAGE,
                beam_size=BEAM_SIZE,
                vad_filter=VAD_FILTER,
            )

            text = " ".join(seg.text.strip() for seg in segments).strip()

            writer.writerow([fname, text])
            csv_file.flush()  # dərhal diskə yaz — proses kəsilsə belə itməsin

            preview = text[:60] + ("..." if len(text) > 60 else "")
            print(f"✅ {preview}")

    except KeyboardInterrupt:
        print("\n\n⏸️  Dayandırıldı. İndiyə qədərki nəticələr saxlanıldı.")
        print(f"   Davam etmək üçün skripti yenidən işə sal: python scripts/8_transcribe.py")
    finally:
        csv_file.close()

    print(f"\n✅ Nəticələr yadda saxlanıldı: {OUTPUT_CSV}")
    print("\nNövbəti addım:")
    print(f"  1. '{os.path.basename(OUTPUT_CSV)}' faylını aç, avtomatik")
    print("     transkriptləri yoxla/düzəlt (Whisper səhv edə bilər)")
    print("  2. Düzəldilmiş versiyanı 'metadata.csv' adı ilə yadda saxla")
    print("  3. python scripts/5_validate.py")


if __name__ == "__main__":
    main()
