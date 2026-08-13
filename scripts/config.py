"""
Layihənin bütün tənzimlənə bilən parametrləri burada saxlanılır.
Skriptlərin heç birini dəyişmədən, sadəcə bu faylı redaktə edərək
davranışı tənzimləyə bilərsən.
"""

import os

# ------------------------------------------------------------------
# QOVLUQ YOLLARI (skriptlər bu fayla nəzərən yerləşdiyi üçün avtomatik hesablanır)
# ------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_AUDIO_DIR = os.path.join(PROJECT_ROOT, "raw_audio")
NORMALIZED_DIR = os.path.join(PROJECT_ROOT, "normalized")
SEGMENTS_DIR = os.path.join(PROJECT_ROOT, "segments")
METADATA_TEMPLATE = os.path.join(PROJECT_ROOT, "metadata_template.csv")
METADATA_FINAL = os.path.join(PROJECT_ROOT, "metadata.csv")

# ------------------------------------------------------------------
# YOUTUBE ENDİRMƏ AYARLARI
# ------------------------------------------------------------------
# Hər sətirdə bir YouTube linki olan mətn faylı.
# Ya birbaşa bu faylı redaktə et, ya da urls.txt yolunu dəyiş.
URLS_FILE = os.path.join(PROJECT_ROOT, "urls.txt")

# ------------------------------------------------------------------
# NORMALLAŞDIRMA AYARLARI (fine-tuning modelləri üçün standart)
# ------------------------------------------------------------------
TARGET_SAMPLE_RATE = 16000   # MMS/VITS üçün standart. XTTS üçün 22050 lazım ola bilər.
TARGET_CHANNELS = 1          # mono

# ------------------------------------------------------------------
# BÖLMƏ (SPLITTING) AYARLARI
# ------------------------------------------------------------------
MIN_CLIP_LEN_MS = 2000        # minimum parça uzunluğu (2 saniyə)
MAX_CLIP_LEN_MS = 12000       # maksimum parça uzunluğu (12 saniyə)
SILENCE_THRESH_DB = -40       # susqunluq həssaslığı (dB) — parçalar çox böyükdürsə artır, çox kiçikdirsə azalt
MIN_SILENCE_LEN_MS = 400      # bu qədər susqunluq = bölmə nöqtəsi hesab olunur
KEEP_SILENCE_MS = 200         # hər parçanın əvvəl/sonunda saxlanılan kiçik susqunluq (kəskin kəsilməsin deyə)

# ------------------------------------------------------------------
# CSV KODLAŞDIRMASI
# ------------------------------------------------------------------
# utf-8-sig Excel-də Azərbaycan hərflərinin (ə, ş, ç, ğ, ı, ö, ü) düzgün
# görünməsi üçündür. Bunu dəyişmə.
CSV_ENCODING = "utf-8-sig"
