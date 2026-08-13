# Azərbaycan Dilli TTS Fine-Tuning Layihəsi

Bu layihə real bir diktorun səsindən istifadə edərək, Azərbaycan dilində
təbii səslənən Text-to-Speech (TTS) modeli qurmaq üçün lazım olan bütün
data hazırlığı pipeline-ını əhatə edir: audio endirmə → normallaşdırma →
kiçik parçalara bölmə → manual transkripsiya → fine-tuning üçün format.

Model olaraq [MMS-TTS](https://huggingface.co/facebook/mms-tts-azj-script_latin)
(Meta-nın çoxdilli TTS modeli, Azərbaycan dilini artıq dəstəkləyir) fine-tune
ediləcək — [`ylacombe/finetune-hf-vits`](https://github.com/ylacombe/finetune-hf-vits)
alətindən istifadə etməklə.

---
---

## Qovluq strukturu

```
az_voice_project/
│
├── raw_audio/              ← YouTube-dan endirilən orijinal fayllar (video1.wav, video2.wav, ...)
├── normalized/              ← 16kHz/mono formata çevrilmiş versiya
├── segments/                ← Kiçik parçalara bölünmüş audio (clip_0001.wav, clip_0002.wav, ...)
├── hf_dataset/               ← (6-cı addımdan sonra yaranır) HuggingFace dataset formatı
├── finetuned_model/          ← (fine-tuning-dən sonra) hazır model burada saxlanılır
│
├── urls.txt                  ← Sənin dolduracağın YouTube linkləri
├── metadata_template.csv     ← Avtomatik yaranan boş şablon (audio adı + boş mətn)
├── metadata.csv               ← Sənin əl ilə transkript etdiyin son fayl
│
├── requirements.txt
├── README.md                  ← bu fayl
│
└── scripts/
    ├── config.py               ← bütün tənzimlənə bilən parametrlər
    ├── 1_download.py           ← YouTube-dan audio endirmək
    ├── 2_normalize.py          ← formatı standartlaşdırmaq
    ├── 3_split.py               ← susqunluğa görə kiçik parçalara bölmək
    ├── 4_make_template.py       ← transkripsiya üçün boş CSV şablonu yaratmaq
    ├── 5_validate.py             ← doldurulmuş metadata.csv-ni yoxlamaq
    └── 6_build_dataset.py        ← fine-tuning üçün HuggingFace dataset formatına çevirmək
```

---

## Quraşdırma (birdəfəlik)

### 1. Python paketləri

```bash
cd az_voice_project
pip install -r requirements.txt
```

### 2. ffmpeg

```bash
# Mac
brew install ffmpeg

# Linux
sudo apt install ffmpeg

# Windows
# https://ffmpeg.org/download.html saytından endir və PATH-ə əlavə et
```

---

## Addım-addım istifadə

Bütün əmrləri layihənin **kök qovluğundan** (`az_voice_project/`) işə sal.

### 1-ci addım — Audio endir

`urls.txt` faylını aç, hər sətirdə bir YouTube linki yaz (`#` ilə başlayan
sətirlər nəzərə alınmır). Sonra:

```bash
python scripts/1_download.py
```

**Nəticə:** `raw_audio/video1.wav`, `raw_audio/video2.wav`, ...

### 2-ci addım — Formatı normallaşdır

```bash
python scripts/2_normalize.py
```

**Nəticə:** `normalized/` qovluğunda eyni fayllar, amma 16kHz/mono formatda.

### 3-cü addım — Kiçik parçalara böl

```bash
python scripts/3_split.py
```

**Nəticə:** `segments/clip_0001.wav`, `segments/clip_0002.wav`, ...

Əgər parçalar çox böyük/kiçik çıxırsa, `scripts/config.py` faylında
`SILENCE_THRESH_DB` dəyərini dəyiş (aşağı = daha həssas bölmə) və
3-cü addımı təkrar işə sal.

### 4-cü addım — Transkripsiya şablonu yarat

```bash
python scripts/4_make_template.py
```

**Nəticə:** `metadata_template.csv` — audio fayl adları artıq sırayla
doldurulub, `text` sütunu isə boşdur.

### 5-ci addım — Manual transkripsiya (sənin işin)

1. `metadata_template.csv`-ni Excel / Google Sheets-də aç
2. Hər sətirdə göstərilən audio faylını (`segments/` qovluğunda) dinlə
3. Eşitdiyini dəqiq şəkildə `text` sütununa yaz (durğu işarələrini
   unutma — `.`, `,`, `?`, `!` modelin intonasiyanı öyrənməsinə kömək edir)
4. Anlaşılmaz / fon küylü / səhv parçaları sil (sətri sil, istəsən
   audio faylını da)
5. Doldurub qurtarandan sonra faylı **`metadata.csv`** adı ilə
   layihənin kök qovluğuna yadda saxla

### 6-cı addım — Yoxlama

```bash
python scripts/5_validate.py
```

Bu, boş sətirləri, tapılmayan faylları, çox qısa mətnləri və ümumi
audio uzunluğunu göstərir. Fine-tuning üçün minimum **80-150 nümunə**
tövsiyə olunur — nə qədər çox, bir o qədər yaxşı.

### 7-ci addım — Dataset formatına çevir

```bash
python scripts/6_build_dataset.py
```

**Nəticə:** `hf_dataset/` qovluğu — bu, birbaşa fine-tuning skriptinə
ötürülə bilər.

---

## Fine-Tuning (Google Colab-da, GPU lazımdır)

`hf_dataset/` qovluğunu Google Drive-a yüklə (və ya Colab-a birbaşa
kopyala), sonra Colab-da:

```python
!git clone https://github.com/ylacombe/finetune-hf-vits.git
%cd finetune-hf-vits
!pip install -r requirements.txt
!pip install accelerate datasets

!python finetune_mms.py \
  --model_name_or_path facebook/mms-tts-azj-script_latin \
  --dataset_name /content/drive/MyDrive/hf_dataset \
  --output_dir /content/az-voice-finetuned \
  --num_train_epochs 200 \
  --per_device_train_batch_size 16 \
  --learning_rate 2e-5 \
  --fp16
```

> Dəqiq flaqlar repo versiyasına görə dəyişə bilər — işə salmadan əvvəl
> [repo-nun README-sini](https://github.com/ylacombe/finetune-hf-vits)
> yoxla.

Fine-tuning bitəndən sonra, nəticə modeli Colab-dan endirib
`finetuned_model/` qovluğuna qoya bilərsən.

---

## Tənzimləmə (`scripts/config.py`)

Bütün parametrlər (qovluq yolları, sample rate, bölmə həssaslığı və s.)
tək faylda — `scripts/config.py` — toplanıb. Skriptlərin heç birini
dəyişmədən, sadəcə bu faylı redaktə edərək davranışı tənzimləyə bilərsən.

| Parametr | Nə üçün | Defolt |
|---|---|---|
| `TARGET_SAMPLE_RATE` | Audio nümunələmə tezliyi | 16000 (MMS/VITS üçün standart) |
| `SILENCE_THRESH_DB` | Bölmə həssaslığı | -40 (aşağı = daha çox bölünür) |
| `MIN_CLIP_LEN_MS` / `MAX_CLIP_LEN_MS` | Parça uzunluq sərhədləri | 2000 / 12000 ms |
| `MIN_SILENCE_LEN_MS` | Bölmə nöqtəsi hesab olunan minimum susqunluq | 400 ms |

---

## Tövsiyə olunan iş axını (xülasə)

```
urls.txt doldur
      ↓
1_download.py        → raw_audio/
      ↓
2_normalize.py        → normalized/
      ↓
3_split.py             → segments/
      ↓
4_make_template.py     → metadata_template.csv
      ↓
(sən əl ilə transkript edirsən)
      ↓
metadata.csv (əl ilə yadda saxlanılır)
      ↓
5_validate.py           (yoxlama)
      ↓
6_build_dataset.py       → hf_dataset/
      ↓
Google Colab-da fine-tuning (GPU)
      ↓
finetuned_model/
```
