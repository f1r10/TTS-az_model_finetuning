# AZJ VITS/MMS Fine-Tuning — README

Bu sənəd `AZJ_VITS_Finetune.ipynb` notebook-unun **necə işlədiyini**, hansı faylın **nə üçün** olduğunu və konfiqurasiyadakı hər parametri **dəyişdikdə nə baş verəcəyini** sadə dildə izah edir.

---

## 1. Bu layihə nə edir?

Meta-nın `facebook/mms-tts-azj-script_latin` adlı hazır Azərbaycan dilli mətn-səs (TTS) modelini götürüb, **sizin öz səs nümunələriniz üzərində əlavə öyrədirik** (buna "fine-tuning" deyilir). Nəticədə model sizin dataset-inizdəki səsə/tələffüzə/intonasiyaya daha yaxın nəticə verməyə başlayır.

Model növü **VITS** (Variational Inference with adversarial learning for end-to-end Text-to-Speech) adlanır — bu, mətni birbaşa səs dalğasına çevirən, GAN (rəqib şəbəkə) prinsipi ilə öyrədilən bir arxitekturadır.

---

## 2. Ümumi iş axını (addım-addım məntiq)

```
1) Dataset-i yoxla  →  2) train/test-ə böl  →  3) Əsas MMS modelini yüklə
   →  4) Fine-tuning reposunu qur  →  5) Modeli HF formatına çevir
   →  6) Konfiqurasiya (azj_finetune.json) yaz  →  7) Təlimi başlat
   →  8) (lazım olsa) kəsilən yerdən davam et  →  9) Nəticəni test et
   →  10) (istəyə bağlı) Hugging Face Hub-a yüklə
```

Notebook-da hər bölmə məhz bu ardıcıllıqla, nömrələnmiş şəkildə (1-dən 22-yə qədər) düzülüb.

---

## 3. Fayl və qovluq strukturu (Colab mühitində)

Notebook işlədikcə `/content/` altında aşağıdakı fayl/qovluqlar yaranır:

| Yol | Nə üçündür |
|---|---|
| `/content/hf_dataset.zip` | Sizin əvvəlcədən yüklədiyiniz, ilkin (bölünməmiş) dataset. |
| `/content/dataset/` | Zip-dən açılmış ilkin dataset. |
| `/content/azj_tts_dataset/` | `train`/`test`-ə bölünmüş, təlim skriptinin oxuya biləcəyi **son dataset**. Konfiqurasiyada `dataset_name` bura işarə edir. |
| `/content/azj-script_latin-full.tar.gz` | Meta-dan yüklənən orijinal, tam (generator+discriminator) checkpoint arxivi. |
| `/content/mms-azj-full/` | Yuxarıdakı arxivin açılmış hissəsi (`D_100000.pth` — discriminator çəkiləri buradadır). |
| `/content/mms-azj-hf/` | Hugging Face-dən yüklənmiş, yalnız **generator** hissəsini ehtiva edən model (tokenizer də daxil). |
| `/content/azj-mms-training/` | Generator + çevrilmiş discriminator birləşdirilərək əldə olunan, **fine-tuning-ə hazır başlanğıc model**. Konfiqurasiyada `model_name_or_path` bura işarə edir. |
| `/content/finetune-hf-vits/` | GitHub-dan klonlanmış təlim kodu (`ylacombe/finetune-hf-vits`). |
| `/content/finetune-hf-vits/azj_finetune.json` | **Əsas konfiqurasiya faylı** — bütün təlim parametrləri burada (bax: bölmə 5). |
| `/content/azj_vits_finetuned/` | Təlimin **çıxışı** — fine-tune edilmiş model və `checkpoint-<N>` adlı ara nəticələr burada saxlanılır. |

---

## 4. Sistem necə işləyir (mərhələ-mərhələ)

### 4.1. Dataset hazırlığı
Sizin dataset-iniz Hugging Face `datasets` formatındadır (Arrow faylları + metadata). Hər sətirdə bir `audio` (dalğa massivi + sampling rate) və uyğun `text` (mətn) olmalıdır. Model bu cütlükləri "bu mətni bu cür səsləndir" şəklində öyrənir.

Dataset **mütləq** `train` və `test` adlı iki hissəyə bölünməlidir, çünki `finetune-hf-vits` skripti `datasets.load_dataset()` funksiyasından istifadə edir və bu funksiya yalnız `train` / `test` / `validation` adlarını avtomatik tanıyır (`eval` adını tanımır — bu, orijinal (düzəlişsiz) versiyalarda ən çox rast gəlinən xəta mənbəyi idi).

### 4.2. Əsas modelin hazırlanması
VITS/GAN arxitekturasında iki hissə var:
- **Generator** — mətndən səs yaradan hissə (bunu sonda istifadə edəcəyik).
- **Discriminator** — təlim zamanı generatorun yaratdığı səsin "real" səsə nə qədər bənzədiyini qiymətləndirən hissə (yalnız təlim zamanı lazımdır, sonda atılır).

Hugging Face-dəki `facebook/mms-tts-azj-script_latin` modelində **yalnız generator** var. Discriminator isə yalnız Meta-nın öz saytındakı tam arxivdə mövcuddur. Ona görə iki mənbədən yükləyib, `convert_original_discriminator_checkpoint.py` skripti ilə birləşdiririk.

### 4.3. Mühit uyğunlaşdırması (patch-lər)
`finetune-hf-vits` kodu köhnə `transformers` versiyasına (təxminən 2023-cü il) uyğun yazılıb. Bugünkü ən yeni `transformers` versiyaları ilə bəzi API-lər (məsələn `TrainingArguments.overwrite_output_dir`) artıq mövcud deyil. Ona görə:
- `transformers`, `accelerate`, `huggingface_hub` **konkret, sınaqdan keçmiş versiyalara** sabitlənir.
- Skriptdəki bir neçə kiçik hissə (telemetry çağırışı, `speaker_id` emalı, `plot.py` köməkçi faylı) avtomatik düzəldilir.

Bu addımları **atlamayın** — əks halda təlim başlamazdan əvvəl xəta ilə dayanacaq.

### 4.4. Təlim (fine-tuning)
`accelerate launch run_vits_finetuning.py azj_finetune.json` əmri konfiqurasiya faylındakı bütün parametrləri oxuyub təlimi başladır. Hər `save_steps` addımdan bir (defolt: 25) `output_dir` içində `checkpoint-<addım>` adlı bir qovluq yaranır — bu, o anki modelin tam vəziyyətinin (çəkilər + optimizer vəziyyəti) surətidir.

### 4.5. Kəsilmə və davametmə
Google Colab-ın pulsuz versiyasında sessiyalar bəzən kəsilir (timeout, internet problemi və s.). Bu halda, təlimi sıfırdan başlamaq **lazım deyil** — son checkpoint-i tapıb, konfiqurasiyaya `resume_from_checkpoint` olaraq yazıb, təlimi yenidən işə salmaq kifayətdir. Notebook-un 17-ci bölməsi bunu avtomatik edir.

### 4.6. Nəticənin qiymətləndirilməsi
Fine-tune edilmiş modeli yükləyib istənilən mətni səsləndirə bilərsiniz. Bir neçə checkpoint-i müqayisə etmək tövsiyə olunur, çünki bəzən ən son checkpoint həmişə ən yaxşısı olmur (overfitting baş verə bilər — yəni model dataset-i "əzbərləyib", yeni mətnlərdə pis səsləndirə bilər).

---

## 5. Konfiqurasiya parametrləri (`azj_finetune.json`) — tam izah

Bu fayl 13-cü bölmədə yaradılır. Aşağıda **hər parametrin** nə etdiyi və dəyişdirilməsinin nəyə təsir edəcəyi izah olunub.

### 5.1. Yol / dataset parametrləri

| Parametr | Nə edir | Dəyişsəniz nə olar |
|---|---|---|
| `project_name` | Sadəcə təlimin adı (log-larda görünür). | Sərbəst ad verə bilərsiniz, funksional təsiri yoxdur. |
| `model_name_or_path` | Başlanğıc modelin yolu (`/content/azj-mms-training`). | Yanlış yol versəniz, model tapılmır və xəta alırsınız. |
| `dataset_name` | Təlim üçün istifadə olunacaq dataset-in yolu. | Mütləq 5-ci addımda yaradılan `train`/`test` bölünmüş dataset-ə işarə etməlidir. |
| `audio_column_name` / `text_column_name` | Dataset-dəki hansı sütunun audio, hansının mətn olduğunu bildirir. | Öz dataset-inizdə sütun adları fərqlidirsə, buraya uyğun adları yazın. |
| `train_split_name` / `eval_split_name` | Dataset-in hansı bölməsinin təlim, hansının qiymətləndirmə üçün olduğu. | **`eval_split_name` mütləq `"test"` olmalıdır** — `"eval"` yazsanız, skript bölməni tapa bilmir və xəta verir. |
| `output_dir` | Fine-tune edilmiş modelin və checkpoint-lərin yazılacağı yer. | Fərqli qovluq versəniz, nəticələr orada saxlanılır (əvvəlki nəticələrlə qarışmaz). |
| `overwrite_output_dir` | `true` olsa, mövcud `output_dir` içindəkilər üzərinə yazıla bilər. | `false` etsəniz, eyni qovluqda əvvəlki nəticə varsa, xəta ala bilərsiniz. |

### 5.2. Təlim rejimi

| Parametr | Nə edir | Dəyişsəniz nə olar |
|---|---|---|
| `do_train` / `do_eval` | Təlim və/və ya qiymətləndirmənin işə düşüb-düşməyəcəyi. | `do_eval: false` etsəniz, təlim sürətlənir, amma keyfiyyəti izləyə bilməzsiniz. |
| `num_train_epochs` | Bütün dataset üzərindən neçə dəfə keçiləcəyi. | Az dataset (məs. <100 nümunə) üçün çox epoch (50+) lazım ola bilər; böyük dataset üçün az epoch kifayət edə bilər. Çox epoch = daha uzun təlim vaxtı, overfitting riski. |
| `preprocessing_only` | `true` olsa, yalnız dataset emalı edilir, təlim başlamır. | Yalnız dataset-in düzgün emal olunduğunu yoxlamaq üçün faydalıdır. |

### 5.3. Batch və yaddaş

| Parametr | Nə edir | Dəyişsəniz nə olar |
|---|---|---|
| `per_device_train_batch_size` / `per_device_eval_batch_size` | Hər addımda eyni anda neçə nümunənin emal olunacağı. | Böyütsəniz təlim sürətlənə bilər, amma **GPU yaddaşı bitə bilər** ("CUDA out of memory" xətası). Colab-ın pulsuz T4 GPU-su (15 GB) üçün 4–8 arası təhlükəsizdir. |
| `gradient_accumulation_steps` | Neçə addım toplanıb sonra bir dəfə çəkilər yenilənəcəyi (effektiv batch size-ı böyütmək üçün, yaddaş sərf etmədən). | Effektiv batch = `batch_size × gradient_accumulation_steps`. Yaddaş azdırsa, bunu artırıb `batch_size`-ı azalda bilərsiniz. |
| `fp16` | 16-bit (yarım dəqiqlikli) hesablama istifadə olunsun mu. | `true` = daha az yaddaş, daha sürətli təlim (demək olar ki, həmişə açıq saxlanmalıdır GPU-da). |
| `dataloader_num_workers` / `preprocessing_num_workers` | Dataset-i paralel emal edən proses sayı. | Colab-da adətən 2 kifayətdir; çox artırsanız fayda verməyə bilər. |

### 5.4. Öyrənmə sürəti (learning rate) və optimizasiya

| Parametr | Nə edir | Dəyişsəniz nə olar |
|---|---|---|
| `learning_rate` | Modelin çəkilərinin hər addımda nə qədər dəyişəcəyi. | Çox böyük dəyər → təlim qeyri-stabil olur (səs "cızıltılı" çıxa bilər). Çox kiçik → təlim çox yavaş gedir. `2e-5` fine-tuning üçün təhlükəsiz başlanğıcdır. |
| `adam_beta1` / `adam_beta2` | Adam optimizatorunun daxili parametrləri (momentum). | Adətən dəyişdirilmir — standart VITS dəyərləridir. |
| `warmup_ratio` | Təlimin əvvəlində öyrənmə sürətinin tədricən artırıldığı hissə. | Çox kiçik dataset-lərdə bir az artırmaq təlimi stabilləşdirə bilər. |
| `do_step_schedule_per_epoch` / `lr_decay` | Öyrənmə sürətinin hər epoch-dan sonra necə azaldılacağı. | `lr_decay` 1-ə nə qədər yaxındırsa, sürət bir o qədər yavaş azalır. |

### 5.5. Dataset filtri

| Parametr | Nə edir | Dəyişsəniz nə olar |
|---|---|---|
| `min_duration_in_seconds` / `max_duration_in_seconds` | Bu diapazondan kənar audio nümunələri təlimdən **çıxarılır**. | Dataset-inizin real uzunluq statistikasına (4-cü bölmədə hesablanır) uyğunlaşdırın — əks halda faydalı nümunələr atıla bilər, ya da çox uzun/qısa nümunələr xəta yarada bilər. |
| `max_tokens_length` | Mətnin maksimum token (simvol əsaslı vahid) sayı. | Çox uzun cümlələriniz varsa artırın; yoxdursa toxunmağa ehtiyac yoxdur. |

### 5.6. Kaib (loss) çəkiləri — modelin nəyə daha çox "diqqət" edəcəyi

VITS bir neçə fərqli itki (loss) funksiyasının çəkili cəmini minimuma endirir:

| Parametr | Hansı komponentə aiddir | Artırsanız nə olar |
|---|---|---|
| `weight_gen` | Generatorun "real görünmə" itkisi | Səsin təbiiliyinə təsir edir. |
| `weight_disc` | Discriminator itkisi | Çox böyük olsa, generator "aldatmağa" çətinləşir, təlim yavaşıya bilər. |
| `weight_fmaps` | Feature-matching itkisi (ara qatların uyğunluğu) | Səsin sabitliyinə kömək edir. |
| `weight_mel` | Mel-spektroqram itkisi (ən vacib keyfiyyət göstəricisi) | Ən böyük çəki adətən buna verilir (`35`) — səsin əsas tembr/keyfiyyətinə birbaşa təsir edir. |
| `weight_kl` | KL-divergence itkisi (latent uzayın nizamlılığı) | Çox dəyişdirilmir. |
| `weight_duration` | Fonemlərin müddət proqnozunun düzgünlüyü | Tələffüz sürəti/ritminə təsir edir. |

**Tövsiyə:** bu çəkiləri dəyişməzdən əvvəl standart dəyərlərlə bir dəfə tam təlim aparıb nəticəni dinləyin — yalnız problem görsəniz (məsələn səs "robota bənzəyir" və ya "sürəti qəribədir") uyğun çəkini kiçik addımlarla dəyişin.

### 5.7. Loglama, saxlama, təkrarlana bilənlik

| Parametr | Nə edir | Dəyişsəniz nə olar |
|---|---|---|
| `logging_steps` | Neçə addımdan bir təlim statistikası çap olunacağı. | Kiçik dəyər = daha tez-tez məlumat, amma daha "səs-küylü" log. |
| `eval_steps` | Neçə addımdan bir qiymətləndirmə (test dataset üzərində) aparılacağı. | Çox tez-tez etsəniz təlim yavaşıyar. |
| `save_steps` | Neçə addımdan bir checkpoint yaradılacağı. | Kiçik dəyər = tez-tez saxlama (kəsilmə riskinə qarşı yaxşı), amma daha çox disk yeri tutar. |
| `save_total_limit` | Diskdə eyni anda maksimum neçə checkpoint saxlanılacağı (köhnələr avtomatik silinir). | Böyütsəniz daha çox tarixçə saxlanılır, amma disk yeri daha tez bitə bilər. |
| `report_to` | Nəticələrin hansı xarici xidmətə (məs. Weights & Biases) göndəriləcəyi. | `[]` = heç yerə göndərilmir (yalnız lokal). İstəsəniz `["wandb"]` əlavə edə bilərsiniz (əlavə quraşdırma tələb edir). |
| `seed` | Təsadüfi ədəd generatorunun başlanğıc nöqtəsi (nəticələrin təkrarlana bilməsi üçün). | Dəyişsəniz, dataset bölgüsü/qarışdırılması bir az fərqli ola bilər, amma nəticə keyfiyyətinə ciddi təsir etməz. |
| `full_generation_sample_text` | Qiymətləndirmə zamanı modeldən səsləndirilməsi istənən nümunə cümlə (log-larda dinləmək üçün). | İstədiyiniz Azərbaycan cümləsi ilə əvəz edə bilərsiniz. |
| `resume_from_checkpoint` | (Notebook avtomatik əlavə edir) Hansı checkpoint-dən davam ediləcəyi. | Əl ilə dəyişməyə ehtiyac yoxdur — 17-ci bölmə bunu özü idarə edir. |

---

## 6. Hansı hüceyrələri **dəyişməməli**, hansıları **sərbəst dəyişə bilərsiniz**

**Toxunmamaq tövsiyə olunur** (təlimin işləməsi üçün kritikdir):
- Bölmə 7 — `transformers`/`accelerate`/`huggingface_hub` versiya sabitləməsi.
- Bölmə 11 — `pad_token_id` düzəlişi.
- Bölmə 14 — telemetry, `speaker_id`, `plot.py` patch-ləri.
- Konfiqurasiyada `dataset_name`, `model_name_or_path`, `eval_split_name`, `train_split_name`.

**Sərbəst dəyişə bilərsiniz** (öz ehtiyacınıza görə):
- `num_train_epochs`, `learning_rate`, `per_device_train_batch_size` (yaddaşa görə).
- `min_duration_in_seconds` / `max_duration_in_seconds` (öz dataset statistikanıza görə).
- `full_generation_sample_text`, test mətnləri (bölmə 19, 21).
- `repo_name` / `your_username` (Hub-a yükləmə, bölmə 20-21).

---

## 7. Tez-tez rastlanan xətalar və həlləri

| Xəta / simptom | Ehtimal olunan səbəb | Həll |
|---|---|---|
| `ValueError: Some keys are not used by the HfArgumentParser` | `transformers` versiyası çox yenidir | Bölmə 7-ni yenidən işə salın (versiya sabitləmə). |
| `load_dataset` boş/xəta qaytarır | `eval_split_name` `"eval"` olub | Konfiqurasiyada `"test"` olmalıdır (bölmə 13). |
| `CUDA out of memory` | Batch size çox böyükdür | `per_device_train_batch_size`-ı azaldın (məs. 4-ə), yaxud `gradient_accumulation_steps`-i artırın. |
| Təlim naməlum yerdə dayanıb | Colab sessiyası kəsilib | Bölmə 17-ni işə salın (avtomatik davam etdirmə). |
| Checkpoint-dən model yüklənmir | `config.json` checkpoint qovluğunda yoxdur | Bölmə 19-dakı "config.json kopyala" hüceyrəsini işə salın. |
| Disk yeri bitir | Çoxlu checkpoint / böyük arxiv faylları qalıb | Bölmə 18-i işə salın; `save_total_limit`-i azaldın. |
| Səs qəribə/robot kimi səslənir | Az epoch, kiçik dataset, ya da uyğunsuz `weight_*` dəyərləri | Daha çox epoch/data ilə sınayın; son bir neçə checkpoint-i müqayisə edin (bölmə 19). |

---

## 8. Tövsiyələr

- **Ən azı 30–60 dəqiqəlik təmiz audio** (çoxlu qısa cümlə şəklində) fine-tuning üçün minimal, yaxşı nəticə üçün bir neçə saat tövsiyə olunur.
- Audio nümunələri **təmiz, arxa fonsuz, sabit səviyyəli** olmalıdır — keyfiyyətli giriş, keyfiyyətli nəticə deməkdir.
- Təlim zamanı bir neçə checkpoint-i (məsələn hər 100-200 addımdan bir) saxlayıb sonda müqayisə etmək, "ən yaxşı" nöqtəni tapmağa kömək edir (çünki ən son checkpoint həmişə ən yaxşısı olmaya bilər).
- Nəticədən razı qaldıqdan sonra modeli Hugging Face Hub-a yükləmək, onu başqa layihələrdə asanlıqla istifadə etməyə imkan verir.
