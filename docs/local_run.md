# Локальні прогони (GPU / LM Studio)

Усе, що не потребує GPU чи мультимодальної моделі, рахується скриптами в `experiments/` на CPU.
Тут — кроки, які потребують локальної машини: кадри URFD, YOLOv8-Pose, LM Studio, відео UCF-Crime.
Команди виконуються з кореня репозиторію; на Windows — у WSL або Git Bash (потрібні `curl`, `unzip`).

## 0. Середовище

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev,pose,vlm,vad]"
pip install torch --index-url https://download.pytorch.org/whl/cu128   # збірка PyTorch під CUDA для RTX 50xx
python -m pytest -q
```

## 1. Кадри URFD (≈ 4.3 ГБ)

```bash
bash tools/download_urfd.sh data
```

Кадри — у `data/frames/` (не потрапляють у git).

## 2. Перевірка VLM для епізодів тривоги (пріоритет 1, ≈ 300 запитів)

1. LM Studio: завантажити модель Gemma 4 E4B (та сама, що в статтях), увімкнути локальний сервер (порт 1234).
2. Дізнатися ідентифікатор моделі і запустити:

```bash
python experiments/ch4_fusion/query_vlm.py --list-models
python experiments/ch4_fusion/query_vlm.py --model <id> --kind event
```

Скрипт показує швидкість і залишок часу; його можна перервати і запустити знову — відповіді дописуються в
`results/vlm/urfd_responses_gemma-4-E4B.jsonl`, вже отримані пропускаються. Запити з помилками повторюються наступним запуском.

## 3. Ознаки з сирими ключовими точками і XCoM (пріоритет 2, GPU ≈ кілька хвилин)

```bash
python experiments/ch3_kinematic/extract_features.py --device 0
```

Результат — `data/features_kp.parquet`.

## 4. Покадрова перевірка VLM (пріоритет 3, ≈ 3000 запитів)

```bash
python experiments/ch4_fusion/query_vlm.py --model <id> --kind frame
```

Той самий файл відповідей.

## 5. UCF-Crime: CLIP-скоринг тестових відео (розділ 5)

Потрібні тестові відео UCF-Crime (290 відео: аномальні з тестового списку та `Testing_Normal_Videos_Anomaly`)
і файл `Temporal_Anomaly_Annotation_for_Testing_Videos.txt` — з офіційної сторінки набору даних
(Sultani et al., CVPR 2018). Структура каталогу довільна: відео шукаються рекурсивно за іменем.
Ваги CLIP завантажуються автоматично під час першого запуску.

```bash
python experiments/ch5_ucfcrime/score_clip.py \
    --videos-dir /path/to/UCF-Crime \
    --annotations /path/to/Temporal_Anomaly_Annotation_for_Testing_Videos.txt \
    --model ViT-L-14 --pretrained openai --stride 16 --device cuda
```

Результат — `results/ch5/clip_scores.parquet` (у git) і `results/ch5/clip_embeddings.npz` (великий, у git не йде).

Далі (CPU, можна і тут, і локально) — нульовий рівень, конформна маршрутизація і запити до VLM:

```bash
python experiments/ch5_ucfcrime/run_zeroshot.py
```

і перевірка VLM для маршрутизованих сегментів (кліп із 6 кадрів, запит із переліком 14 категорій):

```bash
python experiments/ch4_fusion/query_vlm.py --model <id> --kind ucf_segment \
    --requests results/vlm/ucf_requests.jsonl --out results/vlm/ucf_responses.jsonl \
    --videos-dir /path/to/UCF-Crime
```

## Що повернути

Закомітити й запушити (або надіслати файлами):

- `results/vlm/urfd_responses_gemma-4-E4B.jsonl`
- `data/features_kp.parquet`
- `results/ch5/clip_scores.parquet`
- `results/vlm/ucf_responses.jsonl` (якщо дійшли до кроку з VLM для UCF-Crime)

Далі на CPU: `experiments/ch4_fusion/analyze_vlm.py` (злиття свідчень, рівень події і кадру),
`experiments/ch3_kinematic/run_event.py --features data/features_kp.parquet --feature-set xcom` (XCoM),
`experiments/ch5_ucfcrime/run_zeroshot.py` і `experiments/ch5_ucfcrime/analyze_vlm.py` (розділ 5).
