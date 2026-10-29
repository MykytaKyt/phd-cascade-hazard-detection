# Прогони на RunPod (A100 80GB)

Доступ: Jupyter пода `https://$RUNPOD_POD_ID-8888.proxy.runpod.net`, токен у `$RUNPOD_JUPYTER_TOKEN`
(REST API Jupyter: `/api/contents` для файлів, ядро/термінал через WebSocket).

Принцип: ті самі запити, ті самі промпти, температура 0; змінюється лише модель. Без донавчання і без
підбору промптів на тестових даних. Відповіді кешуються у `results/vlm/<dataset>_responses_<model>.jsonl`.

## Порядок

1. Підготовка пода: `git clone` репозиторію, `pip install -e ".[vad]"` + `vllm`; URFD — `tools/download_urfd.sh`;
   UCF-Crime (290 тестових відео) — надсилає автор (`runpodctl send`).
2. CLIP на UCF-Crime (розділ 5):
   - ViT-L-14 `openai` з QuickGELU — основний (виправлення помилки першого прогону);
   - ViT-H-14 `laion2b_s32b_b79k` — абляція енкодера.
   Далі `run_zeroshot.py`: перевірити `ece_cal` ≈ 0.04, `routed_pct` > 0, 1000 запитів.
3. VLM (сервер vLLM з OpenAI-сумісним API на порту 8000; `query_vlm.py --base-url http://localhost:8000/v1`):

   | Модель | Параметри | Питання |
   |---|---|---|
   | Gemma 4 E4B | ≈ 4B | базова; дозапитати 15 кліпів після виправлення `clip_frames` (7cb57fe) |
   | Qwen3-VL-8B-Instruct | 8B | та сама вагова категорія, інше сімейство |
   | Qwen3-VL-32B-Instruct | 32B | ефект розміру (AWQ, якщо bf16 не вміщується з запасом) |
   | старша Gemma 4 | ≈ 26–31B | ефект розміру в межах сімейства |

   Точні чекпоінти перевірити на Hugging Face перед запуском. Кожна модель: URFD (3 308 запитів) і UCF-Crime
   (1000 сегментів).
4. Абляція вікна кліпу для найкращої моделі: −10…+50 кадрів проти −40…+10.
5. Аналіз: `experiments/ch4_fusion/analyze_vlm.py`, `experiments/ch5_ucfcrime/analyze_vlm.py` для кожної моделі;
   таблиця «модель → AUC істинних проти хибних тривог → виявлення / хибні тривоги після злиття → умова (2.15)».
6. Закомітити результати (без `*.npz` і ваг), зупинити под.

Оцінка: ≈ 5–6 GPU-год.
