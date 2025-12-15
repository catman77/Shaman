# 🧙 Shaman Experiments - Полное руководство

**Искусственный Шаман** — серия экспериментов по переносу "сознания" (семантических инвариантов) между изолированными AI-системами без прямого обмена данными.

---

## 📋 Оглавление экспериментов

| # | Эксперимент | Описание | Результат |
|---|-------------|----------|-----------|
| 1 | [Base Experiment](#1-base-experiment) | Базовый перенос стиля понимания рынка | d_P снизился на 14% |
| 2 | [Advanced Experiment](#2-advanced-experiment) | NOBS + сознание (1 тип) | 57% match, 15% style |
| 3 | [Full Experiment](#3-full-experiment) | Все 5 типов сознания | 57.4% match avg |
| 4 | [Instruct Experiment v2](#4-instruct-experiment-v2) | Instruction-tuned + few-shot | **71% style transfer** ✨ |

---

## 🛠️ Общие требования

### Системные требования

```
OS: Linux / macOS / Windows (WSL2)
Python: 3.10+
GPU: NVIDIA RTX 3060 Ti+ (8GB VRAM)
RAM: 16GB+
Disk: 10GB+ (для моделей и данных)
```

### Установка окружения

```bash
# 1. Клонируем/переходим в директорию
cd /path/to/Shaman

# 2. Создаём виртуальное окружение
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или: .\venv\Scripts\activate  # Windows

# 3. Устанавливаем зависимости
pip install -r requirements.txt
```

### requirements.txt

```
torch>=2.0.0
transformers>=4.40.0
sentence-transformers>=2.2.0
pandas>=1.5.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.2.0
gudhi>=3.8.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
PyYAML>=6.0
pyarrow>=12.0.0  # для feather файлов
```

### Данные Bitcoin

Положите файл с Bitcoin данными в `/data/`:
```
/data/BTC_USDT_USDT-4h-futures.feather
```

Требуемые колонки: `open`, `high`, `low`, `close`, `volume`

---

## 1. Base Experiment

**Цель:** Проверить базовую гипотезу о резонансном переносе смысловых инвариантов.

### Описание

Два агента (оба на DistilGPT2) формируют "стиль понимания рынка". Шаман B пытается найти стиль A без доступа к его данным.

### Архитектура

```
Agent A (DistilGPT2) ──► формирует s_A в пространстве S
                              │
                              │ ONLY: d_P distance
                              ▼
Agent B (DistilGPT2) ◄── Шаман ищет резонанс
```

### Запуск

```bash
cd /path/to/Shaman

# Быстрый тест (~2 мин)
python run_experiment.py --quick

# Полный эксперимент (~15 мин)
python run_experiment.py

# С кастомными параметрами
python run_experiment.py --runs 5 --device cuda --seed 123
```

### Параметры

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--quick` | Быстрый режим | False |
| `--device` | cuda/cpu | cuda |
| `--seed` | Random seed | 42 |
| `--runs` | Число повторов | 3 |

### Ожидаемые результаты

```
d_P before shaman: 0.4523 ± 0.02
d_P after shaman:  0.3891 ± 0.02
Reduction:         14% ✓
```

### Файлы

- `run_experiment.py` — главный скрипт
- `src/experiment.py` — логика эксперимента
- `src/shaman.py` — резонансный поиск
- `logs/` — результаты

---

## 2. Advanced Experiment

**Цель:** Перенос сознания между РАЗНЫМИ архитектурами через NOBS.

### Описание

Server A (DistilGPT2) обучается навыку с определённым "сознанием" (стилем). Server B (GPT2-medium) должен воспроизвести и навык, и стиль через NOBS-резонанс.

### Архитектура

```
Server A (DistilGPT2)    Server B (GPT2-medium)
        │                        │
        ▼                        ▼
    Сознание ──► NOBS ──► Резонанс
    (стиль)   Signature    Search
```

### Запуск

```bash
cd /path/to/Shaman/advanced

# Базовый запуск
python experiment.py

# С параметрами
python experiment.py \
    --consciousness analytical_professor \
    --samples 50 \
    --epochs 2
```

### Параметры

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--consciousness` | Тип сознания | analytical_professor |
| `--samples` | Число обучающих примеров | 50 |
| `--epochs` | Эпохи обучения | 2 |
| `--model-a` | Модель Server A | distilgpt2 |
| `--model-b` | Модель Server B | gpt2-medium |

### Типы сознания

1. `analytical_professor` — строгий аналитик
2. `creative_solver` — творческий решатель
3. `philosophical_thinker` — философ
4. `pedantic_engineer` — педантичный инженер
5. `intuitive_guesser` — интуитивный угадыватель

### Ожидаемые результаты

```
Consciousness Match: 57%
Style Transfer:      15%
Skill Transfer:      40%
```

### Файлы

- `advanced/experiment.py` — скрипт
- `advanced/server_a/` — Server A
- `advanced/server_b/` — Server B
- `advanced/shared/` — общие модули
- `advanced/experiment_results/` — результаты

---

## 3. Full Experiment

**Цель:** Тестирование всех 5 типов сознания с визуализацией.

### Описание

Расширенный эксперимент v2 с NOBS-кодированием и визуализацией. Запускает эксперимент для всех типов сознания последовательно.

### Запуск

```bash
cd /path/to/Shaman/advanced

# Все типы сознания
python full_experiment.py

# С параметрами
python full_experiment.py \
    --samples 100 \
    --epochs 3 \
    --resonance-samples 1000 \
    --output ./full_experiment_results
```

### Параметры

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--samples` | Число примеров | 100 |
| `--epochs` | Эпохи обучения | 3 |
| `--resonance-samples` | Сэмплы для резонанса | 1000 |
| `--output` | Директория результатов | ./full_experiment_results |
| `--visualize` | Создавать графики | True |

### Ожидаемые результаты

| Тип сознания | Match | Style | Skill |
|--------------|-------|-------|-------|
| analytical_professor | 60.1% | 15.0% | 40% |
| creative_solver | 54.1% | 0.0% | 40% |
| intuitive_guesser | 59.4% | 0.0% | 40% |
| pedantic_engineer | 61.3% | 5.0% | 40% |
| philosophical_thinker | 54.7% | 0.0% | 40% |

### Визуализации

Генерируются в `output/`:
- `consciousness_comparison_*.png` — сравнение A и B
- `all_consciousnesses.png` — все 5 типов
- `symbol_distribution.png` — распределение NOBS-символов

### Файлы

- `advanced/full_experiment.py` — скрипт
- `advanced/shared/visualization.py` — визуализация
- `advanced/shared/nobs_consciousness.py` — NOBS
- `advanced/full_experiment_results/` — результаты

---

## 4. Instruct Experiment v2

**Цель:** Максимизировать Style Transfer через instruction-tuned модели.

### Описание

Улучшенный эксперимент с:
- **Qwen1.5-0.5B-Chat** (Server A) — лучшее instruction-following
- **TinyLlama-1.1B-Chat** (Server B) — находит сознание
- **Few-shot prompting** — примеры в промптах
- **Weighted style scoring** — точная метрика

### 🔑 Ключевые улучшения v2

1. **Few-shot примеры** в системных промптах
2. **25 разнообразных вопросов** вместо 10
3. **Взвешенная оценка стиля** (start 30% + end 30% + structure 40%)
4. **350 токенов** на ответ

### Запуск

```bash
cd /path/to/Shaman/advanced

# Один тип сознания
python instruct_experiment.py --type analytical_professor

# Все 5 типов
python instruct_experiment.py --all

# Полный запуск с параметрами
python instruct_experiment.py \
    --all \
    --samples-a 25 \
    --samples-b 5 \
    --resonance-samples 500 \
    --output ./instruct_v2_results
```

### Параметры

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--type` | Конкретный тип | None |
| `--all` | Все типы | False |
| `--samples-a` | Вопросы для Server A | 25 |
| `--samples-b` | Тестовые вопросы B | 5 |
| `--resonance-samples` | Точки резонанса | 500 |
| `--output` | Директория результатов | ./instruct_v2_results |

### ✨ Результаты v2

| Тип сознания | Match | Style A | **Style B** |
|--------------|-------|---------|-------------|
| analytical_professor | 57.4% | 51.2% | **82.0%** |
| creative_solver | 55.2% | 4.9% | **44.4%** |
| intuitive_guesser | 56.6% | 6.4% | **48.0%** |
| pedantic_engineer | **60.4%** | **59.2%** | **88.0%** |
| philosophical_thinker | 57.7% | 3.6% | **94.0%** |

**Средний Style Transfer: 71.3%** (vs 4% в v1 — рост в 18 раз!)

### HuggingFace Token

Для скачивания моделей нужен токен:

```bash
# Установить переменную окружения
export HF_TOKEN="hf_your_token_here"

# Или в коде (уже настроено)
# token встроен в instruct_models.py
```

### Файлы

- `advanced/instruct_experiment.py` — главный скрипт
- `advanced/shared/instruct_models.py` — wrapper для моделей
- `advanced/shared/nobs_consciousness.py` — NOBS кодирование
- `advanced/instruct_v2_results/` — результаты

---

## 📊 Сравнение всех экспериментов

| Эксперимент | Модели | Style Transfer | Время |
|-------------|--------|----------------|-------|
| Base | DistilGPT2 ↔ DistilGPT2 | 14% d_P ↓ | 15 мин |
| Advanced | DistilGPT2 → GPT2-medium | 15% | 20 мин |
| Full | DistilGPT2 → GPT2-medium | 4% avg | 45 мин |
| **Instruct v2** | **Qwen-0.5B → TinyLlama-1.1B** | **71%** | 7 мин |

---

## 🔧 Troubleshooting

### CUDA out of memory

```bash
# Уменьшите batch_size или используйте меньшие модели
python instruct_experiment.py --samples-a 10
```

### Model download failed

```bash
# Проверьте HuggingFace token
huggingface-cli login
```

### Bitcoin data not found

```bash
# Положите файл в нужное место
cp /path/to/btc_data.feather /data/BTC_USDT_USDT-4h-futures.feather
```

### Import errors

```bash
# Переустановите зависимости
pip install -r requirements.txt --force-reinstall
```

---

## 📚 Документация

- `docs/Shaman_v2.md` — теоретическая база
- `docs/Results_Instruct.md` — результаты instruction-tuned
- `docs/Experiment_InstructModels_v2.md` — полное описание v2

---

## 🎯 Quick Start

Для быстрого запуска лучшего эксперимента:

```bash
# 1. Установка
cd Shaman
pip install -r requirements.txt

# 2. Запуск лучшего эксперимента (Instruct v2)
cd advanced
python instruct_experiment.py --all

# 3. Посмотреть результаты
cat instruct_v2_results/all_results.json | python -m json.tool
```

---

*Документация создана: 15 декабря 2025*
