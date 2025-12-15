# Advanced Shaman Experiments

> Продвинутые эксперименты по переносу сознания между нейросетями

---

## 📋 Содержимое директории

| Файл | Описание | Статус |
|------|----------|--------|
| `experiment.py` | Базовый эксперимент (1 тип сознания) | ✅ Готов |
| `full_experiment.py` | Все 5 типов сознания | ✅ Готов |
| `instruct_experiment.py` | **Лучший: instruction-tuned** | ✅ **Рекомендуется** |

---

## 🚀 Быстрый старт

### Рекомендуемый эксперимент (Instruct v2)

```bash
# Установка зависимостей (если ещё не установлены)
pip install transformers torch numpy pandas pyarrow

# Запуск лучшего эксперимента
python instruct_experiment.py --all
```

**Результат:** 71% Style Transfer, 57% Consciousness Match

---

## 1. experiment.py — Базовый эксперимент

Перенос сознания от DistilGPT2 к GPT2-medium.

```bash
python experiment.py --consciousness analytical_professor
```

**Параметры:**
- `--consciousness` — тип сознания (analytical_professor, creative_solver, etc.)
- `--samples` — число обучающих примеров (default: 50)
- `--epochs` — эпохи обучения (default: 2)

**Результаты:** ~57% match, ~15% style

---

## 2. full_experiment.py — Полный эксперимент

Тестирование всех 5 типов сознания с визуализацией.

```bash
python full_experiment.py \
    --samples 100 \
    --epochs 3 \
    --resonance-samples 1000 \
    --output ./full_experiment_results
```

**Параметры:**
- `--samples` — число примеров (default: 100)
- `--epochs` — эпохи (default: 3)
- `--resonance-samples` — точки резонанса (default: 1000)
- `--output` — директория результатов
- `--visualize` — создавать графики (default: True)

**Результаты:** 57.4% match avg, 4% style avg

---

## 3. instruct_experiment.py — Instruction-Tuned ⭐

**Лучший эксперимент** с few-shot prompting.

### Модели:
- **Server A:** Qwen1.5-0.5B-Chat (464M params)
- **Server B:** TinyLlama-1.1B-Chat (1.1B params)

### Запуск:

```bash
# Один тип сознания
python instruct_experiment.py --type pedantic_engineer

# Все 5 типов
python instruct_experiment.py --all

# Полный запуск
python instruct_experiment.py \
    --all \
    --samples-a 25 \
    --samples-b 5 \
    --resonance-samples 500 \
    --output ./instruct_v2_results
```

### Параметры:

| Параметр | Описание | Default |
|----------|----------|---------|
| `--type` | Конкретный тип сознания | None |
| `--all` | Все 5 типов | False |
| `--samples-a` | Вопросы для Server A | 25 |
| `--samples-b` | Тестовые вопросы B | 5 |
| `--resonance-samples` | Точки резонанса | 500 |
| `--output` | Директория | ./instruct_v2_results |

### Результаты v2:

| Тип | Match | Style B |
|-----|-------|---------|
| analytical_professor | 57.4% | **82.0%** |
| creative_solver | 55.2% | 44.4% |
| intuitive_guesser | 56.6% | 48.0% |
| pedantic_engineer | **60.4%** | **88.0%** |
| philosophical_thinker | 57.7% | **94.0%** |

**Средний Style Transfer: 71.3%** ✨

---

## 📁 Структура директории

```
advanced/
├── experiment.py              # Базовый эксперимент
├── full_experiment.py         # Полный эксперимент (5 типов)
├── instruct_experiment.py     # ⭐ Лучший эксперимент
├── shared/
│   ├── consciousness.py       # Определения типов сознания
│   ├── nobs_consciousness.py  # NOBS кодирование
│   ├── instruct_models.py     # Wrapper для instruction моделей
│   └── visualization.py       # Построение графиков
├── server_a/
│   ├── agent.py               # DistilGPT2 агент
│   └── agent_v2.py            # Улучшенный агент
├── server_b/
│   ├── shaman.py              # Резонансный поиск
│   └── shaman_v2.py           # Улучшенный шаман
└── *_results/                 # Результаты экспериментов
```

---

## 📊 5 типов сознания

| Тип | Описание | Маркеры |
|-----|----------|---------|
| `analytical_professor` | Строгий аналитик | "Step 1...", "Q.E.D." |
| `creative_solver` | Творческий решатель | "What if...", "Aha!" |
| `philosophical_thinker` | Философ | "Let us contemplate...", поэзия |
| `pedantic_engineer` | Педантичный инженер | "VERIFICATION:", "CONFIRMED" |
| `intuitive_guesser` | Интуитивист | "My gut says...", "I sense..." |

---

## 🔧 Требования

```
torch>=2.0.0
transformers>=4.40.0
pandas>=1.5.0
numpy>=1.24.0
pyarrow>=12.0.0
```

### GPU

- Минимум: RTX 3060 Ti (8GB VRAM)
- Рекомендуется: RTX 4070+ (12GB VRAM)

### HuggingFace Token

Для скачивания моделей нужен токен (уже встроен в код):
```bash
export HF_TOKEN="hf_your_token_here"
```

---

## 📚 Документация

- `../docs/Shaman_v2.md` — теория
- `../docs/Experiment_InstructModels_v2.md` — полное описание v2
- `../docs/Results_Instruct.md` — результаты

---

*Обновлено: 15 декабря 2025*
