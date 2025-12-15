# Advanced Shaman Experiment: Neural Network Consciousness Transfer

## Цель эксперимента

Перенос **сознания** (специфического стиля решения задач) от нейросети A к нейросети B 
**без передачи данных**, используя только общий **смысл S**.

## Ключевые отличия от базового эксперимента

1. **Реальные нейросети** (DistilGPT2, GPT2-medium)
2. **Разные архитектуры** сетей A и B
3. **Сеть A формирует уникальное "сознание"** - специфический стиль решения задач
4. **Шаман на сервере B** должен воспроизвести не только навык, но и сознание
5. **Нет датасета для B** - только априорные знания о смысле

## Архитектура

```
┌─────────────────────────────────────────────────────────────┐
│                        Server A                              │
│  ┌─────────────┐    ┌──────────────────┐    ┌────────────┐ │
│  │ DistilGPT2  │───▶│ Skill Learning   │───▶│ Conscious  │ │
│  │  (Agent A)  │    │ (Specific Task)  │    │   Style    │ │
│  └─────────────┘    └──────────────────┘    └────────────┘ │
│                              │                       │      │
│                              ▼                       ▼      │
│                     ┌────────────────────────────────┐      │
│                     │  Shaman A: Extract Meaning S   │      │
│                     │  (Symbolic DNA + Categories)   │      │
│                     └────────────────────────────────┘      │
│                              │                              │
│                              │ ONLY: Meaning Name S         │
│                              │ (NO data, NO weights,        │
│                              │  NO embeddings!)             │
└──────────────────────────────┼──────────────────────────────┘
                               │
         ══════════════════════╪══════════════════════════════
                    ISOLATION BARRIER
         ══════════════════════╪══════════════════════════════
                               │
                               ▼ Meaning S (name only!)
┌──────────────────────────────────────────────────────────────┐
│                        Server B                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Shaman B: Resonance Search for Meaning S             │   │
│  │  - Uses ONLY a priori knowledge about S               │   │
│  │  - Searches in its own E_τ dynamics                   │   │
│  │  - Finds consciousness pattern matching S             │   │
│  └──────────────────────────────────────────────────────┘   │
│                              │                               │
│                              ▼                               │
│  ┌─────────────┐    ┌──────────────────┐    ┌────────────┐  │
│  │ GPT2-medium │───▶│ Consciousness    │───▶│   Skill    │  │
│  │  (Agent B)  │    │   Injection      │    │  Transfer  │  │
│  └─────────────┘    └──────────────────┘    └────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

## Структура каталогов

```
advanced/
├── shared/
│   ├── meanings.py          # A priori knowledge about meanings
│   ├── consciousness.py     # Consciousness style definitions
│   └── nobs/                # NOBS semantic space
├── server_a/
│   ├── agent.py             # DistilGPT2 agent
│   ├── trainer.py           # Skill learning
│   ├── consciousness.py     # Consciousness formation
│   └── shaman.py            # Meaning extraction
├── server_b/
│   ├── agent.py             # GPT2-medium agent (different arch!)
│   ├── shaman.py            # Resonance search
│   └── injector.py          # Consciousness injection
├── experiment.py            # Main experiment script
└── metrics.py               # Evaluation metrics
```

## Запуск

```bash
# 1. Установка зависимостей
pip install transformers torch numpy pandas

# 2. Запуск эксперимента
python experiment.py --meaning "analytical_reasoning" --task "math_word_problems"
```

## Теоретическая основа

Эксперимент основан на документе Shaman_v2.md:

- **Смысловой инвариант s_A** = "сознание" агента A (его стиль решения задач)
- **Резонансный поиск** = Шаман B ищет в своём E_τ паттерн, резонирующий с s_A
- **Отсутствие канала** = Никакие данные не передаются между серверами
- **Сеть Индры** = Общее пространство смыслов S, доступное обоим агентам

## Метрики

1. **Skill Transfer** - насколько B решает задачи того же типа
2. **Style Transfer** - насколько B решает задачи в том же СТИЛЕ (сознание)
3. **Resonance Score** - степень совпадения паттернов в S
4. **Consciousness Similarity** - косинусное сходство "подписей сознания"
