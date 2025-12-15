# Distributed Shaman Experiment

Система распределённого эксперимента "Artificial Shaman" - передача семантических инвариантов между агентами **БЕЗ КАКОГО-ЛИБО канала данных**.

## Архитектура

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                    ┌─────────────────────────────┐                          │
│                    │   SHARED KNOWLEDGE BASE     │                          │
│                    │      (meanings.py)          │                          │
│                    │                             │                          │
│                    │  • bullish_trend            │                          │
│                    │  • bearish_trend            │                          │
│                    │  • high_volatility          │                          │
│                    │  • accumulation             │                          │
│                    │  • breakout                 │                          │
│                    │  • ...                      │                          │
│                    └──────────┬──────────────────┘                          │
│                               │                                             │
│                    ┌──────────┴──────────┐                                  │
│                    │   meaning_name      │                                  │
│                    │  (только название!) │                                  │
│                    └──────────┬──────────┘                                  │
│           ┌───────────────────┼───────────────────┐                         │
│           │                   │                   │                         │
│           ▼                   │                   ▼                         │
│   ┌───────────────────┐       │       ┌────────────────────────┐            │
│   │     SERVER A      │       │       │       SERVER B         │            │
│   │     (Learner)     │  ═══╪═══  │    (Shaman)          │            │
│   │                   │   NO DATA │                        │            │
│   │  ┌─────────────┐  │  TRANSFER │  ┌──────────────────┐  │            │
│   │  │ Data 0-50%  │  │           │  │   Data 50-100%   │  │            │
│   │  │ (свои данные)│  │           │  │  (свои данные)   │  │            │
│   │  └─────────────┘  │           │  └──────────────────┘  │            │
│   │        │          │           │           │            │            │
│   │        ▼          │           │           ▼            │            │
│   │  ┌─────────────┐  │           │  ┌──────────────────┐  │            │
│   │  │ NOBS Space  │  │           │  │    NOBS Space    │  │            │
│   │  └─────────────┘  │           │  └──────────────────┘  │            │
│   │        │          │           │           │            │            │
│   │        ▼          │           │           ▼            │            │
│   │  ┌─────────────┐  │           │  ┌──────────────────┐  │            │
│   │  │   Learn     │  │           │  │     Search       │  │            │
│   │  │  meaning    │  │           │  │    meaning       │  │            │
│   │  └─────────────┘  │           │  └──────────────────┘  │            │
│   │        │          │           │           │            │            │
│   │        ▼          │           │           ▼            │            │
│   │  Local model      │           │    Shaman report       │            │
│   │  (не передаётся)  │           │    (результаты)        │            │
│   │                   │           │                        │            │
│   └───────────────────┘           └────────────────────────┘            │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

## Ключевая идея: ЧИСТЫЙ ЭКСПЕРИМЕНТ

**Server A и Server B НЕ обмениваются никакими данными!**

Единственное, что они разделяют:
- **meanings.py** - априорная база знаний о семантических паттернах
- **Название смысла** (например, `"bullish_trend"`) - просто строка!

Это как если бы два человека знали, что такое "восходящий тренд" из учебника,
но работали с совершенно разными графиками и никогда не общались.

### Что НЕ передаётся:
- ❌ Сырые данные (OHLCV)
- ❌ Эмбеддинги
- ❌ Центроиды
- ❌ Распределения символов
- ❌ Морфизмы
- ❌ Хэши данных
- ❌ Вообще никакие метаданные!

### Что передаётся:
- ✅ Только название смысла (строка типа `"bullish_trend"`)

## Семантические паттерны (meanings.py)

Оба сервера используют **одинаковую** априорную базу знаний:

| Паттерн | Описание | Доминантный символ |
|---------|----------|-------------------|
| `bullish_trend` | Устойчивый рост | P (Promotion) |
| `bearish_trend` | Устойчивое падение | I (Inhibition) |
| `high_volatility` | Высокая волатильность | Z (Zap) |
| `low_volatility` | Низкая волатильность | S (Stability) |
| `accumulation` | Фаза накопления | S |
| `distribution` | Фаза распределения | S |
| `breakout` | Пробой уровня | Λ (Lambda) |
| `reversal` | Разворот тренда | Ω (Omega) |
| `trend_exhaustion` | Истощение тренда | S, Ω |
| `momentum_shift` | Смена импульса | Ω, Λ |

## NOBS Framework

Семантическое пространство построено на Natural Observation-Based Computing:

### Символический алфавит Σ

| Символ | Значение |
|--------|----------|
| S | Stability (doji) |
| P | Promotion (bullish) |
| I | Inhibition (bearish) |
| Z | Zap (high volatility) |
| Ω | Omega (trend end) |
| Λ | Lambda (breakout) |

### Категории C_n

- **C_1**: Одиночные символы как объекты
- **C_2**: Биграммы (PP, PI, IS, ...)
- **C_3**: Триграммы
- **C_4**: 4-граммы
- **C_5**: 5-граммы

### Свободная энергия F

```
F[σ] = E[σ] - T·S[σ]
```

где E - энергия (от волатильности), S - энтропия (от разнообразия символов).

## Установка и запуск

### Локальный запуск (без Docker)

```bash
# Server A: обучается распознавать смысл
cd distributed/server_a
pip install -r requirements.txt
python server.py \
    --meaning bullish_trend \
    --data /path/to/BTC_USDT_USDT-4h-futures.feather \
    --portion-start 0.0 \
    --portion-end 0.5

# Server B: ищет тот же смысл в ДРУГИХ данных
cd distributed/server_b
pip install -r requirements.txt
python server.py \
    --meaning bullish_trend \
    --data /path/to/BTC_USDT_USDT-4h-futures.feather \
    --portion-start 0.5 \
    --portion-end 1.0
```

### Docker

```bash
# Подготовка
mkdir -p distributed/data distributed/reports
cp data/BTC_USDT_USDT-4h-futures.feather distributed/data/

# Сборка
cd distributed
docker-compose build

# Server A
docker-compose run server_a

# Server B (после Server A)
docker-compose run server_b
```

## Структура файлов

```
distributed/
├── docker-compose.yml
├── README.md
├── shared/                      # Общая база знаний (единственное общее!)
│   ├── __init__.py
│   └── meanings.py              # Априорные знания о смыслах
├── server_a/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── server.py                # Learner
│   ├── nobs/                    # NOBS модуль
│   └── shared/                  # Копия общей базы
└── server_b/
    ├── Dockerfile
    ├── requirements.txt
    ├── server.py                # Shaman
    ├── nobs/                    # NOBS модуль
    └── shared/                  # Копия общей базы
```

## Верификация чистоты эксперимента

Чтобы убедиться, что эксперимент "чистый":

1. **Server A** и **Server B** могут запускаться на физически разных машинах
2. Между ними **нет сетевого соединения**
3. Единственный общий файл - `meanings.py` (априорные знания)
4. Название смысла может быть передано человеком устно или на бумажке

## Теоретическое обоснование

См. документы:
- `docs/Shaman_v2.md` - концепция искусственного шамана
- `docs/NOBS_Final/` - NOBS framework LaTeX paper

## Лицензия

MIT
