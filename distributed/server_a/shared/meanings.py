# Semantic Meaning Configurations
# 
# Эти конфиги описывают "смыслы" - семантические паттерны,
# которые могут быть найдены в данных.
#
# Server A обучается распознавать определённый смысл.
# Server B (Shaman) получает ТОЛЬКО название смысла и должен
# самостоятельно найти соответствующую конфигурацию в своём пространстве.

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import json


class SemanticPattern(Enum):
    """Базовые семантические паттерны"""
    
    # Трендовые паттерны
    BULLISH_TREND = "bullish_trend"           # Устойчивый рост
    BEARISH_TREND = "bearish_trend"           # Устойчивое падение
    SIDEWAYS = "sideways"                      # Боковик
    
    # Паттерны волатильности  
    HIGH_VOLATILITY = "high_volatility"        # Высокая волатильность
    LOW_VOLATILITY = "low_volatility"          # Низкая волатильность
    VOLATILITY_EXPANSION = "volatility_expansion"  # Расширение волатильности
    VOLATILITY_CONTRACTION = "volatility_contraction"  # Сжатие волатильности
    
    # Структурные паттерны
    ACCUMULATION = "accumulation"              # Накопление
    DISTRIBUTION = "distribution"              # Распределение
    BREAKOUT = "breakout"                      # Пробой
    REVERSAL = "reversal"                      # Разворот
    
    # Сложные паттерны
    TREND_EXHAUSTION = "trend_exhaustion"      # Истощение тренда
    MOMENTUM_SHIFT = "momentum_shift"          # Смена импульса


@dataclass
class MeaningConfig:
    """
    Конфигурация смысла.
    
    Это ЕДИНСТВЕННОЕ что передаётся между контекстами:
    - Server A читает этот конфиг для обучения
    - Server B читает этот конфиг для поиска
    
    НЕТ передачи данных, эмбеддингов, центроидов - 
    только абстрактное описание смысла.
    """
    
    # Название смысла (ключевое поле)
    meaning_name: str
    
    # Человекочитаемое описание
    description: str
    
    # Ожидаемые характеристики в NOBS-пространстве
    # (это не данные - это априорные знания о том, как смысл
    # должен проявляться в символическом пространстве)
    expected_symbols: Dict[str, str] = field(default_factory=dict)
    # Например: {"dominant": "P", "rare": "I"} для bullish_trend
    
    # Ожидаемый диапазон свободной энергии
    expected_energy_range: str = "medium"  # "low", "medium", "high"
    
    # Ожидаемые морфизмы (структурные переходы)
    expected_morphisms: List[str] = field(default_factory=list)
    # Например: ["PP", "PS", "SP"] для bullish_trend
    
    def to_dict(self) -> dict:
        return {
            "meaning_name": self.meaning_name,
            "description": self.description,
            "expected_symbols": self.expected_symbols,
            "expected_energy_range": self.expected_energy_range,
            "expected_morphisms": self.expected_morphisms
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'MeaningConfig':
        return cls(
            meaning_name=data["meaning_name"],
            description=data["description"],
            expected_symbols=data.get("expected_symbols", {}),
            expected_energy_range=data.get("expected_energy_range", "medium"),
            expected_morphisms=data.get("expected_morphisms", [])
        )
    
    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'MeaningConfig':
        with open(filepath, 'r') as f:
            return cls.from_dict(json.load(f))


# ============================================================
# ПРЕДОПРЕДЕЛЁННЫЕ КОНФИГУРАЦИИ СМЫСЛОВ
# ============================================================

MEANINGS = {
    SemanticPattern.BULLISH_TREND: MeaningConfig(
        meaning_name="bullish_trend",
        description="Устойчивое восходящее движение цены с преобладанием бычьих свечей",
        expected_symbols={"dominant": "P", "secondary": "S", "rare": "I"},
        expected_energy_range="medium",
        expected_morphisms=["PP", "PS", "SP", "PPP"]
    ),
    
    SemanticPattern.BEARISH_TREND: MeaningConfig(
        meaning_name="bearish_trend", 
        description="Устойчивое нисходящее движение цены с преобладанием медвежьих свечей",
        expected_symbols={"dominant": "I", "secondary": "S", "rare": "P"},
        expected_energy_range="medium",
        expected_morphisms=["II", "IS", "SI", "III"]
    ),
    
    SemanticPattern.HIGH_VOLATILITY: MeaningConfig(
        meaning_name="high_volatility",
        description="Период высокой волатильности с резкими движениями в обе стороны",
        expected_symbols={"dominant": "Z", "secondary": "Λ", "rare": "S"},
        expected_energy_range="high",
        expected_morphisms=["ZZ", "ZΛ", "ΛZ", "PI", "IP"]
    ),
    
    SemanticPattern.LOW_VOLATILITY: MeaningConfig(
        meaning_name="low_volatility",
        description="Период низкой волатильности, узкий диапазон, много doji",
        expected_symbols={"dominant": "S", "secondary": "P", "rare": "Z"},
        expected_energy_range="low",
        expected_morphisms=["SS", "SP", "PS", "SSS"]
    ),
    
    SemanticPattern.ACCUMULATION: MeaningConfig(
        meaning_name="accumulation",
        description="Фаза накопления: боковик с постепенным увеличением объёмов",
        expected_symbols={"dominant": "S", "secondary": "P", "rare": "I"},
        expected_energy_range="low",
        expected_morphisms=["SS", "SP", "PS", "SSP"]
    ),
    
    SemanticPattern.DISTRIBUTION: MeaningConfig(
        meaning_name="distribution",
        description="Фаза распределения: боковик после роста с увеличением волатильности",
        expected_symbols={"dominant": "S", "secondary": "I", "rare": "P"},
        expected_energy_range="medium",
        expected_morphisms=["SS", "SI", "IS", "SSI"]
    ),
    
    SemanticPattern.BREAKOUT: MeaningConfig(
        meaning_name="breakout",
        description="Пробой уровня с резким увеличением волатильности и импульсом",
        expected_symbols={"dominant": "Λ", "secondary": "P", "rare": "S"},
        expected_energy_range="high",
        expected_morphisms=["SΛ", "ΛP", "ΛΛ", "SΛP"]
    ),
    
    SemanticPattern.REVERSAL: MeaningConfig(
        meaning_name="reversal",
        description="Разворот тренда с сигналом Omega",
        expected_symbols={"dominant": "Ω", "secondary": "Z", "rare": "S"},
        expected_energy_range="high",
        expected_morphisms=["PΩ", "ΩI", "IΩ", "ΩP", "ZΩ"]
    ),
    
    SemanticPattern.TREND_EXHAUSTION: MeaningConfig(
        meaning_name="trend_exhaustion",
        description="Истощение тренда: замедление движения, появление противоположных сигналов",
        expected_symbols={"dominant": "S", "secondary": "Ω", "rare": "Z"},
        expected_energy_range="medium",
        expected_morphisms=["PS", "PP", "PΩ", "PPΩ", "IIΩ"]
    ),
    
    SemanticPattern.MOMENTUM_SHIFT: MeaningConfig(
        meaning_name="momentum_shift",
        description="Смена импульса: переход от одного направления к другому",
        expected_symbols={"dominant": "Ω", "secondary": "Λ", "rare": "S"},
        expected_energy_range="high",
        expected_morphisms=["PΩI", "IΩP", "ΩΛ", "ΛΩ"]
    ),
}


def get_meaning(pattern: SemanticPattern) -> MeaningConfig:
    """Получить конфигурацию смысла по паттерну"""
    return MEANINGS[pattern]


def get_meaning_by_name(name: str) -> Optional[MeaningConfig]:
    """Получить конфигурацию смысла по имени"""
    for pattern, config in MEANINGS.items():
        if config.meaning_name == name:
            return config
    return None


def list_meanings() -> List[str]:
    """Список всех доступных смыслов"""
    return [config.meaning_name for config in MEANINGS.values()]
