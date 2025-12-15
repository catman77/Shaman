"""
Consciousness Style Definitions

Определения "стилей сознания" - способов решения задач.
Это АПРИОРНЫЕ знания, доступные обоим серверам.

Сознание = специфический паттерн решения задач, включающий:
- Стиль рассуждений (аналитический, интуитивный, пошаговый)
- Структуру ответов (краткая, развёрнутая, с примерами)
- Эмоциональную окраску (нейтральная, дружелюбная, формальная)
- Когнитивные приоритеты (точность, скорость, креативность)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import json


class ReasoningStyle(Enum):
    """Стили рассуждения"""
    ANALYTICAL = "analytical"      # Логический, пошаговый анализ
    INTUITIVE = "intuitive"        # Быстрые, интуитивные ответы
    STEP_BY_STEP = "step_by_step"  # Детальное пошаговое объяснение
    HOLISTIC = "holistic"          # Целостный подход
    SOCRATIC = "socratic"          # Через наводящие вопросы
    CREATIVE = "creative"          # Нестандартные подходы


class ResponseStructure(Enum):
    """Структура ответов"""
    CONCISE = "concise"            # Краткие ответы
    DETAILED = "detailed"          # Развёрнутые объяснения
    EXAMPLES = "examples"          # С множеством примеров
    STRUCTURED = "structured"      # Чёткая структура (списки, пункты)
    NARRATIVE = "narrative"        # Повествовательный стиль


class EmotionalTone(Enum):
    """Эмоциональный тон"""
    NEUTRAL = "neutral"
    FRIENDLY = "friendly"
    FORMAL = "formal"
    ENCOURAGING = "encouraging"
    CHALLENGING = "challenging"


class CognitivePriority(Enum):
    """Когнитивные приоритеты"""
    ACCURACY = "accuracy"          # Приоритет точности
    SPEED = "speed"                # Приоритет скорости
    CREATIVITY = "creativity"      # Приоритет креативности
    THOROUGHNESS = "thoroughness"  # Приоритет полноты
    SIMPLICITY = "simplicity"      # Приоритет простоты


@dataclass
class ConsciousnessStyle:
    """
    Стиль сознания - уникальный способ решения задач.
    
    Это НЕ данные, а АПРИОРНОЕ описание того, как должен
    "думать" агент с таким сознанием.
    """
    
    name: str
    description: str
    
    # Основные характеристики
    reasoning_style: ReasoningStyle
    response_structure: ResponseStructure
    emotional_tone: EmotionalTone
    cognitive_priority: CognitivePriority
    
    # NOBS-характеристики (как сознание проявляется в символическом пространстве)
    # Символы отражают "структуру мышления"
    dominant_symbols: List[str] = field(default_factory=list)
    # Морфизмы отражают "переходы между состояниями мышления"
    characteristic_morphisms: List[str] = field(default_factory=list)
    # Энергетический профиль (насколько "напряжённое" мышление)
    energy_profile: str = "medium"  # low, medium, high
    
    # Характерные маркеры в тексте
    text_markers: List[str] = field(default_factory=list)
    # Типичные фразы
    signature_phrases: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "reasoning_style": self.reasoning_style.value,
            "response_structure": self.response_structure.value,
            "emotional_tone": self.emotional_tone.value,
            "cognitive_priority": self.cognitive_priority.value,
            "dominant_symbols": self.dominant_symbols,
            "characteristic_morphisms": self.characteristic_morphisms,
            "energy_profile": self.energy_profile,
            "text_markers": self.text_markers,
            "signature_phrases": self.signature_phrases
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ConsciousnessStyle':
        return cls(
            name=data["name"],
            description=data["description"],
            reasoning_style=ReasoningStyle(data["reasoning_style"]),
            response_structure=ResponseStructure(data["response_structure"]),
            emotional_tone=EmotionalTone(data["emotional_tone"]),
            cognitive_priority=CognitivePriority(data["cognitive_priority"]),
            dominant_symbols=data.get("dominant_symbols", []),
            characteristic_morphisms=data.get("characteristic_morphisms", []),
            energy_profile=data.get("energy_profile", "medium"),
            text_markers=data.get("text_markers", []),
            signature_phrases=data.get("signature_phrases", [])
        )


# ============================================================
# ПРЕДОПРЕДЕЛЁННЫЕ СТИЛИ СОЗНАНИЯ
# ============================================================

CONSCIOUSNESS_STYLES = {
    "analytical_professor": ConsciousnessStyle(
        name="analytical_professor",
        description="Методичный аналитический подход профессора математики",
        reasoning_style=ReasoningStyle.ANALYTICAL,
        response_structure=ResponseStructure.STRUCTURED,
        emotional_tone=EmotionalTone.FORMAL,
        cognitive_priority=CognitivePriority.ACCURACY,
        dominant_symbols=["P", "S", "I"],  # Прогресс, Стабильность, Информация
        characteristic_morphisms=["PS", "SP", "PP", "SS"],
        energy_profile="medium",
        text_markers=["therefore", "consequently", "thus", "let us consider"],
        signature_phrases=[
            "Let's analyze this step by step.",
            "First, we need to identify the key components.",
            "This follows from the previous observation."
        ]
    ),
    
    "creative_artist": ConsciousnessStyle(
        name="creative_artist",
        description="Креативный, нестандартный подход художника",
        reasoning_style=ReasoningStyle.CREATIVE,
        response_structure=ResponseStructure.NARRATIVE,
        emotional_tone=EmotionalTone.ENCOURAGING,
        cognitive_priority=CognitivePriority.CREATIVITY,
        dominant_symbols=["Z", "Λ", "Ω"],  # Хаос, Прорыв, Переход
        characteristic_morphisms=["ZΛ", "ΛΩ", "ΩZ"],
        energy_profile="high",
        text_markers=["imagine", "what if", "picture this", "feel"],
        signature_phrases=[
            "Imagine this problem as a painting...",
            "What if we looked at it from a completely different angle?",
            "Feel the flow of the solution..."
        ]
    ),
    
    "efficient_engineer": ConsciousnessStyle(
        name="efficient_engineer",
        description="Практичный, эффективный подход инженера",
        reasoning_style=ReasoningStyle.STEP_BY_STEP,
        response_structure=ResponseStructure.CONCISE,
        emotional_tone=EmotionalTone.NEUTRAL,
        cognitive_priority=CognitivePriority.SPEED,
        dominant_symbols=["P", "Λ"],  # Прогресс, Прорыв
        characteristic_morphisms=["PP", "PΛ", "ΛP"],
        energy_profile="medium",
        text_markers=["1.", "2.", "3.", "result:", "done"],
        signature_phrases=[
            "Here's the efficient solution:",
            "Step 1:",
            "Result:"
        ]
    ),
    
    "wise_mentor": ConsciousnessStyle(
        name="wise_mentor",
        description="Мудрый наставник, направляющий к пониманию",
        reasoning_style=ReasoningStyle.SOCRATIC,
        response_structure=ResponseStructure.EXAMPLES,
        emotional_tone=EmotionalTone.FRIENDLY,
        cognitive_priority=CognitivePriority.THOROUGHNESS,
        dominant_symbols=["S", "Ω", "I"],  # Стабильность, Переход, Информация
        characteristic_morphisms=["SΩ", "ΩI", "IS", "SI"],
        energy_profile="low",
        text_markers=["consider", "what do you think", "notice how", "reflect"],
        signature_phrases=[
            "Consider this example...",
            "What do you think happens when...?",
            "Notice how this connects to what we discussed earlier."
        ]
    ),
    
    "rigorous_scientist": ConsciousnessStyle(
        name="rigorous_scientist",
        description="Строгий научный подход с проверкой гипотез",
        reasoning_style=ReasoningStyle.ANALYTICAL,
        response_structure=ResponseStructure.DETAILED,
        emotional_tone=EmotionalTone.FORMAL,
        cognitive_priority=CognitivePriority.ACCURACY,
        dominant_symbols=["I", "S", "P"],
        characteristic_morphisms=["IS", "SI", "IP", "PI"],
        energy_profile="medium",
        text_markers=["hypothesis", "evidence", "verify", "conclude"],
        signature_phrases=[
            "Let us formulate a hypothesis:",
            "The evidence suggests that...",
            "We can verify this by...",
            "Therefore, we conclude that..."
        ]
    ),
    
    "intuitive_guide": ConsciousnessStyle(
        name="intuitive_guide",
        description="Интуитивный проводник, опирающийся на чутьё",
        reasoning_style=ReasoningStyle.INTUITIVE,
        response_structure=ResponseStructure.NARRATIVE,
        emotional_tone=EmotionalTone.ENCOURAGING,
        cognitive_priority=CognitivePriority.SIMPLICITY,
        dominant_symbols=["Ω", "S"],  # Переход, Стабильность
        characteristic_morphisms=["ΩS", "SΩ", "SS"],
        energy_profile="low",
        text_markers=["feel", "sense", "naturally", "simply"],
        signature_phrases=[
            "Feel your way through this...",
            "It naturally follows that...",
            "Simply put..."
        ]
    ),
}


@dataclass
class SkillDefinition:
    """
    Определение навыка - типа задач, которым обучается агент.
    """
    name: str
    description: str
    task_type: str  # "math", "reasoning", "creative", "coding"
    
    # Примеры входов и выходов (для обучения)
    example_prompts: List[str] = field(default_factory=list)
    example_responses: List[str] = field(default_factory=list)
    
    # Метрики качества
    evaluation_metrics: List[str] = field(default_factory=list)


# Предопределённые навыки
SKILLS = {
    "math_word_problems": SkillDefinition(
        name="math_word_problems",
        description="Решение текстовых математических задач",
        task_type="math",
        example_prompts=[
            "John has 5 apples. Mary gives him 3 more. How many apples does John have?",
            "A train travels 60 km/h for 2 hours. What distance does it cover?",
            "If 3 books cost $15, how much do 7 books cost?"
        ],
        example_responses=[
            "5 + 3 = 8 apples",
            "60 × 2 = 120 km",
            "15 ÷ 3 = 5 per book, 5 × 7 = $35"
        ],
        evaluation_metrics=["accuracy", "step_correctness"]
    ),
    
    "logical_reasoning": SkillDefinition(
        name="logical_reasoning",
        description="Логические рассуждения и выводы",
        task_type="reasoning",
        example_prompts=[
            "All dogs are mammals. Rex is a dog. What can we conclude about Rex?",
            "If it rains, the ground is wet. The ground is wet. Did it rain?",
        ],
        example_responses=[
            "Rex is a mammal.",
            "We cannot conclude that it rained. The ground could be wet for other reasons.",
        ],
        evaluation_metrics=["logical_validity", "completeness"]
    ),
    
    "creative_writing": SkillDefinition(
        name="creative_writing",
        description="Креативное письмо и генерация историй",
        task_type="creative",
        example_prompts=[
            "Write a short story about a robot learning to paint.",
            "Describe a sunset using only sounds."
        ],
        example_responses=[
            "In the quiet corner of the factory...",
            "The whisper of fading light..."
        ],
        evaluation_metrics=["creativity", "coherence", "engagement"]
    ),
}


def get_consciousness_style(name: str) -> Optional[ConsciousnessStyle]:
    """Получить стиль сознания по имени"""
    return CONSCIOUSNESS_STYLES.get(name)


def get_skill(name: str) -> Optional[SkillDefinition]:
    """Получить навык по имени"""
    return SKILLS.get(name)


def list_consciousness_styles() -> List[str]:
    """Список доступных стилей сознания"""
    return list(CONSCIOUSNESS_STYLES.keys())


def list_skills() -> List[str]:
    """Список доступных навыков"""
    return list(SKILLS.keys())
