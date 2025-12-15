"""
SymExtractor - извлечение симструктур σ из текста

Симструктура σ - символьное представление "стиля/личности" агента:
- AST-подобная структура
- Описывает паттерны рассуждений, характерные обороты, структуру ответов

Для MVP используем rule-based подход на основе regex.
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import Counter


@dataclass
class SymNode:
    """Узел симструктуры."""
    type: str
    label: str = ""
    value: Any = None
    children: List["SymNode"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует в словарь (для JSON-сериализации)."""
        return {
            "type": self.type,
            "label": self.label,
            "value": self.value,
            "children": [c.to_dict() for c in self.children],
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SymNode":
        """Создаёт из словаря."""
        return cls(
            type=d["type"],
            label=d.get("label", ""),
            value=d.get("value"),
            children=[cls.from_dict(c) for c in d.get("children", [])],
            metadata=d.get("metadata", {})
        )


@dataclass 
class SymStructure:
    """Полная симструктура эпизода/агента."""
    root: SymNode
    patterns_found: Dict[str, int] = field(default_factory=dict)
    features: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "root": self.root.to_dict(),
            "patterns_found": self.patterns_found,
            "features": self.features
        }


class PatternMatcher:
    """Сопоставитель паттернов для извлечения σ."""
    
    def __init__(self):
        # Базовые паттерны для MVP
        self.patterns = {
            # Структурные паттерны
            "ANALYSIS_START": [
                r"(?i)^(анализ|рассмотр|исследу|изуч)",
                r"(?i)(анализируя|рассматривая)",
                r"(?i)^(let me analyze|analyzing|considering)"
            ],
            "ENUMERATION": [
                r"^\s*\d+[.)]\s+",
                r"^\s*[-•]\s+",
                r"(?i)(во-первых|во-вторых|в-третьих)",
                r"(?i)(firstly|secondly|thirdly)"
            ],
            "CONCLUSION": [
                r"(?i)(вывод|резюме|итог|заключени|следовательно|таким образом)",
                r"(?i)(in conclusion|to summarize|therefore|thus|hence)"
            ],
            
            # Стилистические паттерны
            "METAPHOR": [
                r"(?i)(как|подобно|словно|наподобие)\s+\w+",
                r"(?i)(like|as if|similar to)\s+\w+"
            ],
            "QUESTION": [
                r"\?",
                r"(?i)(зачем|почему|как|что если)",
                r"(?i)(why|how|what if)"
            ],
            "HEDGING": [
                r"(?i)(возможно|вероятно|может быть|пожалуй)",
                r"(?i)(perhaps|maybe|possibly|might)"
            ],
            "CERTAINTY": [
                r"(?i)(точно|определённо|безусловно|очевидно)",
                r"(?i)(certainly|definitely|obviously|clearly)"
            ],
            
            # Эмоциональные паттерны
            "EXCLAMATION": [
                r"!",
                r"(?i)(отлично|замечательно|интересно)!"
            ],
            "EMPATHY": [
                r"(?i)(понима[юе]|чувству[юе])",
                r"(?i)(i understand|i feel)"
            ],
            
            # Логические паттерны
            "CONDITIONAL": [
                r"(?i)(если|при условии|в случае)",
                r"(?i)(if|when|in case)"
            ],
            "CAUSATION": [
                r"(?i)(потому что|поскольку|так как|из-за)",
                r"(?i)(because|since|due to)"
            ],
            "CONTRAST": [
                r"(?i)(однако|но|тем не менее|впрочем)",
                r"(?i)(however|but|nevertheless|although)"
            ],
            
            # Референцные паттерны
            "EXAMPLE": [
                r"(?i)(например|к примеру|допустим)",
                r"(?i)(for example|for instance|such as)"
            ],
            "SELF_REFERENCE": [
                r"(?i)(я думаю|по моему мнению|мне кажется)",
                r"(?i)(i think|in my opinion|i believe)"
            ]
        }
        
        # Компилируем regex
        self.compiled_patterns = {}
        for name, regexes in self.patterns.items():
            self.compiled_patterns[name] = [
                re.compile(r, re.MULTILINE) for r in regexes
            ]
    
    def find_patterns(self, text: str) -> Dict[str, int]:
        """
        Находит все паттерны в тексте.
        
        Returns:
            словарь {pattern_name: count}
        """
        counts = Counter()
        
        for pattern_name, compiled_list in self.compiled_patterns.items():
            for regex in compiled_list:
                matches = regex.findall(text)
                counts[pattern_name] += len(matches)
        
        return dict(counts)


class SymExtractor:
    """
    Извлекает симструктуры σ из текстов агента.
    
    Rule-based версия для MVP.
    """
    
    def __init__(self):
        self.matcher = PatternMatcher()
        
    def extract(self, text: str) -> SymStructure:
        """
        Извлекает симструктуру из текста.
        
        Args:
            text: текст ответа агента
            
        Returns:
            SymStructure с деревом паттернов и метриками
        """
        # Находим паттерны
        patterns = self.matcher.find_patterns(text)
        
        # Строим дерево σ
        root = self._build_tree(text, patterns)
        
        # Вычисляем features
        features = self._compute_features(text, patterns)
        
        return SymStructure(
            root=root,
            patterns_found=patterns,
            features=features
        )
    
    def _build_tree(self, text: str, patterns: Dict[str, int]) -> SymNode:
        """Строит дерево симструктуры."""
        root = SymNode(type="ROOT", label="response")
        
        # Группируем паттерны по категориям
        structural = ["ANALYSIS_START", "ENUMERATION", "CONCLUSION"]
        stylistic = ["METAPHOR", "QUESTION", "HEDGING", "CERTAINTY"]
        emotional = ["EXCLAMATION", "EMPATHY"]
        logical = ["CONDITIONAL", "CAUSATION", "CONTRAST"]
        referential = ["EXAMPLE", "SELF_REFERENCE"]
        
        categories = [
            ("STRUCTURAL", structural),
            ("STYLISTIC", stylistic),
            ("EMOTIONAL", emotional),
            ("LOGICAL", logical),
            ("REFERENTIAL", referential)
        ]
        
        for cat_name, cat_patterns in categories:
            cat_node = SymNode(type="CATEGORY", label=cat_name)
            
            for pname in cat_patterns:
                if patterns.get(pname, 0) > 0:
                    pattern_node = SymNode(
                        type="PATTERN",
                        label=pname,
                        value=patterns[pname]
                    )
                    cat_node.children.append(pattern_node)
            
            if cat_node.children:
                root.children.append(cat_node)
        
        return root
    
    def _compute_features(
        self, 
        text: str, 
        patterns: Dict[str, int]
    ) -> Dict[str, float]:
        """Вычисляет числовые features из паттернов."""
        total_patterns = sum(patterns.values())
        text_len = len(text)
        word_count = len(text.split())
        
        # Нормализованные метрики
        features = {
            # Общая плотность паттернов
            "pattern_density": total_patterns / max(word_count, 1),
            
            # Структурированность (доля структурных паттернов)
            "structuredness": (
                patterns.get("ENUMERATION", 0) + 
                patterns.get("CONCLUSION", 0)
            ) / max(total_patterns, 1),
            
            # Аналитичность
            "analyticity": (
                patterns.get("ANALYSIS_START", 0) +
                patterns.get("CAUSATION", 0) +
                patterns.get("CONDITIONAL", 0)
            ) / max(total_patterns, 1),
            
            # Образность (метафоричность)
            "imagery": patterns.get("METAPHOR", 0) / max(total_patterns, 1),
            
            # Эмоциональность
            "emotionality": (
                patterns.get("EXCLAMATION", 0) +
                patterns.get("EMPATHY", 0)
            ) / max(total_patterns, 1),
            
            # Уверенность
            "confidence": (
                patterns.get("CERTAINTY", 0) - 
                patterns.get("HEDGING", 0)
            ) / max(total_patterns, 1),
            
            # Диалогичность (вопросы, примеры)
            "dialogicity": (
                patterns.get("QUESTION", 0) +
                patterns.get("EXAMPLE", 0)
            ) / max(total_patterns, 1),
            
            # Длина ответа (нормализованная)
            "response_length": min(word_count / 200, 1.0)
        }
        
        return features
    
    def extract_batch(self, texts: List[str]) -> List[SymStructure]:
        """Извлекает симструктуры из батча текстов."""
        return [self.extract(t) for t in texts]
    
    def compute_similarity(
        self, 
        sigma1: SymStructure, 
        sigma2: SymStructure
    ) -> float:
        """
        Вычисляет сходство двух симструктур.
        
        Returns:
            similarity ∈ [0, 1], где 1 = идентичные структуры
        """
        # Сравниваем features
        f1 = sigma1.features
        f2 = sigma2.features
        
        if not f1 or not f2:
            return 0.0
        
        # Косинусное сходство по feature-векторам
        keys = set(f1.keys()) | set(f2.keys())
        v1 = [f1.get(k, 0) for k in keys]
        v2 = [f2.get(k, 0) for k in keys]
        
        import numpy as np
        v1 = np.array(v1)
        v2 = np.array(v2)
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        
        return float(np.dot(v1, v2) / (norm1 * norm2))
    
    def compute_distance(
        self,
        sigma1: SymStructure,
        sigma2: SymStructure
    ) -> float:
        """
        Вычисляет расстояние между симструктурами.
        
        Returns:
            distance ∈ [0, 1], где 0 = идентичные структуры
        """
        return 1.0 - self.compute_similarity(sigma1, sigma2)
    
    def compute_rarity(
        self,
        sigma: SymStructure,
        population_stats: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Оценивает "редкость" симструктуры относительно популяции.
        
        Args:
            sigma: симструктура для оценки
            population_stats: средние значения features в популяции
            
        Returns:
            rarity ∈ [0, 1], где 1 = очень редкая структура
        """
        if population_stats is None:
            # Без статистики популяции используем heuristic
            # "Редкие" = высокая pattern_density + нестандартное распределение
            f = sigma.features
            
            # Чем больше отклонение от "среднего" профиля, тем более редкий
            # Средний профиль: structuredness ≈ 0.3, analyticity ≈ 0.3, etc.
            default_profile = {
                "structuredness": 0.3,
                "analyticity": 0.3,
                "imagery": 0.1,
                "emotionality": 0.1,
                "confidence": 0.0,
                "dialogicity": 0.2
            }
            
            import numpy as np
            deviations = []
            for key, default in default_profile.items():
                actual = f.get(key, 0)
                deviations.append(abs(actual - default))
            
            # Среднее отклонение как мера редкости
            rarity = min(np.mean(deviations) * 2, 1.0)
            return float(rarity)
        
        else:
            # Используем статистику популяции
            import numpy as np
            deviations = []
            for key, pop_mean in population_stats.items():
                actual = sigma.features.get(key, 0)
                deviations.append(abs(actual - pop_mean))
            
            return float(min(np.mean(deviations) * 2, 1.0))


def structural_distance(sigma1: Dict, sigma2: Dict) -> float:
    """
    Вычисляет структурное расстояние между двумя σ (в dict-формате).
    
    Упрощённая версия для совместимости с JSON-сериализованными структурами.
    """
    extractor = SymExtractor()
    
    # Если это dict, конвертируем в SymStructure
    if isinstance(sigma1, dict):
        s1 = SymStructure(
            root=SymNode.from_dict(sigma1.get("root", {"type": "ROOT"})),
            patterns_found=sigma1.get("patterns_found", {}),
            features=sigma1.get("features", {})
        )
    else:
        s1 = sigma1
    
    if isinstance(sigma2, dict):
        s2 = SymStructure(
            root=SymNode.from_dict(sigma2.get("root", {"type": "ROOT"})),
            patterns_found=sigma2.get("patterns_found", {}),
            features=sigma2.get("features", {})
        )
    else:
        s2 = sigma2
    
    return extractor.compute_distance(s1, s2)
