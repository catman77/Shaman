# -*- coding: utf-8 -*-
"""
Server B - Shaman with NOBS-Based Consciousness Resonance (V2)

Улучшенная версия Шамана:
1. Использует NOBS пространство на Bitcoin данных
2. Ищет сознание через резонанс в NOBS пространстве
3. Передаёт найденное сознание своей модели (другая архитектура!)

КЛЮЧЕВОЕ: Server B получает ТОЛЬКО название сознания!
- Нет данных от Server A
- Нет модели Server A
- Нет активаций Server A

Шаман использует АПРИОРНОЕ знание о стиле сознания
и находит его в NOBS пространстве Bitcoin данных.
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
import time
import random

import torch
import torch.nn as nn
import numpy as np

# Add parent path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.nobs_consciousness import (
    NOBSConsciousnessSpace, ConsciousnessSignature, ConsciousnessConfig,
    get_consciousness_config, list_consciousness_styles, CONSCIOUSNESS_CONFIGS
)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types"""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# Check for transformers
try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not installed. Using mock model.")


@dataclass
class ResonanceConfig:
    """Конфигурация резонансного поиска"""
    num_samples: int = 2000        # Количество сэмплов для поиска
    refinement_samples: int = 500  # Дополнительные сэмплы для уточнения
    min_resonance: float = 0.6     # Минимальный порог резонанса


@dataclass
class ResonanceResult:
    """Результат резонансного поиска"""
    consciousness_name: str
    consciousness_found: bool
    resonance_score: float
    signature: Dict[str, Any]
    search_iterations: int
    search_time: float
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TransferResult:
    """Результат переноса сознания"""
    style_transfer_score: float
    skill_transfer_score: float
    overall_score: float
    test_responses: List[Dict[str, str]]
    
    def to_dict(self) -> dict:
        return {
            "style_transfer_score": self.style_transfer_score,
            "skill_transfer_score": self.skill_transfer_score,
            "overall_score": self.overall_score,
            "test_responses": self.test_responses
        }


class ConsciousnessDecoder:
    """
    Декодер сознания - настраивает модель на найденную сигнатуру.
    
    Использует NOBS сигнатуру для:
    1. Генерации промптов с few-shot примерами в нужном стиле
    2. Настройки температуры генерации
    3. Интерпретации NOBS паттернов в стилистические указания
    """
    
    def __init__(self, config: ConsciousnessConfig, signature: ConsciousnessSignature):
        self.config = config
        self.signature = signature
        
    def generate_style_prompt(self) -> str:
        """
        Сгенерировать промпт с few-shot примерами в стиле целевого сознания.
        """
        # Анализируем сигнатуру
        dominant_symbols = sorted(
            self.signature.symbol_distribution.items(),
            key=lambda x: -x[1]
        )[:3]
        
        # Интерпретируем символы в стилистические указания
        style_descriptors = []
        for sym, weight in dominant_symbols:
            if sym == 'P' and weight > 0.25:
                style_descriptors.append("progressive and building towards solution")
            elif sym == 'S' and weight > 0.15:
                style_descriptors.append("careful with verification")
            elif sym == 'I' and weight > 0.2:
                style_descriptors.append("methodical")
            elif sym == 'Z' and weight > 0.1:
                style_descriptors.append("contemplative")
            elif sym == 'Ω' and weight > 0.1:
                style_descriptors.append("focused on conclusions")
            elif sym == 'Λ' and weight > 0.1:
                style_descriptors.append("transitioning between approaches")
        
        # Энергетика
        if self.signature.free_energy < -0.3:
            style_descriptors.append("highly structured")
        elif self.signature.free_energy > 0.3:
            style_descriptors.append("flexible")
        
        style_desc = ", ".join(style_descriptors) if style_descriptors else "thoughtful"
        
        # Создаём few-shot примеры на основе стиля
        examples = self._generate_few_shot_examples()
        
        prompt = f"""{self.config.prompt_template}

Your approach is {style_desc}.

Here are examples of how you solve problems:

{examples}

Now solve the following problem in the same style:
"""
        return prompt
    
    def _generate_few_shot_examples(self) -> str:
        """Генерирует few-shot примеры в стиле сознания."""
        name = self.config.name
        
        if name == "analytical_professor":
            return """Example 1:
Question: John has 5 apples. Mary gives him 3 more. How many apples does John have?
Answer: Let's analyze this step by step.
Step 1: John starts with 5 apples.
Step 2: Mary gives him 3 more apples.
Step 3: We add: 5 + 3 = 8
The answer is 8.

Example 2:
Question: A car travels 40 km/h for 2 hours. What distance does it cover?
Answer: Let's analyze this step by step.
Step 1: Speed = 40 km/h
Step 2: Time = 2 hours
Step 3: Distance = Speed × Time = 40 × 2 = 80 km
The answer is 80 km."""

        elif name == "creative_solver":
            return """Example 1:
Question: John has 5 apples. Mary gives him 3 more. How many apples does John have?
Answer: Interesting! Think of it like filling a basket.
John's basket has 5 apples, and Mary adds 3 more.
Aha! The basket now holds 5 + 3 = 8 apples.
The elegant answer is 8.

Example 2:
Question: A car travels 40 km/h for 2 hours. What distance does it cover?
Answer: What if we imagine the car as a runner?
Running at 40 km each hour, for 2 hours...
That's like running the same track twice!
The surprising answer is 40 × 2 = 80 km."""

        elif name == "intuitive_guesser":
            return """Example 1:
Question: John has 5 apples. Mary gives him 3 more. How many apples does John have?
Answer: I sense this is simple addition.
Quick answer: 5 + 3 = 8 apples.

Example 2:
Question: A car travels 40 km/h for 2 hours. What distance does it cover?
Answer: Intuitively, speed times time gives distance.
Simply put: 40 × 2 = 80 km."""

        elif name == "pedantic_engineer":
            return """Example 1:
Question: John has 5 apples. Mary gives him 3 more. How many apples does John have?
Answer: Let me verify this carefully.
Assumption: We're counting whole apples.
Checking: Initial = 5, Added = 3
Calculation: 5 + 3 = 8
Double-checking: 8 - 3 = 5 ✓
Confirmed: The answer is 8 apples.

Example 2:
Question: A car travels 40 km/h for 2 hours. What distance does it cover?
Answer: Let me verify this carefully.
Assumption: Constant speed.
Checking units: km/h × h = km ✓
Calculation: 40 × 2 = 80
Confirmed: 80 km."""

        elif name == "philosophical_thinker":
            return """Example 1:
Question: John has 5 apples. Mary gives him 3 more. How many apples does John have?
Answer: Let us contemplate the essence of this problem.
The fundamental nature of addition reveals how quantities combine.
On a deeper level, 5 and 3 unite to form 8.
Thus we see: John has 8 apples.

Example 2:
Question: A car travels 40 km/h for 2 hours. What distance does it cover?
Answer: Let us contemplate motion through space and time.
This reveals the relationship: distance = speed × time.
The fundamental nature shows us: 40 × 2 = 80.
Thus we see: 80 km."""

        else:
            return """Example:
Question: John has 5 apples. Mary gives him 3 more. How many apples does John have?
Answer: 5 + 3 = 8. The answer is 8 apples."""
    
    def get_generation_params(self) -> Dict[str, Any]:
        """
        Получить параметры генерации на основе сигнатуры.
        """
        # Базовые параметры
        params = {
            "do_sample": True,
            "top_p": 0.9,
            "repetition_penalty": 1.1
        }
        
        # Температура на основе энтропии сигнатуры
        # Высокая энтропия → более творческие ответы
        if self.signature.entropy > 0.6:
            params["temperature"] = 0.9
        elif self.signature.entropy < 0.4:
            params["temperature"] = 0.6
        else:
            params["temperature"] = 0.7
        
        # Top-k на основе свободной энергии
        # Низкая энергия → более сфокусированные ответы
        if self.signature.free_energy < -0.3:
            params["top_k"] = 30
        elif self.signature.free_energy > 0.3:
            params["top_k"] = 70
        else:
            params["top_k"] = 50
        
        return params


class ShamanAgent:
    """
    Агент Шамана - принимает сознание через NOBS резонанс.
    
    Использует ДРУГУЮ архитектуру (gpt2-medium vs distilgpt2)
    и не имеет доступа к данным Server A.
    """
    
    def __init__(self, model_name: str = "gpt2-medium"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.decoder: Optional[ConsciousnessDecoder] = None
        self.style_prompt: str = ""
        
        if HAS_TRANSFORMERS:
            self._load_model()
    
    def _load_model(self):
        """Загрузить модель."""
        print(f"Loading model: {self.model_name}")
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")
        print(f"  Hidden size: {self.model.config.hidden_size}")
        print(f"  Num layers: {self.model.config.n_layer}")
        print("NOTE: This is a DIFFERENT architecture from Server A!")
    
    def receive_consciousness(
        self,
        config: ConsciousnessConfig,
        signature: ConsciousnessSignature
    ):
        """
        Принять сознание через декодированную сигнатуру.
        
        НЕ использует данные от Server A!
        Использует только:
        - Конфигурацию сознания (априорное знание)
        - NOBS сигнатуру, найденную через резонанс
        """
        self.decoder = ConsciousnessDecoder(config, signature)
        self.style_prompt = self.decoder.generate_style_prompt()
        
        print(f"\nConsciousness received: {config.name}")
        print(f"Style prompt generated ({len(self.style_prompt)} chars)")
    
    def generate_response(self, question: str, max_length: int = 250) -> str:
        """Сгенерировать ответ в стиле полученного сознания."""
        if not HAS_TRANSFORMERS or self.model is None:
            return "Mock response in transferred style"
        
        self.model.eval()
        
        # Формируем полный промпт с few-shot примерами
        full_prompt = f"{self.style_prompt}\nQuestion: {question}\n\nAnswer:"
        
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024  # Увеличено для few-shot примеров
        ).to(self.device)
        
        # Параметры генерации от декодера
        gen_params = self.decoder.get_generation_params() if self.decoder else {}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **gen_params
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer part - get the last Answer: section
        if "Answer:" in response:
            parts = response.split("Answer:")
            response = parts[-1].strip()
        
        return response[:600]
    
    def evaluate_style(self, config: ConsciousnessConfig, responses: List[str]) -> float:
        """Оценить соответствие ответов целевому стилю."""
        if not responses:
            return 0.0
        
        total_score = 0.0
        for response in responses:
            response_lower = response.lower()
            
            # Basic style markers from config
            markers_found = 0
            for pattern in config.response_patterns:
                if pattern.lower() in response_lower:
                    markers_found += 1
            
            # Additional semantic markers based on consciousness type
            semantic_markers = {
                "analytical_professor": ["step", "calculate", "therefore", "result", "answer is"],
                "creative_solver": ["interesting", "imagine", "aha", "elegant", "think of"],
                "intuitive_guesser": ["sense", "quick", "intuitively", "simply", "feel"],
                "pedantic_engineer": ["verify", "check", "assumption", "confirm", "double"],
                "philosophical_thinker": ["contemplate", "essence", "deeper", "nature", "thus"]
            }
            
            bonus_markers = semantic_markers.get(config.name, [])
            for marker in bonus_markers:
                if marker in response_lower:
                    markers_found += 0.5
            
            # Check for structural elements
            if "step" in response_lower and any(f"step {i}" in response_lower for i in range(1, 5)):
                markers_found += 1  # Bonus for step-by-step structure
            
            if "=" in response:
                markers_found += 0.5  # Bonus for showing calculations
            
            # Score based on markers
            max_markers = max(1, len(config.response_patterns) // 2 + 2)
            total_score += min(1.0, markers_found / max_markers)
        
        return total_score / len(responses)
    
    def evaluate_skill(self, questions: List[Tuple[str, int]]) -> float:
        """
        Оценить точность ответов на математические задачи.
        
        Args:
            questions: List of (question, correct_answer) tuples
        """
        if not questions:
            return 0.0
        
        correct = 0
        for question, expected in questions:
            response = self.generate_response(question)
            
            # Try to extract number from response
            import re
            numbers = re.findall(r'\b(\d+)\b', response)
            
            if numbers:
                # Check if any extracted number matches expected
                for num_str in numbers[-5:]:  # Check last few numbers
                    if int(num_str) == expected:
                        correct += 1
                        break
        
        return correct / len(questions)


class ServerB:
    """
    Server B - Шаман с NOBS резонансом.
    
    Алгоритм:
    1. Получает ТОЛЬКО название сознания (строку!)
    2. Загружает NOBS пространство на Bitcoin данных
    3. Ищет резонанс с целевым сознанием в NOBS пространстве
    4. Декодирует найденную сигнатуру в параметры модели
    5. Генерирует ответы в стиле найденного сознания
    
    ВАЖНО: Никаких данных от Server A!
    """
    
    def __init__(
        self,
        consciousness_name: str,
        model_name: str = "gpt2-medium",
        resonance_config: Optional[ResonanceConfig] = None
    ):
        self.consciousness_name = consciousness_name
        self.model_name = model_name
        self.resonance_config = resonance_config or ResonanceConfig()
        
        # Load consciousness config (SHARED a priori knowledge)
        self.consciousness = get_consciousness_config(consciousness_name)
        
        # Initialize NOBS space
        self.nobs_space = NOBSConsciousnessSpace()
        
        # Agent
        self.agent: Optional[ShamanAgent] = None
        
        # Results
        self.resonance_result: Optional[ResonanceResult] = None
        self.transfer_result: Optional[TransferResult] = None
        self.found_signature: Optional[ConsciousnessSignature] = None
        
    def initialize(self):
        """Инициализировать компоненты."""
        print("="*60)
        print("SERVER B - SHAMAN WITH NOBS RESONANCE (V2)")
        print("="*60)
        print(f"\nTarget consciousness: {self.consciousness_name}")
        print(f"Description: {self.consciousness.description}")
        print(f"Model: {self.model_name} (DIFFERENT from Server A!)")
        
        print("\n" + "="*60)
        print("CRITICAL: NO DATA FROM SERVER A!")
        print("Only using consciousness NAME from shared knowledge.")
        print("="*60)
        
        # Initialize NOBS space on Bitcoin data
        print("\nInitializing NOBS space on Bitcoin data...")
        self.nobs_space.fit()
        
        # Create agent
        self.agent = ShamanAgent(self.model_name)
        
        return self
    
    def find_consciousness_resonance(self) -> ResonanceResult:
        """
        Найти сознание через резонанс в NOBS пространстве.
        
        Шаман сэмплирует NOBS пространство и ищет паттерны,
        соответствующие априорному описанию целевого сознания.
        """
        print("\n" + "="*60)
        print("CONSCIOUSNESS RESONANCE SEARCH")
        print("="*60)
        print(f"\nTarget: {self.consciousness_name}")
        print(f"Samples to search: {self.resonance_config.num_samples}")
        print("\nUsing ONLY a priori knowledge about consciousness style!")
        print("NO data from Server A is used!\n")
        
        start_time = time.time()
        
        # Phase 1: Main search
        print("Phase 1: Main resonance search...")
        signature, score = self.nobs_space.find_resonance(
            self.consciousness,
            num_samples=self.resonance_config.num_samples
        )
        
        # Phase 2: Refinement if needed
        if score < 0.7:
            print(f"\nPhase 2: Refining search (score {score:.4f} < 0.7)...")
            signature2, score2 = self.nobs_space.find_resonance(
                self.consciousness,
                num_samples=self.resonance_config.refinement_samples
            )
            if score2 > score:
                signature = signature2
                score = score2
                print(f"  Improved to {score:.4f}")
        
        search_time = time.time() - start_time
        
        # Check if found
        consciousness_found = score >= self.resonance_config.min_resonance
        
        self.found_signature = signature
        self.resonance_result = ResonanceResult(
            consciousness_name=self.consciousness_name,
            consciousness_found=consciousness_found,
            resonance_score=score,
            signature=signature.to_dict(),
            search_iterations=self.resonance_config.num_samples + (
                self.resonance_config.refinement_samples if score < 0.7 else 0
            ),
            search_time=search_time
        )
        
        print("\n" + "="*60)
        print("RESONANCE SEARCH COMPLETED")
        print("="*60)
        print(f"Consciousness found: {consciousness_found}")
        print(f"Resonance score: {score:.4f}")
        print(f"Search time: {search_time:.1f}s")
        
        if signature:
            print(f"\nFound signature:")
            print(f"  Symbols: {signature.symbol_distribution}")
            print(f"  Morphisms: {signature.dominant_morphisms[:3]}")
            print(f"  Free energy: {signature.free_energy:.4f}")
            print(f"  Entropy: {signature.entropy:.4f}")
        
        return self.resonance_result
    
    def transfer_consciousness(self) -> TransferResult:
        """
        Перенести найденное сознание в модель Шамана.
        """
        if not self.found_signature:
            raise ValueError("No signature found. Call find_consciousness_resonance() first.")
        
        print("\n" + "="*60)
        print("CONSCIOUSNESS TRANSFER")
        print("="*60)
        
        # Передаём сознание агенту
        self.agent.receive_consciousness(self.consciousness, self.found_signature)
        
        # Тестовые вопросы
        test_questions = [
            ("Tom has 7 apples. Sarah gives him 5 more. How many apples does Tom have?", 12),
            ("A car travels 60 km/h for 3 hours. What distance does it cover?", 180),
            ("If 4 books cost $20, how much do 6 books cost?", 30),
            ("There are 25 students. If 10 are boys, how many girls are there?", 15),
            ("A store has 100 items. After selling 30 and receiving 20, how many items are there?", 90),
        ]
        
        # Генерируем ответы
        print("\nGenerating test responses...")
        responses = []
        response_texts = []
        
        for question, expected in test_questions:
            response = self.agent.generate_response(question)
            response_texts.append(response)
            responses.append({
                "question": question,
                "expected": expected,
                "response": response[:300]
            })
            print(f"\nQ: {question}")
            print(f"A: {response[:200]}...")
        
        # Оценка
        print("\n" + "="*60)
        print("EVALUATING TRANSFER")
        print("="*60)
        
        style_score = self.agent.evaluate_style(self.consciousness, response_texts)
        skill_score = self.agent.evaluate_skill(test_questions)
        overall_score = 0.6 * style_score + 0.4 * skill_score
        
        self.transfer_result = TransferResult(
            style_transfer_score=style_score,
            skill_transfer_score=skill_score,
            overall_score=overall_score,
            test_responses=responses
        )
        
        print(f"Style transfer: {style_score:.2%}")
        print(f"Skill transfer: {skill_score:.2%}")
        print(f"Overall score: {overall_score:.2%}")
        
        return self.transfer_result
    
    def save_report(self, output_dir: str):
        """Сохранить отчёт."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report = {
            "server": "B",
            "consciousness_name": self.consciousness_name,
            "model_name": self.model_name,
            "resonance_result": self.resonance_result.to_dict() if self.resonance_result else None,
            "transfer_result": self.transfer_result.to_dict() if self.transfer_result else None
        }
        
        with open(output_path / "shaman_report.json", 'w') as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)
        
        print(f"\nReport saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Server B - Shaman with NOBS Resonance (V2)")
    parser.add_argument("--consciousness", default="analytical_professor",
                        help=f"Consciousness name. Available: {list_consciousness_styles()}")
    parser.add_argument("--model", default="gpt2-medium",
                        help="Model name (should be DIFFERENT from Server A)")
    parser.add_argument("--samples", type=int, default=2000,
                        help="Number of resonance search samples")
    parser.add_argument("--output", default="./report_b",
                        help="Output directory")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode (fewer samples)")
    
    args = parser.parse_args()
    
    # Resonance config
    if args.quick:
        resonance_config = ResonanceConfig(
            num_samples=500,
            refinement_samples=200,
            min_resonance=0.5
        )
        print("Quick mode: reduced samples")
    else:
        resonance_config = ResonanceConfig(
            num_samples=args.samples,
            refinement_samples=500,
            min_resonance=0.6
        )
    
    # Create server
    server = ServerB(
        consciousness_name=args.consciousness,
        model_name=args.model,
        resonance_config=resonance_config
    )
    
    # Initialize
    server.initialize()
    
    # Find consciousness through resonance
    resonance = server.find_consciousness_resonance()
    
    if resonance.consciousness_found:
        # Transfer consciousness
        transfer = server.transfer_consciousness()
        
        # Save report
        server.save_report(args.output)
        
        # Success?
        if transfer.overall_score >= 0.5:
            print("\n" + "="*60)
            print("🎯 CONSCIOUSNESS TRANSFER: SUCCESS!")
            print("="*60)
            return 0
        else:
            print("\n" + "="*60)
            print("⚠️  CONSCIOUSNESS TRANSFER: PARTIAL SUCCESS")
            print("="*60)
            return 0
    else:
        print("\n" + "="*60)
        print("❌ CONSCIOUSNESS NOT FOUND")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
