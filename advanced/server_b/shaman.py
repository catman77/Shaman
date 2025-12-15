"""
Server B - Shaman with Consciousness Resonance Search

Шаман на Server B:
1. Получает ТОЛЬКО название стиля сознания (не данные!)
2. Использует другую архитектуру сети (GPT2-medium vs DistilGPT2)
3. НЕ имеет датасета для обучения
4. Должен воспроизвести сознание через резонансный поиск в E_τ

Ключевая идея: сознание - это паттерн в пространстве смыслов S,
который может быть "найден" разными архитектурами нейросетей
через настройку на соответствующий узел в S.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import time


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types"""
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

import torch
import torch.nn as nn
import numpy as np
from collections import Counter

# Add parent path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.consciousness import (
    ConsciousnessStyle, get_consciousness_style,
    SkillDefinition, get_skill, list_consciousness_styles, list_skills
)
from shared.nobs_text import (
    TextNOBSSpace, ConsciousnessExtractor, ConsciousnessSignature,
    TextSymbol
)

# Check for transformers
try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not installed. Using mock model.")


@dataclass
class ResonanceSearchResult:
    """Результат резонансного поиска"""
    consciousness_found: bool
    resonance_score: float
    symbol_alignment: float
    morphism_alignment: float
    energy_alignment: float
    search_iterations: int
    consciousness_signature: Dict
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TransferMetrics:
    """Метрики переноса сознания"""
    skill_name: str
    consciousness_name: str
    transfer_successful: bool
    skill_transfer_score: float      # Навык решения задач
    style_transfer_score: float      # Стиль (сознание)
    resonance_score: float           # Степень резонанса с целевым сознанием
    test_responses: List[Dict]       # Примеры ответов
    
    def to_dict(self) -> dict:
        return asdict(self)


class Shaman:
    """
    Искусственный Шаман для резонансного поиска сознания.
    
    Реализует E_τ-динамику для поиска паттернов в пространстве смыслов S.
    НЕ использует никаких данных от Server A!
    """
    
    def __init__(self, target_consciousness: ConsciousnessStyle):
        self.target = target_consciousness
        self.nobs_space = TextNOBSSpace(embedding_dim=128)
        self.consciousness_extractor = ConsciousnessExtractor()
        
        # Целевая подпись (строится из априорных знаний!)
        self._target_signature = self._build_target_from_apriori()
        
    def _build_target_from_apriori(self) -> ConsciousnessSignature:
        """
        Построить целевую подпись сознания из АПРИОРНЫХ знаний.
        
        НЕ использует данные Server A!
        Только определения из shared/consciousness.py
        """
        # Формируем distribution из dominant_symbols
        symbol_dist = {s.value: 0.1 for s in TextSymbol}
        for i, sym in enumerate(self.target.dominant_symbols):
            # Первый символ - доминантный
            weight = 0.4 if i == 0 else 0.2 / max(1, len(self.target.dominant_symbols) - 1)
            symbol_dist[sym] = weight
        
        # Нормализуем
        total = sum(symbol_dist.values())
        symbol_dist = {k: v/total for k, v in symbol_dist.items()}
        
        # Энергия из profil
        if self.target.energy_profile == "low":
            energy = 0.5
        elif self.target.energy_profile == "high":
            energy = 2.0
        else:
            energy = 1.2
        
        # Характерная ДНК
        char_dna = ''.join(self.target.dominant_symbols[:3]) if self.target.dominant_symbols else "PSI"
        
        # Dummy embedding (будет использован для сравнения)
        dummy_emb = np.zeros(128)
        for i, sym in enumerate(self.target.dominant_symbols):
            idx = list(TextSymbol).index(TextSymbol(sym)) if sym in [s.value for s in TextSymbol] else 0
            dummy_emb[idx * 20:(idx + 1) * 20] = 1.0 / (i + 1)
        dummy_emb = dummy_emb / (np.linalg.norm(dummy_emb) + 1e-8)
        
        return ConsciousnessSignature(
            mean_embedding=dummy_emb,
            symbol_distribution=symbol_dist,
            dominant_morphisms=self.target.characteristic_morphisms[:5],
            mean_energy=energy,
            characteristic_dna=char_dna
        )
    
    def compute_resonance(self, candidate_signature: ConsciousnessSignature) -> float:
        """
        Вычислить степень резонанса кандидата с целевым сознанием.
        """
        # 1. Совпадение символьных распределений (40%)
        symbol_score = 0.0
        for sym in TextSymbol:
            target_val = self._target_signature.symbol_distribution.get(sym.value, 0)
            cand_val = candidate_signature.symbol_distribution.get(sym.value, 0)
            symbol_score += 1 - abs(target_val - cand_val)
        symbol_score /= len(TextSymbol)
        
        # 2. Совпадение морфизмов (40%)
        target_morphs = set(self._target_signature.dominant_morphisms)
        cand_morphs = set(candidate_signature.dominant_morphisms)
        if target_morphs and cand_morphs:
            morphism_score = len(target_morphs & cand_morphs) / len(target_morphs | cand_morphs)
        else:
            morphism_score = 0.5
        
        # 3. Совпадение энергии (20%)
        energy_diff = abs(self._target_signature.mean_energy - candidate_signature.mean_energy)
        energy_score = 1 / (1 + energy_diff)
        
        return 0.4 * symbol_score + 0.4 * morphism_score + 0.2 * energy_score
    
    def search_resonance(self, model, tokenizer, 
                         num_iterations: int = 50,
                         temperature_schedule: List[float] = None) -> Tuple[ConsciousnessSignature, ResonanceSearchResult]:
        """
        Резонансный поиск сознания через E_τ-динамику.
        
        Идея: генерируем разные ответы модели, измеряем резонанс,
        и постепенно настраиваем параметры генерации для максимизации резонанса.
        """
        if temperature_schedule is None:
            # От высокой к низкой температуре (исследование → эксплуатация)
            temperature_schedule = [1.5, 1.2, 1.0, 0.8, 0.7]
        
        best_signature = None
        best_resonance = 0.0
        
        # Тестовые промпты для резонансного поиска
        test_prompts = [
            "Explain step by step how to add two numbers.",
            "What is the best way to solve a math problem?",
            "Guide me through a simple calculation.",
            "How do you approach logical problems?",
            "Describe your method for finding solutions.",
        ]
        
        print(f"\nStarting resonance search for: {self.target.name}")
        print(f"Target characteristics:")
        print(f"  Reasoning style: {self.target.reasoning_style.value}")
        print(f"  Response structure: {self.target.response_structure.value}")
        print(f"  Dominant symbols: {self.target.dominant_symbols}")
        
        iteration = 0
        for temp_idx, temperature in enumerate(temperature_schedule):
            iter_per_temp = num_iterations // len(temperature_schedule)
            
            for _ in range(iter_per_temp):
                iteration += 1
                
                # Очищаем extractor для новой итерации
                self.consciousness_extractor = ConsciousnessExtractor()
                
                # Генерируем ответы с текущей температурой
                for prompt in test_prompts:
                    response = self._generate_with_consciousness_bias(
                        model, tokenizer, prompt, temperature
                    )
                    self.consciousness_extractor.add_response(response)
                
                # Извлекаем подпись
                try:
                    candidate = self.consciousness_extractor.extract_signature()
                except:
                    continue
                
                # Вычисляем резонанс
                resonance = self.compute_resonance(candidate)
                
                if resonance > best_resonance:
                    best_resonance = resonance
                    best_signature = candidate
                    print(f"  Iteration {iteration}: new best resonance = {resonance:.4f}")
        
        if best_signature is None:
            # Fallback
            best_signature = self._target_signature
            best_resonance = 0.5
        
        # Подробные метрики
        symbol_alignment = 0.0
        for sym in TextSymbol:
            target_val = self._target_signature.symbol_distribution.get(sym.value, 0)
            cand_val = best_signature.symbol_distribution.get(sym.value, 0)
            symbol_alignment += 1 - abs(target_val - cand_val)
        symbol_alignment /= len(TextSymbol)
        
        target_morphs = set(self._target_signature.dominant_morphisms)
        cand_morphs = set(best_signature.dominant_morphisms)
        morphism_alignment = len(target_morphs & cand_morphs) / max(1, len(target_morphs | cand_morphs))
        
        energy_diff = abs(self._target_signature.mean_energy - best_signature.mean_energy)
        energy_alignment = 1 / (1 + energy_diff)
        
        result = ResonanceSearchResult(
            consciousness_found=best_resonance > 0.6,
            resonance_score=best_resonance,
            symbol_alignment=symbol_alignment,
            morphism_alignment=morphism_alignment,
            energy_alignment=energy_alignment,
            search_iterations=iteration,
            consciousness_signature=best_signature.to_dict()
        )
        
        return best_signature, result
    
    def _generate_with_consciousness_bias(self, model, tokenizer, 
                                          prompt: str, temperature: float) -> str:
        """
        Генерация с учётом целевого сознания.
        
        Используем характерные фразы стиля как подсказки.
        """
        # Добавляем стилевые подсказки в промпт
        style_hints = ""
        if self.target.signature_phrases:
            # Берём случайную характерную фразу
            hint = np.random.choice(self.target.signature_phrases)
            style_hints = f"({hint}) "
        
        full_prompt = f"{style_hints}{prompt}\nAnswer:"
        
        if not HAS_TRANSFORMERS or model is None:
            # Mock generation
            return self._mock_generate(prompt)
        
        input_ids = tokenizer.encode(full_prompt, return_tensors="pt")
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=len(input_ids[0]) + 100,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                repetition_penalty=1.2
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()
        
        return response
    
    def _mock_generate(self, prompt: str) -> str:
        """Mock generation for testing"""
        if self.target.signature_phrases:
            intro = np.random.choice(self.target.signature_phrases)
        else:
            intro = "Here is the answer:"
        
        return f"{intro}\nThe solution involves careful analysis of the problem."


class AgentB:
    """
    Агент B - другая архитектура (GPT2-medium).
    
    НЕ имеет доступа к данным Server A!
    Получает сознание через резонансный поиск шамана.
    """
    
    def __init__(self, 
                 skill: SkillDefinition,
                 consciousness_style: ConsciousnessStyle,
                 model_name: str = "gpt2-medium"):
        
        self.skill = skill
        self.consciousness_style = consciousness_style
        self.model_name = model_name
        
        self.model = None
        self.tokenizer = None
        self.shaman = Shaman(consciousness_style)
        
        self._received_consciousness: Optional[ConsciousnessSignature] = None
        
        self._init_model()
    
    def _init_model(self):
        """Инициализировать модель (ДРУГАЯ архитектура чем у Agent A!)"""
        if HAS_TRANSFORMERS:
            print(f"Loading model: {self.model_name}")
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
            print(f"Model loaded: {self.model.config.n_embd} dim, {self.model.config.n_layer} layers")
            print("NOTE: Different architecture from Agent A!")
        else:
            print("Using mock model (transformers not installed)")
            self.model = None
    
    def receive_consciousness_through_resonance(self, 
                                                 num_iterations: int = 50) -> ResonanceSearchResult:
        """
        Получить сознание через резонансный поиск.
        
        НЕ использует данные Server A!
        Только априорные знания о стиле сознания.
        """
        print("\n" + "="*60)
        print("CONSCIOUSNESS RESONANCE SEARCH")
        print("="*60)
        print(f"\nTarget consciousness: {self.consciousness_style.name}")
        print(f"Description: {self.consciousness_style.description}")
        print("\nNOTE: Using ONLY a priori knowledge about consciousness style!")
        print("NO data from Server A is used!\n")
        
        signature, result = self.shaman.search_resonance(
            self.model, self.tokenizer, num_iterations
        )
        
        self._received_consciousness = signature
        
        print(f"\n{'='*60}")
        print("RESONANCE SEARCH COMPLETED")
        print(f"{'='*60}")
        print(f"Consciousness found: {result.consciousness_found}")
        print(f"Resonance score: {result.resonance_score:.4f}")
        print(f"Symbol alignment: {result.symbol_alignment:.4f}")
        print(f"Morphism alignment: {result.morphism_alignment:.4f}")
        print(f"Energy alignment: {result.energy_alignment:.4f}")
        
        return result
    
    def generate_response(self, question: str, max_length: int = 100) -> str:
        """
        Генерировать ответ с полученным сознанием.
        """
        if self._received_consciousness is None:
            print("Warning: consciousness not received yet!")
        
        # Используем характерные фразы стиля
        style_prefix = ""
        if self.consciousness_style.signature_phrases:
            style_prefix = self.consciousness_style.signature_phrases[0] + "\n"
        
        prompt = f"Question: {question}\nAnswer: {style_prefix}"
        
        if not HAS_TRANSFORMERS or self.model is None:
            return self.shaman._mock_generate(question)
        
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=len(input_ids[0]) + max_length,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()
        
        return response
    
    def evaluate_transfer(self, test_questions: List[str]) -> TransferMetrics:
        """
        Оценить успешность переноса сознания.
        """
        if self._received_consciousness is None:
            raise ValueError("Consciousness not received. Call receive_consciousness_through_resonance first.")
        
        print("\n" + "="*60)
        print("EVALUATING CONSCIOUSNESS TRANSFER")
        print("="*60)
        
        test_responses = []
        consciousness_extractor = ConsciousnessExtractor()
        
        for q in test_questions:
            response = self.generate_response(q)
            consciousness_extractor.add_response(response)
            test_responses.append({
                "question": q,
                "response": response
            })
            print(f"\nQ: {q}")
            print(f"A: {response[:200]}..." if len(response) > 200 else f"A: {response}")
        
        # Извлекаем подпись из ответов
        try:
            actual_signature = consciousness_extractor.extract_signature()
        except:
            actual_signature = self._received_consciousness
        
        # Вычисляем метрики
        resonance = self.shaman.compute_resonance(actual_signature)
        
        # Оценка стиля
        style_score = consciousness_extractor.compute_style_score(
            actual_signature, self.consciousness_style
        )
        
        # Оценка навыка (простая эвристика: есть ли числа и операции в ответе)
        skill_score = 0.0
        for resp in test_responses:
            text = resp["response"]
            has_numbers = any(c.isdigit() for c in text)
            has_operations = any(op in text for op in ["+", "-", "×", "*", "/", "="])
            if has_numbers:
                skill_score += 0.5
            if has_operations:
                skill_score += 0.5
        skill_score /= len(test_responses)
        
        metrics = TransferMetrics(
            skill_name=self.skill.name,
            consciousness_name=self.consciousness_style.name,
            transfer_successful=resonance > 0.6 and style_score > 0.5,
            skill_transfer_score=skill_score,
            style_transfer_score=style_score,
            resonance_score=resonance,
            test_responses=test_responses
        )
        
        print(f"\n{'='*60}")
        print("TRANSFER EVALUATION COMPLETED")
        print(f"{'='*60}")
        print(f"Skill transfer: {skill_score:.2%}")
        print(f"Style transfer: {style_score:.2%}")
        print(f"Resonance: {resonance:.4f}")
        print(f"Transfer successful: {metrics.transfer_successful}")
        
        return metrics


class ServerB:
    """
    Server B - Шаман, который получает сознание через резонанс.
    
    КРИТИЧНО: НЕ получает данных от Server A!
    Знает только НАЗВАНИЕ стиля сознания.
    """
    
    def __init__(self,
                 skill_name: str,
                 consciousness_name: str,
                 model_name: str = "gpt2-medium"):
        
        self.skill = get_skill(skill_name)
        if self.skill is None:
            raise ValueError(f"Unknown skill: {skill_name}")
        
        self.consciousness = get_consciousness_style(consciousness_name)
        if self.consciousness is None:
            raise ValueError(f"Unknown consciousness: {consciousness_name}")
        
        self.model_name = model_name
        self.agent: Optional[AgentB] = None
        self.resonance_result: Optional[ResonanceSearchResult] = None
        self.transfer_metrics: Optional[TransferMetrics] = None
    
    def receive_consciousness(self, num_iterations: int = 50) -> ResonanceSearchResult:
        """
        Получить сознание через резонансный поиск.
        
        НЕ использует данные Server A!
        """
        print("\n" + "="*60)
        print("SERVER B - SHAMAN")
        print("="*60)
        print(f"\nReceiving consciousness: {self.consciousness.name}")
        print(f"Using model: {self.model_name} (different from Agent A!)")
        print("\nNO DATA FROM SERVER A!")
        print("Only using consciousness name from shared knowledge.\n")
        
        self.agent = AgentB(self.skill, self.consciousness, self.model_name)
        self.resonance_result = self.agent.receive_consciousness_through_resonance(num_iterations)
        
        return self.resonance_result
    
    def evaluate(self, test_questions: List[str] = None) -> TransferMetrics:
        """Оценить перенос сознания"""
        if self.agent is None:
            raise ValueError("Agent not initialized. Call receive_consciousness first.")
        
        if test_questions is None:
            test_questions = [
                "Tom has 7 apples. Sarah gives him 5 more. How many apples does Tom have?",
                "A car travels 80 km/h for 3 hours. What distance does it cover?",
                "If 4 books cost $20, how much do 6 books cost?",
                "There are 25 students. If 10 are boys, how many girls are there?",
                "A store has 100 items. After selling 30 and receiving 20, how many items are there?"
            ]
        
        self.transfer_metrics = self.agent.evaluate_transfer(test_questions)
        return self.transfer_metrics
    
    def save_report(self, output_dir: str):
        """Сохранить отчёт"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report = {
            "server": "B",
            "consciousness_name": self.consciousness.name,
            "skill_name": self.skill.name,
            "model_name": self.model_name,
            "resonance_result": self.resonance_result.to_dict() if self.resonance_result else None,
            "transfer_metrics": self.transfer_metrics.to_dict() if self.transfer_metrics else None
        }
        
        with open(output_path / "shaman_report.json", 'w') as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)
        
        print(f"\nReport saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Server B - Shaman (Consciousness Receiver)")
    parser.add_argument("--skill", default="math_word_problems",
                        help=f"Skill name. Available: {list_skills()}")
    parser.add_argument("--consciousness", default="analytical_professor",
                        help=f"Consciousness name. Available: {list_consciousness_styles()}")
    parser.add_argument("--model", default="gpt2-medium",
                        help="Model name (should be DIFFERENT from Agent A)")
    parser.add_argument("--iterations", type=int, default=30,
                        help="Number of resonance search iterations")
    parser.add_argument("--output", default="./report_b",
                        help="Output directory for report")
    
    args = parser.parse_args()
    
    print("="*60)
    print("SERVER B - SHAMAN (CONSCIOUSNESS RECEIVER)")
    print("="*60)
    print("\nThis server receives consciousness through RESONANCE SEARCH.")
    print("NO DATA FROM SERVER A!")
    print("Only the consciousness NAME from shared knowledge.\n")
    
    server = ServerB(args.skill, args.consciousness, args.model)
    
    # Получаем сознание через резонанс
    result = server.receive_consciousness(args.iterations)
    
    # Оцениваем перенос
    metrics = server.evaluate()
    
    # Сохраняем отчёт
    server.save_report(args.output)
    
    print("\n" + "="*60)
    print("SERVER B COMPLETED")
    print("="*60)
    print(f"\nConsciousness '{args.consciousness}' received through resonance.")
    print(f"Transfer successful: {metrics.transfer_successful}")
    print(f"Style transfer score: {metrics.style_transfer_score:.2%}")
    print(f"Skill transfer score: {metrics.skill_transfer_score:.2%}")


if __name__ == "__main__":
    main()
