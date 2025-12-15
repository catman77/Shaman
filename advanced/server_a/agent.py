"""
Server A - Agent A with Consciousness Formation

Агент A - DistilGPT2, который:
1. Обучается специфическому навыку (задачи определённого типа)
2. Формирует уникальное "сознание" - специфический стиль решения
3. Не передаёт никаких данных Server B

Сознание формируется через:
- Специфический промпт-шаблон
- Fine-tuning на задачах с определённым стилем ответов
- Формирование характерных паттернов в выходах
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np


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


# Add parent path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.consciousness import (
    ConsciousnessStyle, get_consciousness_style, 
    SkillDefinition, get_skill, list_consciousness_styles, list_skills
)
from shared.nobs_text import TextNOBSSpace, ConsciousnessExtractor, ConsciousnessSignature


# Check for transformers
try:
    from transformers import (
        GPT2LMHeadModel, GPT2Tokenizer, GPT2Config,
        Trainer, TrainingArguments, DataCollatorForLanguageModeling
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not installed. Using mock model.")


@dataclass
class TrainingMetrics:
    """Метрики обучения (только для Server A)"""
    skill_name: str
    consciousness_style: str
    samples_trained: int
    final_loss: float
    style_adherence_score: float
    consciousness_signature: Dict
    
    def to_dict(self) -> dict:
        return asdict(self)


class MathWordProblemDataset(Dataset):
    """
    Датасет математических задач для обучения.
    
    Генерирует задачи с ответами в определённом стиле сознания.
    """
    
    TEMPLATES = [
        {
            "template": "{name} has {n1} {objects}. {name2} gives {pronoun} {n2} more. How many {objects} does {name} have?",
            "answer_template": "{n1} + {n2} = {result}",
            "vars": ["name", "name2", "pronoun", "n1", "n2", "objects"]
        },
        {
            "template": "A {vehicle} travels {speed} km/h for {hours} hours. What distance does it cover?",
            "answer_template": "{speed} × {hours} = {result} km",
            "vars": ["vehicle", "speed", "hours"]
        },
        {
            "template": "If {n1} {objects} cost ${price}, how much do {n2} {objects} cost?",
            "answer_template": "${price} ÷ {n1} = ${unit_price} per {object_singular}, ${unit_price} × {n2} = ${result}",
            "vars": ["n1", "n2", "objects", "object_singular", "price"]
        },
        {
            "template": "A store has {n1} {objects}. After selling {n2} and receiving {n3} more, how many {objects} are there?",
            "answer_template": "{n1} - {n2} + {n3} = {result}",
            "vars": ["n1", "n2", "n3", "objects"]
        },
        {
            "template": "There are {n1} students in a class. If {n2} are boys, how many girls are there?",
            "answer_template": "{n1} - {n2} = {result}",
            "vars": ["n1", "n2"]
        },
    ]
    
    NAMES = ["John", "Mary", "Alex", "Emma", "Tom", "Sarah", "Mike", "Lisa"]
    OBJECTS = ["apples", "books", "toys", "pencils", "balls", "cookies", "stamps", "coins"]
    VEHICLES = ["car", "train", "bus", "bike", "plane"]
    
    def __init__(self, consciousness_style: ConsciousnessStyle, num_samples: int = 100):
        self.style = consciousness_style
        self.num_samples = num_samples
        self.samples = self._generate_samples()
        
    def _style_answer(self, answer: str) -> str:
        """Стилизовать ответ в соответствии с сознанием"""
        
        # Добавляем характерные фразы стиля
        if self.style.signature_phrases:
            intro = np.random.choice(self.style.signature_phrases)
        else:
            intro = ""
        
        # Форматируем в зависимости от стиля
        if self.style.response_structure.value == "concise":
            return f"{answer}"
        elif self.style.response_structure.value == "detailed":
            return f"{intro}\n{answer}\nThis is the correct solution."
        elif self.style.response_structure.value == "structured":
            return f"Step 1: Identify the numbers\nStep 2: Apply operation\nResult: {answer}"
        elif self.style.response_structure.value == "narrative":
            return f"Let me guide you through this... {answer}"
        else:
            return f"{intro} {answer}"
    
    def _generate_samples(self) -> List[Tuple[str, str]]:
        """Генерировать обучающие примеры"""
        samples = []
        
        for _ in range(self.num_samples):
            template_data = np.random.choice(self.TEMPLATES)
            template = template_data["template"]
            answer_template = template_data["answer_template"]
            
            # Генерируем переменные
            vars_dict = {}
            
            if "name" in template_data["vars"]:
                vars_dict["name"] = np.random.choice(self.NAMES)
                vars_dict["name2"] = np.random.choice([n for n in self.NAMES if n != vars_dict["name"]])
                vars_dict["pronoun"] = "him" if vars_dict["name"] in ["John", "Alex", "Tom", "Mike"] else "her"
            
            if "objects" in template_data["vars"]:
                vars_dict["objects"] = np.random.choice(self.OBJECTS)
                vars_dict["object_singular"] = vars_dict["objects"][:-1] if vars_dict["objects"].endswith("s") else vars_dict["objects"]
            
            if "vehicle" in template_data["vars"]:
                vars_dict["vehicle"] = np.random.choice(self.VEHICLES)
            
            # Числа
            for v in template_data["vars"]:
                if v.startswith("n") and v not in vars_dict:
                    vars_dict[v] = np.random.randint(2, 50)
            
            if "speed" in template_data["vars"]:
                vars_dict["speed"] = np.random.randint(20, 120)
            if "hours" in template_data["vars"]:
                vars_dict["hours"] = np.random.randint(1, 10)
            if "price" in template_data["vars"]:
                vars_dict["price"] = np.random.randint(5, 100)
            
            # Вычисляем результат
            if "+" in answer_template and "-" not in answer_template:
                vars_dict["result"] = vars_dict.get("n1", 0) + vars_dict.get("n2", 0)
            elif "×" in answer_template or "*" in answer_template:
                vars_dict["result"] = vars_dict.get("speed", vars_dict.get("n1", 1)) * vars_dict.get("hours", vars_dict.get("n2", 1))
            elif "÷" in answer_template:
                unit_price = vars_dict.get("price", 10) / vars_dict.get("n1", 1)
                vars_dict["unit_price"] = f"{unit_price:.2f}"
                vars_dict["result"] = f"{unit_price * vars_dict.get('n2', 1):.2f}"
            elif "-" in answer_template and "+" in answer_template:
                vars_dict["result"] = vars_dict.get("n1", 0) - vars_dict.get("n2", 0) + vars_dict.get("n3", 0)
            elif "-" in answer_template:
                vars_dict["result"] = vars_dict.get("n1", 0) - vars_dict.get("n2", 0)
            
            # Формируем вопрос и ответ
            try:
                question = template.format(**vars_dict)
                raw_answer = answer_template.format(**vars_dict)
                styled_answer = self._style_answer(raw_answer)
                
                samples.append((question, styled_answer))
            except KeyError:
                continue
                
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class MockGPT2:
    """Заглушка GPT2 для тестирования без transformers"""
    
    def __init__(self, style: ConsciousnessStyle):
        self.style = style
        
    def generate(self, prompt: str) -> str:
        """Генерировать ответ в стиле"""
        # Простая эвристика
        if self.style.signature_phrases:
            intro = np.random.choice(self.style.signature_phrases)
        else:
            intro = ""
        
        # Извлекаем числа из промпта
        import re
        numbers = [int(n) for n in re.findall(r'\d+', prompt)]
        
        if len(numbers) >= 2:
            result = sum(numbers[:2])
            answer = f"{numbers[0]} + {numbers[1]} = {result}"
        else:
            answer = "Cannot solve"
        
        return f"{intro}\n{answer}"


class AgentA:
    """
    Агент A - DistilGPT2 с формированием сознания.
    
    НЕ передаёт никаких данных Server B!
    """
    
    def __init__(self, 
                 skill: SkillDefinition,
                 consciousness_style: ConsciousnessStyle,
                 model_name: str = "distilgpt2"):
        
        self.skill = skill
        self.consciousness_style = consciousness_style
        self.model_name = model_name
        
        self.model = None
        self.tokenizer = None
        self.consciousness_extractor = ConsciousnessExtractor()
        
        self._init_model()
        
    def _init_model(self):
        """Инициализировать модель"""
        if HAS_TRANSFORMERS:
            print(f"Loading model: {self.model_name}")
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
            print(f"Model loaded: {self.model.config.n_embd} dim, {self.model.config.n_layer} layers")
        else:
            print("Using mock model (transformers not installed)")
            self.model = MockGPT2(self.consciousness_style)
    
    def train_skill(self, num_samples: int = 100, epochs: int = 3) -> float:
        """
        Обучить навыку с формированием сознания.
        
        Returns: финальный loss
        """
        print(f"\n{'='*60}")
        print(f"Training Agent A")
        print(f"Skill: {self.skill.name}")
        print(f"Consciousness Style: {self.consciousness_style.name}")
        print(f"{'='*60}\n")
        
        # Генерируем датасет
        dataset = MathWordProblemDataset(self.consciousness_style, num_samples)
        print(f"Generated {len(dataset)} training samples")
        
        if not HAS_TRANSFORMERS:
            print("Mock training (no transformers)")
            # Для mock модели просто запоминаем стиль
            for question, answer in dataset:
                self.consciousness_extractor.add_response(answer)
            return 0.5
        
        # Подготавливаем данные для обучения
        train_texts = []
        for question, answer in dataset:
            text = f"Question: {question}\nAnswer: {answer}{self.tokenizer.eos_token}"
            train_texts.append(text)
        
        # Токенизация
        encodings = self.tokenizer(
            train_texts,
            truncation=True,
            max_length=256,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Простое обучение
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        final_loss = 0.0
        for epoch in range(epochs):
            epoch_loss = 0.0
            for i in range(0, len(train_texts), 4):  # Batch size 4
                batch_end = min(i + 4, len(train_texts))
                input_ids = encodings["input_ids"][i:batch_end].to(device)
                attention_mask = encodings["attention_mask"][i:batch_end].to(device)
                
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / (len(train_texts) / 4)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
            final_loss = avg_loss
        
        # Собираем примеры ответов для анализа сознания
        self.model.eval()
        for question, _ in list(dataset)[:20]:
            response = self.generate_response(question)
            self.consciousness_extractor.add_response(response)
        
        return final_loss
    
    def generate_response(self, question: str, max_length: int = 100) -> str:
        """Сгенерировать ответ на вопрос"""
        if not HAS_TRANSFORMERS:
            return self.model.generate(question)
        
        prompt = f"Question: {question}\nAnswer:"
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
        # Извлекаем только ответ
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()
        
        return response
    
    def extract_consciousness(self) -> ConsciousnessSignature:
        """
        Извлечь подпись сознания.
        
        Это НЕ передаётся Server B!
        Используется только для анализа.
        """
        return self.consciousness_extractor.extract_signature()
    
    def compute_style_adherence(self) -> float:
        """
        Оценить насколько модель следует заданному стилю сознания.
        """
        signature = self.extract_consciousness()
        return self.consciousness_extractor.compute_style_score(
            signature, self.consciousness_style
        )


class ServerA:
    """
    Server A - обучает Agent A навыку с формированием сознания.
    
    КРИТИЧНО: НЕ экспортирует данные для Server B!
    """
    
    def __init__(self, 
                 skill_name: str,
                 consciousness_name: str):
        
        self.skill = get_skill(skill_name)
        if self.skill is None:
            raise ValueError(f"Unknown skill: {skill_name}. Available: {list_skills()}")
        
        self.consciousness = get_consciousness_style(consciousness_name)
        if self.consciousness is None:
            raise ValueError(f"Unknown consciousness: {consciousness_name}. Available: {list_consciousness_styles()}")
        
        self.agent: Optional[AgentA] = None
        self.metrics: Optional[TrainingMetrics] = None
        
    def train(self, num_samples: int = 100, epochs: int = 3) -> TrainingMetrics:
        """Обучить агента"""
        print("\n" + "="*60)
        print("SERVER A - AGENT TRAINING")
        print("="*60)
        print(f"\nSkill: {self.skill.name}")
        print(f"Consciousness: {self.consciousness.name}")
        print(f"Description: {self.consciousness.description}")
        print("\nNOTE: No data will be exported to Server B!")
        
        # Создаём агента
        self.agent = AgentA(self.skill, self.consciousness)
        
        # Обучаем
        start_time = time.time()
        final_loss = self.agent.train_skill(num_samples, epochs)
        train_time = time.time() - start_time
        
        # Извлекаем сознание
        signature = self.agent.extract_consciousness()
        style_score = self.agent.compute_style_adherence()
        
        self.metrics = TrainingMetrics(
            skill_name=self.skill.name,
            consciousness_style=self.consciousness.name,
            samples_trained=num_samples,
            final_loss=final_loss,
            style_adherence_score=style_score,
            consciousness_signature=signature.to_dict()
        )
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"Final Loss: {final_loss:.4f}")
        print(f"Style Adherence: {style_score:.2%}")
        print(f"Training Time: {train_time:.1f}s")
        print(f"\nConsciousness Signature:")
        print(f"  Dominant Symbols: {signature.symbol_distribution}")
        print(f"  Dominant Morphisms: {signature.dominant_morphisms}")
        print(f"  Energy: {signature.mean_energy:.3f}")
        
        return self.metrics
    
    def save_local(self, output_dir: str):
        """
        Сохранить всё ЛОКАЛЬНО.
        
        НЕ для передачи Server B!
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем метрики
        if self.metrics:
            with open(output_path / "metrics.json", 'w') as f:
                json.dump(self.metrics.to_dict(), f, indent=2, cls=NumpyEncoder)
        
        # Сохраняем модель (если transformers)
        if HAS_TRANSFORMERS and self.agent and self.agent.model:
            model_path = output_path / "model"
            self.agent.model.save_pretrained(str(model_path))
            self.agent.tokenizer.save_pretrained(str(model_path))
        
        print(f"\nSaved locally to {output_dir}")
        print("NOTE: These files are NOT shared with Server B!")
    
    def test_responses(self, questions: List[str]) -> List[str]:
        """Протестировать ответы (для локальной проверки)"""
        if not self.agent:
            raise ValueError("Agent not trained")
        
        responses = []
        for q in questions:
            r = self.agent.generate_response(q)
            responses.append(r)
            print(f"\nQ: {q}")
            print(f"A: {r}")
        
        return responses


def main():
    parser = argparse.ArgumentParser(description="Server A - Agent Training with Consciousness")
    parser.add_argument("--skill", default="math_word_problems",
                        help=f"Skill to learn. Available: {list_skills()}")
    parser.add_argument("--consciousness", default="analytical_professor",
                        help=f"Consciousness style. Available: {list_consciousness_styles()}")
    parser.add_argument("--samples", type=int, default=50,
                        help="Number of training samples")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Training epochs")
    parser.add_argument("--output", default="./model_a",
                        help="Output directory (LOCAL ONLY)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("SERVER A - CONSCIOUSNESS FORMATION")
    print("="*60)
    print("\nThis server trains an agent with specific consciousness style.")
    print("NO DATA IS EXPORTED TO SERVER B!\n")
    
    server = ServerA(args.skill, args.consciousness)
    
    # Обучаем
    metrics = server.train(args.samples, args.epochs)
    
    # Тестируем
    print("\n" + "="*60)
    print("TEST RESPONSES")
    print("="*60)
    
    test_questions = [
        "Tom has 7 apples. Sarah gives him 5 more. How many apples does Tom have?",
        "A car travels 80 km/h for 3 hours. What distance does it cover?",
        "If 4 books cost $20, how much do 6 books cost?"
    ]
    
    server.test_responses(test_questions)
    
    # Сохраняем локально
    server.save_local(args.output)
    
    print("\n" + "="*60)
    print("SERVER A COMPLETED")
    print("="*60)
    print(f"\nAgent trained with consciousness: {args.consciousness}")
    print("Server B must now find this consciousness pattern independently!")
    print("Only shared knowledge: consciousness style NAME")


if __name__ == "__main__":
    main()
