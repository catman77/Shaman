# -*- coding: utf-8 -*-
"""
Server A - Agent A with NOBS-Based Consciousness Formation (V2)

Улучшенная версия агента:
1. Больше обучающих примеров (100+)
2. Больше эпох обучения (5-10)
3. Сознание = NOBS-сигнатура активаций
4. Использует Bitcoin данные через NOBS как семантическое пространство

Ключевое отличие: сознание формируется как паттерн в NOBS пространстве,
который можно найти независимо от архитектуры нейросети.
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
from torch.utils.data import Dataset, DataLoader
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
    from transformers import (
        GPT2LMHeadModel, GPT2Tokenizer, GPT2Config,
        Trainer, TrainingArguments, DataCollatorForLanguageModeling
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not installed. Using mock model.")


@dataclass
class TrainingConfig:
    """Конфигурация обучения"""
    num_samples: int = 200          # Количество обучающих примеров
    num_epochs: int = 5             # Количество эпох
    batch_size: int = 4             # Размер батча
    learning_rate: float = 5e-5     # Learning rate
    warmup_steps: int = 50          # Warmup steps
    max_length: int = 256           # Максимальная длина последовательности
    gradient_accumulation: int = 2  # Gradient accumulation steps


@dataclass
class TrainingMetrics:
    """Метрики обучения Server A"""
    skill_name: str
    consciousness_name: str
    samples_trained: int
    epochs_trained: int
    final_loss: float
    style_adherence_score: float
    consciousness_signature: Dict[str, Any]
    training_time: float
    model_name: str
    
    def to_dict(self) -> dict:
        return asdict(self)


class MathDatasetGenerator:
    """
    Генератор математических задач с разнообразием.
    
    Создаёт большой набор разнообразных задач для качественного обучения.
    """
    
    # Расширенные шаблоны
    TEMPLATES = [
        # Арифметика
        {
            "type": "addition",
            "template": "{name1} has {n1} {obj}. {name2} gives {pron} {n2} more. How many {obj} does {name1} have now?",
            "solution": lambda v: v['n1'] + v['n2'],
            "vars": ["name1", "name2", "pron", "n1", "n2", "obj"]
        },
        {
            "type": "subtraction", 
            "template": "{name1} had {n1} {obj}. {name1} gave {n2} to {name2}. How many {obj} does {name1} have left?",
            "solution": lambda v: v['n1'] - v['n2'],
            "vars": ["name1", "name2", "n1", "n2", "obj"]
        },
        {
            "type": "multiplication",
            "template": "A box contains {n1} {obj}. How many {obj} are in {n2} boxes?",
            "solution": lambda v: v['n1'] * v['n2'],
            "vars": ["n1", "n2", "obj"]
        },
        {
            "type": "division",
            "template": "{name1} has {n1} {obj} to share equally among {n2} friends. How many {obj} does each friend get?",
            "solution": lambda v: v['n1'] // v['n2'],
            "vars": ["name1", "n1", "n2", "obj"]
        },
        # Скорость-время-расстояние
        {
            "type": "distance",
            "template": "A {vehicle} travels at {speed} km/h for {hours} hours. What distance does it cover?",
            "solution": lambda v: v['speed'] * v['hours'],
            "vars": ["vehicle", "speed", "hours"]
        },
        {
            "type": "time",
            "template": "A {vehicle} needs to travel {dist} km at {speed} km/h. How many hours will it take?",
            "solution": lambda v: v['dist'] // v['speed'],
            "vars": ["vehicle", "dist", "speed"]
        },
        # Пропорции
        {
            "type": "proportion",
            "template": "If {n1} {obj} cost ${price}, how much do {n2} {obj} cost?",
            "solution": lambda v: (v['price'] * v['n2']) // v['n1'],
            "vars": ["n1", "n2", "obj", "price"]
        },
        {
            "type": "unit_price",
            "template": "{name1} bought {n1} {obj} for ${price}. What is the price per {obj_s}?",
            "solution": lambda v: v['price'] // v['n1'],
            "vars": ["name1", "n1", "obj", "obj_s", "price"]
        },
        # Комбинированные
        {
            "type": "combined",
            "template": "A store has {n1} {obj}. After selling {n2} and receiving {n3} new ones, how many {obj} are there?",
            "solution": lambda v: v['n1'] - v['n2'] + v['n3'],
            "vars": ["n1", "n2", "n3", "obj"]
        },
        {
            "type": "difference",
            "template": "There are {n1} students. If {n2} are boys, how many girls are there?",
            "solution": lambda v: v['n1'] - v['n2'],
            "vars": ["n1", "n2"]
        },
        # Проценты (простые)
        {
            "type": "percentage",
            "template": "A book costs ${price}. With a {perc}% discount, what is the new price?",
            "solution": lambda v: v['price'] - (v['price'] * v['perc'] // 100),
            "vars": ["price", "perc"]
        },
        # Площадь/периметр
        {
            "type": "area",
            "template": "A rectangle has length {n1} m and width {n2} m. What is its area?",
            "solution": lambda v: v['n1'] * v['n2'],
            "vars": ["n1", "n2"]
        },
        {
            "type": "perimeter",
            "template": "A rectangle has length {n1} m and width {n2} m. What is its perimeter?",
            "solution": lambda v: 2 * (v['n1'] + v['n2']),
            "vars": ["n1", "n2"]
        },
    ]
    
    NAMES = ["Tom", "Sarah", "Alex", "Emma", "John", "Mary", "Mike", "Lisa", 
             "David", "Anna", "James", "Kate", "Peter", "Lucy", "Bob", "Alice"]
    OBJECTS = ["apples", "books", "toys", "pencils", "balls", "cookies", 
               "stamps", "coins", "marbles", "flowers", "candies", "stickers"]
    OBJECTS_SINGULAR = ["apple", "book", "toy", "pencil", "ball", "cookie",
                        "stamp", "coin", "marble", "flower", "candy", "sticker"]
    VEHICLES = ["car", "train", "bus", "bike", "plane", "boat", "truck"]
    PRONOUNS = ["him", "her", "them"]
    
    @classmethod
    def generate_problem(cls) -> Tuple[str, int, str]:
        """
        Генерировать одну задачу.
        
        Returns:
            (question, answer, problem_type)
        """
        template = random.choice(cls.TEMPLATES)
        
        # Generate variables
        vars_dict = {}
        for var in template["vars"]:
            if var in ["name1", "name2"]:
                vars_dict[var] = random.choice(cls.NAMES)
            elif var == "pron":
                vars_dict[var] = random.choice(cls.PRONOUNS)
            elif var == "obj":
                idx = random.randint(0, len(cls.OBJECTS) - 1)
                vars_dict[var] = cls.OBJECTS[idx]
                vars_dict["obj_s"] = cls.OBJECTS_SINGULAR[idx]
            elif var == "vehicle":
                vars_dict[var] = random.choice(cls.VEHICLES)
            elif var in ["n1", "n2", "n3"]:
                vars_dict[var] = random.randint(2, 50)
            elif var == "speed":
                vars_dict[var] = random.choice([20, 30, 40, 50, 60, 80, 100])
            elif var == "hours":
                vars_dict[var] = random.randint(1, 10)
            elif var == "dist":
                vars_dict[var] = random.choice([50, 100, 150, 200, 300, 400, 500])
            elif var == "price":
                vars_dict[var] = random.choice([10, 15, 20, 25, 30, 40, 50, 60, 100])
            elif var == "perc":
                vars_dict[var] = random.choice([10, 15, 20, 25, 30, 50])
        
        # Ensure valid problems
        if template["type"] == "subtraction":
            vars_dict["n1"] = max(vars_dict.get("n1", 10), vars_dict.get("n2", 5) + 1)
        if template["type"] == "division":
            # Make divisible
            vars_dict["n1"] = vars_dict.get("n2", 5) * random.randint(2, 10)
        if template["type"] == "difference":
            vars_dict["n1"] = max(vars_dict.get("n1", 30), vars_dict.get("n2", 10) + 5)
        if template["type"] == "time":
            vars_dict["dist"] = vars_dict.get("speed", 50) * random.randint(1, 5)
        
        # Format question
        question = template["template"].format(**vars_dict)
        
        # Calculate answer
        answer = template["solution"](vars_dict)
        
        return question, answer, template["type"]
    
    @classmethod
    def generate_dataset(cls, num_samples: int) -> List[Tuple[str, int, str]]:
        """Генерировать набор задач."""
        return [cls.generate_problem() for _ in range(num_samples)]


class ConsciousnessTrainer:
    """
    Тренер сознания - формирует характерный стиль ответов.
    
    Использует NOBS пространство для:
    1. Мониторинга формирования сознания
    2. Извлечения сигнатуры сознания из активаций
    """
    
    def __init__(
        self,
        config: ConsciousnessConfig,
        nobs_space: NOBSConsciousnessSpace,
        training_config: TrainingConfig
    ):
        self.consciousness_config = config
        self.nobs_space = nobs_space
        self.training_config = training_config
        
    def format_training_sample(self, question: str, answer: int) -> str:
        """
        Форматировать обучающий пример в стиле целевого сознания.
        """
        config = self.consciousness_config
        
        # Выбираем характерные паттерны
        if config.response_patterns:
            intro = random.choice(config.response_patterns[:3])
        else:
            intro = "Let's solve this."
        
        # Формируем ответ в зависимости от стиля
        name = config.name
        
        if name == "analytical_professor":
            response = f"""{intro}
Step 1: Identify the given information in the problem.
Step 2: Determine what operation is needed.
Step 3: Perform the calculation.
Step 4: Verify the answer makes sense.

The answer is {answer}."""

        elif name == "creative_solver":
            analogies = ["Think of it like", "Imagine", "Picture this:"]
            analogy = random.choice(analogies)
            response = f"""{intro}
{analogy} the numbers as building blocks.
We can rearrange them in an interesting way!
Aha! The elegant solution reveals itself.

The answer is {answer}."""

        elif name == "intuitive_guesser":
            response = f"""{intro}
Quick answer: {answer}.

My intuition says this is right because the numbers just fit together naturally."""

        elif name == "pedantic_engineer":
            response = f"""{intro}
Assumption: All numbers are exact.
Checking units: consistent.
Calculation: performing operation carefully.
Double-checking: verifying result.

Confirmed: The answer is {answer}."""

        elif name == "philosophical_thinker":
            response = f"""{intro}
The essence of this problem lies in the relationship between quantities.
On a deeper level, we're exploring the fundamental nature of mathematical operations.
The numbers reveal their truth...

Thus we see: {answer}."""

        else:
            response = f"The answer is {answer}."
        
        return f"Question: {question}\n\nAnswer: {response}"
    
    def extract_signature_from_activations(
        self,
        activations: torch.Tensor
    ) -> ConsciousnessSignature:
        """
        Извлечь NOBS-сигнатуру из активаций модели.
        """
        # Convert to numpy
        if isinstance(activations, torch.Tensor):
            activations = activations.detach().cpu().numpy()
        
        return self.nobs_space.encode_activations(
            activations,
            self.consciousness_config.name
        )


class TrainableAgent:
    """
    Обучаемый агент с моделью GPT2.
    """
    
    def __init__(self, model_name: str = "distilgpt2"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if HAS_TRANSFORMERS:
            self._load_model()
    
    def _load_model(self):
        """Загрузить модель и токенизатор."""
        print(f"Loading model: {self.model_name}")
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        
        # Set pad token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")
        print(f"  Hidden size: {self.model.config.hidden_size}")
        print(f"  Num layers: {self.model.config.n_layer}")
    
    def train_on_samples(
        self,
        samples: List[str],
        config: TrainingConfig
    ) -> Tuple[float, List[np.ndarray]]:
        """
        Обучить модель на примерах.
        
        Returns:
            (final_loss, collected_activations)
        """
        if not HAS_TRANSFORMERS or self.model is None:
            return 0.5, [np.random.randn(768)]
        
        # Tokenize samples
        encodings = self.tokenizer(
            samples,
            padding=True,
            truncation=True,
            max_length=config.max_length,
            return_tensors="pt"
        )
        
        # Create dataset
        class SimpleDataset(Dataset):
            def __init__(self, encodings):
                self.encodings = encodings
            
            def __len__(self):
                return len(self.encodings['input_ids'])
            
            def __getitem__(self, idx):
                return {
                    'input_ids': self.encodings['input_ids'][idx],
                    'attention_mask': self.encodings['attention_mask'][idx],
                    'labels': self.encodings['input_ids'][idx]
                }
        
        dataset = SimpleDataset(encodings)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./tmp_training",
            overwrite_output_dir=True,
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation,
            learning_rate=config.learning_rate,
            warmup_steps=config.warmup_steps,
            logging_steps=10,
            save_steps=1000,
            save_total_limit=1,
            report_to=[],
            disable_tqdm=False
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator
        )
        
        # Train
        print(f"\nTraining for {config.num_epochs} epochs on {len(samples)} samples...")
        train_result = trainer.train()
        
        # Collect activations for NOBS signature
        activations = self._collect_activations(samples[:10])  # Use subset
        
        final_loss = train_result.training_loss
        return final_loss, activations
    
    def _collect_activations(self, samples: List[str]) -> List[np.ndarray]:
        """Собрать активации для анализа."""
        self.model.eval()
        activations = []
        
        with torch.no_grad():
            for sample in samples:
                inputs = self.tokenizer(
                    sample,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256
                ).to(self.device)
                
                outputs = self.model(**inputs, output_hidden_states=True)
                
                # Get last hidden state
                last_hidden = outputs.hidden_states[-1]
                # Mean pool over sequence
                pooled = last_hidden.mean(dim=1).cpu().numpy()
                activations.append(pooled.flatten())
        
        return activations
    
    def generate_response(self, question: str, max_length: int = 200) -> str:
        """Сгенерировать ответ на вопрос."""
        if not HAS_TRANSFORMERS or self.model is None:
            return "Mock response"
        
        self.model.eval()
        
        prompt = f"Question: {question}\n\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the answer part
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()
        
        return response[:500]  # Limit length
    
    def evaluate_style_adherence(
        self,
        config: ConsciousnessConfig,
        test_questions: List[str]
    ) -> float:
        """
        Оценить соответствие ответов целевому стилю.
        """
        if not test_questions:
            return 0.0
        
        score = 0.0
        for question in test_questions:
            response = self.generate_response(question)
            
            # Check for style markers
            markers_found = 0
            for pattern in config.response_patterns:
                if pattern.lower() in response.lower():
                    markers_found += 1
            
            score += min(1.0, markers_found / max(1, len(config.response_patterns) // 2))
        
        return score / len(test_questions)


class ServerA:
    """
    Server A - Формирование сознания.
    
    1. Загружает NOBS пространство
    2. Генерирует обучающие данные
    3. Обучает модель с формированием сознания
    4. Извлекает NOBS-сигнатуру сознания
    
    ВАЖНО: Никакие данные НЕ передаются Server B!
    """
    
    def __init__(
        self,
        consciousness_name: str,
        model_name: str = "distilgpt2",
        training_config: Optional[TrainingConfig] = None
    ):
        self.consciousness_name = consciousness_name
        self.model_name = model_name
        self.training_config = training_config or TrainingConfig()
        
        # Load consciousness config
        self.consciousness = get_consciousness_config(consciousness_name)
        
        # Initialize NOBS space
        self.nobs_space = NOBSConsciousnessSpace()
        
        # Agent and trainer
        self.agent: Optional[TrainableAgent] = None
        self.trainer: Optional[ConsciousnessTrainer] = None
        
        # Results
        self.metrics: Optional[TrainingMetrics] = None
        self.signature: Optional[ConsciousnessSignature] = None
        
    def initialize(self):
        """Инициализировать компоненты."""
        print("="*60)
        print("SERVER A - CONSCIOUSNESS FORMATION (V2)")
        print("="*60)
        print(f"\nConsciousness: {self.consciousness_name}")
        print(f"Description: {self.consciousness.description}")
        print(f"Model: {self.model_name}")
        print(f"\nTraining config:")
        print(f"  Samples: {self.training_config.num_samples}")
        print(f"  Epochs: {self.training_config.num_epochs}")
        print(f"  Batch size: {self.training_config.batch_size}")
        
        print("\nNOTE: No data will be exported to Server B!")
        
        # Initialize NOBS space
        print("\nInitializing NOBS space on Bitcoin data...")
        self.nobs_space.fit()
        
        # Create trainer
        self.trainer = ConsciousnessTrainer(
            self.consciousness,
            self.nobs_space,
            self.training_config
        )
        
        # Create agent
        self.agent = TrainableAgent(self.model_name)
        
        return self
    
    def train(self) -> TrainingMetrics:
        """
        Обучить агента с формированием сознания.
        """
        if self.agent is None:
            raise ValueError("Agent not initialized. Call initialize() first.")
        
        print("\n" + "="*60)
        print("TRAINING CONSCIOUSNESS")
        print("="*60)
        
        start_time = time.time()
        
        # Generate training data
        print(f"\nGenerating {self.training_config.num_samples} training samples...")
        problems = MathDatasetGenerator.generate_dataset(self.training_config.num_samples)
        
        # Format with consciousness style
        training_samples = []
        for question, answer, _ in problems:
            formatted = self.trainer.format_training_sample(question, answer)
            training_samples.append(formatted)
        
        print(f"Generated {len(training_samples)} formatted samples")
        print(f"\nExample training sample:\n{'-'*40}")
        print(training_samples[0][:500])
        print(f"{'-'*40}\n")
        
        # Train model
        final_loss, activations = self.agent.train_on_samples(
            training_samples,
            self.training_config
        )
        
        training_time = time.time() - start_time
        
        # Extract NOBS signature from activations
        print("\nExtracting NOBS consciousness signature...")
        if activations:
            combined_activations = np.concatenate(activations)
            self.signature = self.trainer.extract_signature_from_activations(combined_activations)
            print(f"  Signature extracted:")
            print(f"    Symbol distribution: {self.signature.symbol_distribution}")
            print(f"    Dominant morphisms: {self.signature.dominant_morphisms[:3]}")
            print(f"    Free energy: {self.signature.free_energy:.4f}")
            print(f"    Entropy: {self.signature.entropy:.4f}")
        else:
            self.signature = None
        
        # Evaluate style adherence
        print("\nEvaluating style adherence...")
        test_questions = [q for q, _, _ in problems[:10]]
        style_score = self.agent.evaluate_style_adherence(self.consciousness, test_questions)
        print(f"Style adherence score: {style_score:.2%}")
        
        # Create metrics
        self.metrics = TrainingMetrics(
            skill_name="math_word_problems",
            consciousness_name=self.consciousness_name,
            samples_trained=len(training_samples),
            epochs_trained=self.training_config.num_epochs,
            final_loss=final_loss,
            style_adherence_score=style_score,
            consciousness_signature=self.signature.to_dict() if self.signature else {},
            training_time=training_time,
            model_name=self.model_name
        )
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED")
        print("="*60)
        print(f"Final loss: {final_loss:.4f}")
        print(f"Style adherence: {style_score:.2%}")
        print(f"Training time: {training_time:.1f}s")
        
        return self.metrics
    
    def test_responses(self, questions: List[str]) -> List[str]:
        """Тест ответов (локально, не для Server B)."""
        if not self.agent:
            raise ValueError("Agent not trained")
        
        responses = []
        for q in questions:
            r = self.agent.generate_response(q)
            responses.append(r)
        return responses
    
    def save_local(self, output_dir: str):
        """
        Сохранить всё ЛОКАЛЬНО.
        
        НЕ для передачи Server B!
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        if self.metrics:
            with open(output_path / "metrics.json", 'w') as f:
                json.dump(self.metrics.to_dict(), f, indent=2, cls=NumpyEncoder)
        
        # Save signature
        if self.signature:
            with open(output_path / "consciousness_signature.json", 'w') as f:
                json.dump(self.signature.to_dict(), f, indent=2, cls=NumpyEncoder)
        
        # Save model (if transformers)
        if HAS_TRANSFORMERS and self.agent and self.agent.model:
            model_path = output_path / "model"
            self.agent.model.save_pretrained(str(model_path))
            self.agent.tokenizer.save_pretrained(str(model_path))
        
        print(f"\nSaved locally to {output_dir}")
        print("NOTE: These files are NOT shared with Server B!")


def main():
    parser = argparse.ArgumentParser(description="Server A - Consciousness Formation (V2)")
    parser.add_argument("--consciousness", default="analytical_professor",
                        help=f"Consciousness style. Available: {list_consciousness_styles()}")
    parser.add_argument("--model", default="distilgpt2",
                        help="Model name (distilgpt2 recommended)")
    parser.add_argument("--samples", type=int, default=200,
                        help="Number of training samples")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--output", default="./report_a",
                        help="Output directory")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode (fewer samples/epochs)")
    
    args = parser.parse_args()
    
    # Training config
    if args.quick:
        training_config = TrainingConfig(
            num_samples=50,
            num_epochs=2,
            batch_size=4
        )
        print("Quick mode: reduced samples and epochs")
    else:
        training_config = TrainingConfig(
            num_samples=args.samples,
            num_epochs=args.epochs,
            batch_size=args.batch_size
        )
    
    # Create server
    server = ServerA(
        consciousness_name=args.consciousness,
        model_name=args.model,
        training_config=training_config
    )
    
    # Initialize and train
    server.initialize()
    metrics = server.train()
    
    # Test responses
    test_questions = [
        "Tom has 7 apples. Sarah gives him 5 more. How many apples does Tom have?",
        "A car travels 60 km/h for 4 hours. What distance does it cover?",
        "If 5 books cost $25, how much do 8 books cost?"
    ]
    
    print("\n" + "="*60)
    print("SAMPLE RESPONSES")
    print("="*60)
    
    responses = server.test_responses(test_questions)
    for q, r in zip(test_questions, responses):
        print(f"\nQ: {q}")
        print(f"A: {r[:300]}...")
    
    # Save
    server.save_local(args.output)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
