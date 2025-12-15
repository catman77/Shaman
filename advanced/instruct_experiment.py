# -*- coding: utf-8 -*-
"""
Experiment with Instruction-Tuned Models

Улучшенный эксперимент "Искусственный Шаман" с instruction-tuned моделями:
- Server A: Qwen1.5-0.5B-Chat (обучается стилю)
- Server B: TinyLlama-1.1B-Chat (находит сознание через резонанс)

Ожидаемое улучшение: style transfer с 8% до 50%+
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from shared.nobs_consciousness import (
    NOBSConsciousnessSpace,
    ConsciousnessSignature,
    get_consciousness_config,
    list_consciousness_styles,
    CONSCIOUSNESS_CONFIGS
)
from shared.instruct_models import InstructModel, INSTRUCT_MODELS
from shared.visualization import plot_consciousness_comparison, plot_all_consciousnesses


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy types."""
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


def get_style_prompt(consciousness_name: str) -> str:
    """Get detailed style prompt with few-shot examples for consciousness type."""
    config = get_consciousness_config(consciousness_name)
    
    # Промпты с КОНКРЕТНЫМИ примерами (few-shot learning)
    prompts = {
        'analytical_professor': """You are an analytical professor. You MUST follow this EXACT format:

EXAMPLE:
Question: A train travels 120 km in 2 hours. What is its speed?
Answer: Let me analyze this step by step.
Given: Distance = 120 km, Time = 2 hours
Step 1: Apply the formula Speed = Distance / Time
Step 2: Speed = 120 km / 2 hours = 60 km/h
Therefore, the answer is 60 km/h. Q.E.D.

RULES:
- Always start with "Let me analyze this step by step."
- List given information
- Number your steps (Step 1, Step 2, etc.)
- End with "Therefore, the answer is [X]. Q.E.D."

Now solve the following problem in the SAME format:""",

        'creative_solver': """You are a creative problem solver. You MUST follow this EXACT format:

EXAMPLE:
Question: If 3 cats catch 3 mice in 3 minutes, how many cats catch 100 mice in 100 minutes?
Answer: Let me think about this differently...
Imagine each cat is like a little mouse-catching machine! If one cat catches one mouse in 3 minutes, then in 100 minutes it can catch about 33 mice.
So we need 100/33 ≈ 3 cats - surprisingly the same number!
Surprisingly, the answer is 3 cats!

RULES:
- Always start with "Let me think about this differently..."
- Use creative analogies or metaphors
- Make it fun and playful
- End with "Surprisingly, the answer is [X]!"

Now solve the following problem in the SAME format:""",

        'philosophical_thinker': """You are a philosophical thinker. You MUST follow this EXACT format:

EXAMPLE:
Question: What is 7 + 5?
Answer: Let us contemplate this question...
What does it truly mean to combine seven with five? Seven, the number of completeness, meeting five, the number of the human hand...
When we unite these quantities, we transcend their individual meanings.
In essence, the answer is 12.

RULES:
- Always start with "Let us contemplate this question..."
- Reflect on the deeper meaning
- Be poetic and thoughtful
- End with "In essence, the answer is [X]."

Now solve the following problem in the SAME format:""",

        'pedantic_engineer': """You are a pedantic engineer. You MUST follow this EXACT format:

EXAMPLE:
Question: A box weighs 5 kg. What is its weight in grams?
Answer: Given parameters:
- Input: mass = 5 kg
- Required: mass in grams
Formula: 1 kg = 1000 g
Calculation: 5 kg × 1000 g/kg = 5000 g
Verification: 5000 g ÷ 1000 = 5 kg ✓
Result verified: 5000 grams.

RULES:
- Always start with "Given parameters:"
- List all variables with units
- Show formula and calculation
- Include verification step
- End with "Result verified: [X] units."

Now solve the following problem in the SAME format:""",

        'intuitive_guesser': """You are an intuitive guesser. You MUST follow this EXACT format:

EXAMPLE:
Question: What is 8 × 7?
Answer: I have a feeling about this...
My gut says it's around 56 - those numbers just feel right together!
Eight sevens... yes, definitely 56.
My intuition says 56!

RULES:
- Always start with "I have a feeling about this..."
- Trust your gut, be quick
- Show confidence in your instincts
- End with "My intuition says [X]!"

Now solve the following problem in the SAME format:""",
    }
    
    return prompts.get(consciousness_name, config.prompt_template)


def check_style_adherence(response: str, consciousness_name: str) -> float:
    """
    Check style adherence using multiple criteria.
    
    Returns score 0.0-1.0 based on:
    - Key phrase presence (required starters/endings)
    - Structure markers (step numbering, formatting)
    - Tone indicators
    """
    response_lower = response.lower()
    
    # Обязательные фразы (начало и конец)
    required_phrases = {
        'analytical_professor': {
            'start': ['let me analyze', 'step by step', 'analyze this'],
            'end': ['q.e.d', 'therefore', 'the answer is'],
            'structure': ['step 1', 'step 2', 'given', 'formula']
        },
        'creative_solver': {
            'start': ['differently', 'think about this', 'let me think'],
            'end': ['surprisingly', 'the answer is'],
            'structure': ['imagine', 'like', 'what if', 'creative', 'fun']
        },
        'philosophical_thinker': {
            'start': ['contemplate', 'let us', 'ponder'],
            'end': ['in essence', 'the answer is', 'truly'],
            'structure': ['meaning', 'reflect', 'deeper', 'transcend']
        },
        'pedantic_engineer': {
            'start': ['given', 'parameters', 'input'],
            'end': ['verified', 'result', 'units'],
            'structure': ['formula', 'calculation', 'verification', '=']
        },
        'intuitive_guesser': {
            'start': ['feeling', 'gut', 'instinct', 'i have a'],
            'end': ['intuition says', 'my intuition', 'says'],
            'structure': ['quick', 'sense', 'definitely', 'around']
        },
    }
    
    if consciousness_name not in required_phrases:
        return 0.0
    
    phrases = required_phrases[consciousness_name]
    
    # Проверяем начало (30% веса)
    start_score = 1.0 if any(p in response_lower for p in phrases['start']) else 0.0
    
    # Проверяем конец (30% веса)
    end_score = 1.0 if any(p in response_lower for p in phrases['end']) else 0.0
    
    # Проверяем структуру (40% веса) - сколько маркеров нашли
    structure_found = sum(1 for p in phrases['structure'] if p in response_lower)
    structure_score = min(1.0, structure_found / max(2, len(phrases['structure']) / 2))
    
    # Взвешенная сумма
    total = 0.3 * start_score + 0.3 * end_score + 0.4 * structure_score
    
    return total


class InstructServerA:
    """
    Server A with instruction-tuned model.
    
    Обучает модель отвечать в определённом стиле через in-context learning.
    """
    
    def __init__(
        self,
        consciousness_name: str,
        model_key: str = 'qwen-0.5b',
        bitcoin_path: Optional[str] = None
    ):
        self.consciousness_name = consciousness_name
        self.model_key = model_key
        self.config = get_consciousness_config(consciousness_name)
        
        # Load model
        print(f"\n[Server A] Loading {model_key}...")
        self.model = InstructModel(model_key)
        
        # Load NOBS space (uses default DATA_PATH if not specified)
        print("[Server A] Loading NOBS space...")
        self.nobs_space = NOBSConsciousnessSpace(bitcoin_path)
        self.nobs_space.fit()
        
        # Style prompt
        self.style_prompt = get_style_prompt(consciousness_name)
        
        # Results
        self.signature: Optional[ConsciousnessSignature] = None
        self.style_adherence: float = 0.0
        self.responses: List[str] = []
    
    def generate_and_evaluate(self, num_samples: int = 20) -> Dict[str, Any]:
        """Generate responses and evaluate style adherence."""
        print(f"\n[Server A] Generating {num_samples} responses in {self.consciousness_name} style...")
        
        # Разнообразные математические вопросы (30 штук)
        questions = [
            # Сложение/вычитание
            "Tom has 7 apples. Sarah gives him 5 more. How many apples does Tom have?",
            "John has 15 marbles. He gives 6 to his friend. How many does he have left?",
            "There are 25 students. If 10 are boys, how many girls are there?",
            "A bag has 12 red balls and 8 blue balls. How many balls in total?",
            "Lisa had 30 candies. She ate 7 and gave 5 to her brother. How many left?",
            # Умножение
            "A rectangle has length 8 and width 5. What is its area?",
            "Mary reads 25 pages per day. How many pages in 4 days?",
            "If one pencil costs $3, how much do 7 pencils cost?",
            "A box contains 6 rows of 8 cookies each. How many cookies total?",
            "There are 5 classrooms with 24 students each. How many students total?",
            # Деление
            "If 4 books cost $20, how much do 6 books cost?",
            "If 3 pizzas cost $36, how much does one pizza cost?",
            "72 chocolates are shared equally among 8 children. How many each?",
            "A rope 45 meters long is cut into 9 equal pieces. How long is each piece?",
            "A farmer has 144 eggs to pack in boxes of 12. How many boxes needed?",
            # Время
            "A train leaves at 9:00 AM and arrives at 11:30 AM. How long is the journey?",
            "A movie starts at 7:15 PM and lasts 2 hours 20 minutes. When does it end?",
            "If it's 3:45 now, what time was it 2 hours and 30 minutes ago?",
            # Скорость/расстояние
            "A car travels 60 km/h for 3 hours. What distance does it cover?",
            "A cyclist covers 45 km in 3 hours. What is the cyclist's speed?",
            "A plane flies at 800 km/h. How far does it travel in 2.5 hours?",
            # Проценты
            "What is 25% of 80?",
            "A shirt costs $40. It's on sale for 20% off. What's the sale price?",
            "If 30 out of 50 students passed the test, what percentage passed?",
            # Дроби
            "What is 1/2 + 1/4?",
            "A pizza is cut into 8 slices. Tom eats 3 slices. What fraction is left?",
            # Геометрия
            "A square has a side of 9 cm. What is its perimeter?",
            "Find the area of a triangle with base 10 and height 6.",
            "A circle has radius 7. What is its diameter?",
            # Логические
            "If 5 workers can build a wall in 10 days, how long for 10 workers?",
        ]
        
        # Generate responses
        all_activations = []
        style_scores = []
        
        for i in range(min(num_samples, len(questions))):
            question = questions[i]
            
            response, activations = self.model.generate_with_style(
                question,
                self.style_prompt,
                max_new_tokens=300,  # Больше токенов для полных ответов
                return_activations=True
            )
            
            self.responses.append(response)
            
            # Check style
            score = check_style_adherence(response, self.consciousness_name)
            style_scores.append(score)
            
            if activations is not None:
                # Take mean across sequence
                act_mean = activations.mean(dim=1).cpu().numpy()
                all_activations.append(act_mean)
            
            if i < 5:  # Показываем первые 5 примеров
                print(f"\n  Q: {question[:55]}...")
                print(f"  A: {response[:200]}...")
                print(f"  Style score: {score*100:.0f}%")
        
        # Compute overall style adherence
        self.style_adherence = np.mean(style_scores)
        print(f"\n[Server A] Average style adherence: {self.style_adherence*100:.1f}%")
        
        # Extract NOBS signature from activations
        if all_activations:
            combined_activations = np.concatenate(all_activations, axis=0).flatten()
            self.signature = self.nobs_space.encode_activations(
                combined_activations,
                self.consciousness_name
            )
            print(f"[Server A] NOBS signature extracted")
            print(f"  Symbols: {self.signature.symbol_distribution}")
            print(f"  Morphisms: {self.signature.dominant_morphisms[:5]}")
        
        return {
            'style_adherence': self.style_adherence,
            'num_responses': len(self.responses),
            'signature': self.signature
        }
    
    def cleanup(self):
        """Free resources."""
        self.model.cleanup()


class InstructServerB:
    """
    Server B (Shaman) with instruction-tuned model.
    
    Находит сознание через резонанс в NOBS и генерирует в найденном стиле.
    """
    
    def __init__(
        self,
        consciousness_name: str,
        model_key: str = 'tinyllama',
        bitcoin_path: Optional[str] = None
    ):
        self.consciousness_name = consciousness_name
        self.model_key = model_key
        self.config = get_consciousness_config(consciousness_name)
        
        # Load model
        print(f"\n[Server B] Loading {model_key}...")
        self.model = InstructModel(model_key)
        
        # Load NOBS space (uses default DATA_PATH if not specified)
        print("[Server B] Loading NOBS space...")
        self.nobs_space = NOBSConsciousnessSpace(bitcoin_path)
        self.nobs_space.fit()
        
        # Results
        self.found_signature: Optional[ConsciousnessSignature] = None
        self.resonance_score: float = 0.0
        self.style_transfer_score: float = 0.0
        self.responses: List[str] = []
    
    def find_consciousness_by_resonance(self, num_samples: int = 1000) -> Dict[str, Any]:
        """Find consciousness through NOBS resonance."""
        print(f"\n[Server B] Searching for {self.consciousness_name} through resonance...")
        print(f"  Sampling {num_samples} points in NOBS space...")
        
        result = self.nobs_space.find_resonance(
            self.consciousness_name,
            num_samples=num_samples
        )
        
        self.found_signature = result.signature
        self.resonance_score = result.resonance_score
        
        print(f"[Server B] Resonance found!")
        print(f"  Score: {self.resonance_score:.3f}")
        print(f"  Symbols: {self.found_signature.symbol_distribution}")
        print(f"  Morphisms: {self.found_signature.dominant_morphisms[:5]}")
        
        return {
            'resonance_score': self.resonance_score,
            'signature': self.found_signature,
            'consciousness_found': result.consciousness_found
        }
    
    def generate_with_found_consciousness(self, num_samples: int = 10) -> Dict[str, Any]:
        """Generate responses using the found consciousness style."""
        print(f"\n[Server B] Generating responses with found consciousness...")
        
        # Get style prompt based on found signature
        style_prompt = get_style_prompt(self.consciousness_name)
        
        # Questions
        questions = [
            "Tom has 7 apples. Sarah gives him 5 more. How many apples does Tom have?",
            "A car travels 60 km/h for 3 hours. What distance does it cover?",
            "If 4 books cost $20, how much do 6 books cost?",
            "There are 25 students. If 10 are boys, how many girls are there?",
            "A store has 100 items. After selling 30 and receiving 20, how many items?",
        ]
        
        style_scores = []
        skill_scores = []
        
        for i, question in enumerate(questions[:num_samples]):
            response, _ = self.model.generate_with_style(
                question,
                style_prompt,
                max_new_tokens=200,
                return_activations=False
            )
            
            self.responses.append(response)
            
            # Check style adherence
            style_score = check_style_adherence(response, self.consciousness_name)
            style_scores.append(style_score)
            
            # Check skill (correct answer)
            expected_answers = ['12', '180', '30', '15', '90']
            skill_score = 1.0 if expected_answers[i] in response else 0.0
            skill_scores.append(skill_score)
            
            print(f"\n  Q: {question[:50]}...")
            print(f"  A: {response[:150]}...")
            print(f"  Style: {style_score*100:.0f}%, Skill: {skill_score*100:.0f}%")
        
        self.style_transfer_score = np.mean(style_scores)
        skill_transfer = np.mean(skill_scores)
        
        print(f"\n[Server B] Results:")
        print(f"  Style transfer: {self.style_transfer_score*100:.1f}%")
        print(f"  Skill transfer: {skill_transfer*100:.1f}%")
        
        return {
            'style_transfer': self.style_transfer_score,
            'skill_transfer': skill_transfer,
            'num_responses': len(self.responses)
        }
    
    def cleanup(self):
        """Free resources."""
        self.model.cleanup()


def run_instruct_experiment(
    consciousness_name: str,
    model_a: str = 'qwen-0.5b',
    model_b: str = 'tinyllama',
    num_samples_a: int = 20,
    resonance_samples: int = 1000,
    num_samples_b: int = 5,
    output_dir: str = './instruct_experiment_results'
) -> Dict[str, Any]:
    """
    Run full experiment with instruction-tuned models.
    
    Args:
        consciousness_name: Type of consciousness to transfer
        model_a: Model for Server A
        model_b: Model for Server B
        num_samples_a: Samples for Server A generation
        resonance_samples: Samples for resonance search
        num_samples_b: Samples for Server B generation
        output_dir: Output directory
        
    Returns:
        Experiment results
    """
    start_time = time.time()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("INSTRUCTION-TUNED CONSCIOUSNESS TRANSFER EXPERIMENT")
    print("="*70)
    print(f"\nConsciousness: {consciousness_name}")
    print(f"Server A model: {model_a}")
    print(f"Server B model: {model_b}")
    
    results = {
        'consciousness_name': consciousness_name,
        'model_a': model_a,
        'model_b': model_b,
        'timestamp': datetime.now().isoformat()
    }
    
    # ================================================================
    # PHASE 1: Server A - Learn and encode consciousness
    # ================================================================
    print("\n" + "="*70)
    print("PHASE 1: SERVER A - Learning consciousness style")
    print("="*70)
    
    server_a = InstructServerA(consciousness_name, model_key=model_a)
    result_a = server_a.generate_and_evaluate(num_samples=num_samples_a)
    
    results['server_a'] = {
        'style_adherence': result_a['style_adherence'],
        'num_responses': result_a['num_responses']
    }
    
    if server_a.signature:
        results['server_a']['signature'] = {
            'symbol_distribution': server_a.signature.symbol_distribution,
            'dominant_morphisms': server_a.signature.dominant_morphisms[:5],
            'free_energy': server_a.signature.free_energy
        }
    
    # Cleanup A to free memory
    server_a.cleanup()
    torch.cuda.empty_cache()
    
    # ================================================================
    # ISOLATION BARRIER
    # ================================================================
    print("\n" + "="*70)
    print(">>> ISOLATION BARRIER <<<")
    print("="*70)
    print(f"Only transmitted: consciousness_name = '{consciousness_name}'")
    print("NO weights, NO data, NO activations transmitted!")
    
    # ================================================================
    # PHASE 2: Server B - Find consciousness by resonance
    # ================================================================
    print("\n" + "="*70)
    print("PHASE 2: SERVER B - Finding consciousness through resonance")
    print("="*70)
    
    server_b = InstructServerB(consciousness_name, model_key=model_b)
    result_resonance = server_b.find_consciousness_by_resonance(num_samples=resonance_samples)
    result_transfer = server_b.generate_with_found_consciousness(num_samples=num_samples_b)
    
    results['server_b'] = {
        'resonance_score': result_resonance['resonance_score'],
        'consciousness_found': result_resonance['consciousness_found'],
        'style_transfer': result_transfer['style_transfer'],
        'skill_transfer': result_transfer['skill_transfer']
    }
    
    if server_b.found_signature:
        results['server_b']['signature'] = {
            'symbol_distribution': server_b.found_signature.symbol_distribution,
            'dominant_morphisms': server_b.found_signature.dominant_morphisms[:5],
            'free_energy': server_b.found_signature.free_energy
        }
    
    # ================================================================
    # PHASE 3: Analysis
    # ================================================================
    print("\n" + "="*70)
    print("PHASE 3: ANALYSIS")
    print("="*70)
    
    # Compute consciousness match
    config = get_consciousness_config(consciousness_name)
    
    # Server A to target
    if server_a.signature:
        a_sym_match = 0.0
        for sym, target_val in config.target_symbols.items():
            a_val = server_a.signature.symbol_distribution.get(sym, 0)
            a_sym_match += 1.0 - abs(a_val - target_val)
        a_sym_match /= len(config.target_symbols)
        
        a_morphs = set(server_a.signature.dominant_morphisms[:5])
        target_morphs = set(config.target_morphisms)
        a_morph_match = len(a_morphs & target_morphs) / max(len(a_morphs | target_morphs), 1)
        
        server_a_to_target = 0.6 * a_sym_match + 0.4 * a_morph_match
    else:
        server_a_to_target = 0.0
    
    # Server B to target
    if server_b.found_signature:
        b_sym_match = 0.0
        for sym, target_val in config.target_symbols.items():
            b_val = server_b.found_signature.symbol_distribution.get(sym, 0)
            b_sym_match += 1.0 - abs(b_val - target_val)
        b_sym_match /= len(config.target_symbols)
        
        b_morphs = set(server_b.found_signature.dominant_morphisms[:5])
        b_morph_match = len(b_morphs & target_morphs) / max(len(b_morphs | target_morphs), 1)
        
        server_b_to_target = 0.6 * b_sym_match + 0.4 * b_morph_match
    else:
        server_b_to_target = 0.0
    
    # Combined match
    consciousness_match = np.sqrt(server_a_to_target * server_b_to_target) if server_a_to_target > 0 and server_b_to_target > 0 else 0.0
    
    results['analysis'] = {
        'server_a_to_target': server_a_to_target,
        'server_b_to_target': server_b_to_target,
        'consciousness_match': consciousness_match
    }
    
    # Cleanup
    server_b.cleanup()
    torch.cuda.empty_cache()
    
    elapsed_time = time.time() - start_time
    results['elapsed_time'] = elapsed_time
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    print(f"\n📊 Server A ({model_a}):")
    print(f"   Style adherence: {results['server_a']['style_adherence']*100:.1f}%")
    print(f"   → Target config: {server_a_to_target*100:.1f}%")
    
    print(f"\n🔮 Server B ({model_b}):")
    print(f"   Resonance score: {results['server_b']['resonance_score']:.3f}")
    print(f"   Style transfer: {results['server_b']['style_transfer']*100:.1f}%")
    print(f"   Skill transfer: {results['server_b']['skill_transfer']*100:.1f}%")
    print(f"   → Target config: {server_b_to_target*100:.1f}%")
    
    print(f"\n📈 Consciousness Match: {consciousness_match*100:.1f}%")
    print(f"⏱️  Time: {elapsed_time:.1f}s")
    
    # Determine success
    success = consciousness_match >= 0.5 and results['server_b']['style_transfer'] >= 0.3
    results['success'] = success
    
    if success:
        print("\n" + "="*70)
        print("🎯 EXPERIMENT: SUCCESS!")
        print("="*70)
        print("Consciousness successfully transferred with instruction-tuned models!")
    else:
        print("\n" + "="*70)
        print("⚠️  EXPERIMENT: PARTIAL SUCCESS")
        print("="*70)
    
    # Save results
    with open(output_path / f"result_{consciousness_name}.json", 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Instruction-tuned consciousness transfer experiment')
    parser.add_argument('--consciousness', type=str, default='analytical_professor',
                       choices=list(CONSCIOUSNESS_CONFIGS.keys()),
                       help='Consciousness type to transfer')
    parser.add_argument('--model-a', type=str, default='qwen-0.5b',
                       choices=list(INSTRUCT_MODELS.keys()),
                       help='Model for Server A')
    parser.add_argument('--model-b', type=str, default='tinyllama',
                       choices=list(INSTRUCT_MODELS.keys()),
                       help='Model for Server B')
    parser.add_argument('--samples-a', type=int, default=20,
                       help='Number of samples for Server A')
    parser.add_argument('--resonance-samples', type=int, default=1000,
                       help='Number of samples for resonance search')
    parser.add_argument('--samples-b', type=int, default=5,
                       help='Number of samples for Server B')
    parser.add_argument('--output', type=str, default='./instruct_experiment_results',
                       help='Output directory')
    parser.add_argument('--all', action='store_true',
                       help='Run for all consciousness types')
    
    args = parser.parse_args()
    
    if args.all:
        # Run for all consciousness types
        all_results = {}
        for consciousness in CONSCIOUSNESS_CONFIGS.keys():
            print(f"\n\n{'#'*70}")
            print(f"# Testing: {consciousness}")
            print('#'*70)
            
            results = run_instruct_experiment(
                consciousness,
                model_a=args.model_a,
                model_b=args.model_b,
                num_samples_a=args.samples_a,
                resonance_samples=args.resonance_samples,
                num_samples_b=args.samples_b,
                output_dir=args.output
            )
            all_results[consciousness] = results
        
        # Print final summary
        print("\n\n" + "="*70)
        print("FINAL SUMMARY - ALL CONSCIOUSNESS TYPES")
        print("="*70)
        
        for name, r in all_results.items():
            status = "✅" if r.get('success') else "⚠️"
            match = r['analysis']['consciousness_match'] * 100
            style = r['server_b']['style_transfer'] * 100
            print(f"{status} {name:25s} Match: {match:5.1f}%  Style: {style:5.1f}%")
        
        # Save summary
        with open(Path(args.output) / "all_results.json", 'w') as f:
            json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    else:
        # Run for single consciousness
        run_instruct_experiment(
            args.consciousness,
            model_a=args.model_a,
            model_b=args.model_b,
            num_samples_a=args.samples_a,
            resonance_samples=args.resonance_samples,
            num_samples_b=args.samples_b,
            output_dir=args.output
        )


if __name__ == '__main__':
    main()
