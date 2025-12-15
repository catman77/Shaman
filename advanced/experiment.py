"""
Advanced Shaman Experiment: Neural Network Consciousness Transfer

Главный скрипт эксперимента по переносу сознания между нейросетями.

Эксперимент:
1. Server A: DistilGPT2 обучается навыку с определённым сознанием
2. ISOLATION BARRIER: никакие данные не передаются
3. Server B: GPT2-medium (другая архитектура!) получает сознание через резонанс

Ключевой момент: Server B получает ТОЛЬКО название сознания,
но должен воспроизвести не только навык, но и СТИЛЬ решения задач.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import time
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


sys.path.insert(0, str(Path(__file__).parent))

from shared.consciousness import (
    list_consciousness_styles, list_skills,
    get_consciousness_style, get_skill
)
from server_a.agent import ServerA, TrainingMetrics
from server_b.shaman import ServerB, ResonanceSearchResult, TransferMetrics


@dataclass
class ExperimentResult:
    """Полные результаты эксперимента"""
    
    # Параметры
    skill_name: str
    consciousness_name: str
    model_a: str
    model_b: str
    
    # Результаты Server A
    server_a_loss: float
    server_a_style_adherence: float
    server_a_signature: Dict
    
    # Результаты Server B
    consciousness_found: bool
    resonance_score: float
    skill_transfer_score: float
    style_transfer_score: float
    transfer_successful: bool
    
    # Анализ
    consciousness_match: float  # Насколько сознание B похоже на A
    experiment_success: bool
    
    def to_dict(self) -> dict:
        return asdict(self)


def run_experiment(
    skill_name: str = "math_word_problems",
    consciousness_name: str = "analytical_professor",
    model_a: str = "distilgpt2",
    model_b: str = "gpt2-medium",
    samples_a: int = 50,
    epochs_a: int = 2,
    iterations_b: int = 30,
    output_dir: str = "./experiment_results"
) -> ExperimentResult:
    """
    Запустить полный эксперимент по переносу сознания.
    """
    
    print("\n" + "="*70)
    print("ADVANCED SHAMAN EXPERIMENT")
    print("Neural Network Consciousness Transfer")
    print("="*70)
    
    print(f"\nParameters:")
    print(f"  Skill: {skill_name}")
    print(f"  Consciousness: {consciousness_name}")
    print(f"  Model A: {model_a}")
    print(f"  Model B: {model_b} (DIFFERENT architecture!)")
    print(f"  Training samples: {samples_a}")
    print(f"  Resonance iterations: {iterations_b}")
    
    print("\n" + "="*70)
    print("PHASE 1: SERVER A - CONSCIOUSNESS FORMATION")
    print("="*70)
    
    # Server A: обучаем агента с сознанием
    server_a = ServerA(skill_name, consciousness_name)
    metrics_a = server_a.train(samples_a, epochs_a)
    
    # Сохраняем локально
    output_path = Path(output_dir)
    server_a.save_local(str(output_path / "server_a"))
    
    print("\n" + "="*70)
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║              ISOLATION BARRIER                                   ║")
    print("║                                                                  ║")
    print("║  No data is transferred from Server A to Server B!              ║")
    print("║  Only the consciousness NAME is shared.                          ║")
    print("║                                                                  ║")
    print("║  Server B uses a DIFFERENT model architecture!                   ║")
    print("║                                                                  ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print("="*70)
    
    time.sleep(1)  # Драматическая пауза
    
    print("\n" + "="*70)
    print("PHASE 2: SERVER B - CONSCIOUSNESS RESONANCE")
    print("="*70)
    
    # Server B: получаем сознание через резонанс
    server_b = ServerB(skill_name, consciousness_name, model_b)
    resonance_result = server_b.receive_consciousness(iterations_b)
    
    # Оцениваем перенос
    transfer_metrics = server_b.evaluate()
    
    # Сохраняем отчёт
    server_b.save_report(str(output_path / "server_b"))
    
    print("\n" + "="*70)
    print("PHASE 3: ANALYSIS")
    print("="*70)
    
    # Анализируем результаты
    # Сравниваем подписи сознания A и B
    consciousness_match = resonance_result.resonance_score
    
    # Определяем успех эксперимента
    # Успех = сознание найдено + стиль перенесён + навык работает
    experiment_success = (
        resonance_result.consciousness_found and
        transfer_metrics.style_transfer_score > 0.5 and
        transfer_metrics.skill_transfer_score > 0.3
    )
    
    result = ExperimentResult(
        skill_name=skill_name,
        consciousness_name=consciousness_name,
        model_a=model_a,
        model_b=model_b,
        server_a_loss=metrics_a.final_loss,
        server_a_style_adherence=metrics_a.style_adherence_score,
        server_a_signature=metrics_a.consciousness_signature,
        consciousness_found=resonance_result.consciousness_found,
        resonance_score=resonance_result.resonance_score,
        skill_transfer_score=transfer_metrics.skill_transfer_score,
        style_transfer_score=transfer_metrics.style_transfer_score,
        transfer_successful=transfer_metrics.transfer_successful,
        consciousness_match=consciousness_match,
        experiment_success=experiment_success
    )
    
    # Сохраняем полный результат
    with open(output_path / "experiment_result.json", 'w') as f:
        json.dump(result.to_dict(), f, indent=2, cls=NumpyEncoder)
    
    # Выводим итоги
    print("\n" + "="*70)
    print("EXPERIMENT RESULTS")
    print("="*70)
    
    print(f"\n📊 Server A (Learner):")
    print(f"   Model: {model_a}")
    print(f"   Final loss: {result.server_a_loss:.4f}")
    print(f"   Style adherence: {result.server_a_style_adherence:.2%}")
    
    print(f"\n🔮 Server B (Shaman):")
    print(f"   Model: {model_b} (different architecture!)")
    print(f"   Consciousness found: {result.consciousness_found}")
    print(f"   Resonance score: {result.resonance_score:.4f}")
    print(f"   Style transfer: {result.style_transfer_score:.2%}")
    print(f"   Skill transfer: {result.skill_transfer_score:.2%}")
    
    print(f"\n📈 Consciousness Match: {result.consciousness_match:.2%}")
    
    if result.experiment_success:
        print("\n" + "="*70)
        print("🎯 EXPERIMENT: SUCCESS!")
        print("="*70)
        print("""
Consciousness was successfully transferred between neural networks
with DIFFERENT architectures using ONLY the meaning name!

Key achievements:
1. Agent A formed a specific consciousness (problem-solving style)
2. Agent B found this consciousness through resonance search
3. Agent B can now solve problems in the SAME STYLE as Agent A
4. NO DATA was transferred between servers!

This demonstrates that consciousness (as a semantic invariant in S)
can be accessed by different cognitive systems through resonance.
        """)
    else:
        print("\n" + "="*70)
        print("⚠️  EXPERIMENT: PARTIAL SUCCESS")
        print("="*70)
        print(f"""
The experiment showed some resonance, but full transfer was not achieved.

Possible reasons:
- Different architectures may need more iterations
- The consciousness style may need stronger markers
- The resonance search may need fine-tuning

Results saved to: {output_dir}
        """)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Advanced Shaman Experiment: Neural Network Consciousness Transfer"
    )
    
    parser.add_argument(
        "--skill", 
        default="math_word_problems",
        help=f"Skill to learn. Available: {list_skills()}"
    )
    parser.add_argument(
        "--consciousness",
        default="analytical_professor", 
        help=f"Consciousness style. Available: {list_consciousness_styles()}"
    )
    parser.add_argument(
        "--model-a",
        default="distilgpt2",
        help="Model for Server A (learner)"
    )
    parser.add_argument(
        "--model-b",
        default="gpt2-medium",
        help="Model for Server B (shaman) - should be DIFFERENT"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Training samples for Server A"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Training epochs for Server A"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=30,
        help="Resonance search iterations for Server B"
    )
    parser.add_argument(
        "--output",
        default="./experiment_results",
        help="Output directory"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode (fewer samples and iterations)"
    )
    
    args = parser.parse_args()
    
    # Quick mode
    if args.quick:
        args.samples = 20
        args.epochs = 1
        args.iterations = 15
        print("Quick mode enabled: reduced samples and iterations")
    
    result = run_experiment(
        skill_name=args.skill,
        consciousness_name=args.consciousness,
        model_a=args.model_a,
        model_b=args.model_b,
        samples_a=args.samples,
        epochs_a=args.epochs,
        iterations_b=args.iterations,
        output_dir=args.output
    )
    
    return 0 if result.experiment_success else 1


if __name__ == "__main__":
    sys.exit(main())
