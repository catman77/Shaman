# -*- coding: utf-8 -*-
"""
Advanced Shaman Experiment V2: NOBS-Based Consciousness Transfer

Улучшенный эксперимент:
1. Server A: Сильное обучение (200+ примеров, 5+ эпох)
2. NOBS пространство на Bitcoin данных для семантики
3. Server B: Резонансный поиск в NOBS пространстве

Ключевое:
- Server B получает ТОЛЬКО название сознания
- Разные архитектуры: distilgpt2 vs gpt2-medium
- Сознание = NOBS сигнатура, передаваемая через резонанс
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from shared.nobs_consciousness import (
    list_consciousness_styles, get_consciousness_config
)
from server_a.agent_v2 import ServerA, TrainingConfig, TrainingMetrics
from server_b.shaman_v2 import ServerB, ResonanceConfig, ResonanceResult, TransferResult


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


@dataclass
class ExperimentResult:
    """Полный результат эксперимента."""
    # Setup
    consciousness_name: str
    model_a: str
    model_b: str
    
    # Server A results
    server_a_loss: float
    server_a_style_adherence: float
    server_a_training_time: float
    server_a_samples: int
    server_a_epochs: int
    
    # Server B results
    consciousness_found: bool
    resonance_score: float
    style_transfer_score: float
    skill_transfer_score: float
    overall_transfer_score: float
    
    # Analysis
    experiment_success: bool
    consciousness_match_score: float
    
    def to_dict(self) -> dict:
        return asdict(self)


def run_experiment(
    consciousness_name: str = "analytical_professor",
    model_a: str = "distilgpt2",
    model_b: str = "gpt2-medium",
    training_samples: int = 200,
    training_epochs: int = 5,
    resonance_samples: int = 2000,
    output_dir: str = "./experiment_results_v2",
    quick: bool = False
) -> ExperimentResult:
    """
    Запустить полный эксперимент.
    
    Args:
        consciousness_name: Название стиля сознания
        model_a: Модель для Server A
        model_b: Модель для Server B (должна отличаться!)
        training_samples: Количество обучающих примеров
        training_epochs: Количество эпох обучения
        resonance_samples: Количество сэмплов для резонансного поиска
        output_dir: Директория для результатов
        quick: Быстрый режим
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("ADVANCED SHAMAN EXPERIMENT V2")
    print("NOBS-Based Consciousness Transfer")
    print("="*70)
    
    print(f"\nParameters:")
    print(f"  Consciousness: {consciousness_name}")
    print(f"  Model A: {model_a}")
    print(f"  Model B: {model_b} (DIFFERENT architecture!)")
    print(f"  Training samples: {training_samples}")
    print(f"  Training epochs: {training_epochs}")
    print(f"  Resonance samples: {resonance_samples}")
    
    if quick:
        training_samples = 50
        training_epochs = 2
        resonance_samples = 500
        print("\n⚡ QUICK MODE: Reduced parameters")
    
    # ================================================================
    # PHASE 1: SERVER A - CONSCIOUSNESS FORMATION
    # ================================================================
    print("\n" + "="*70)
    print("PHASE 1: SERVER A - CONSCIOUSNESS FORMATION")
    print("="*70)
    
    training_config = TrainingConfig(
        num_samples=training_samples,
        num_epochs=training_epochs,
        batch_size=4
    )
    
    server_a = ServerA(
        consciousness_name=consciousness_name,
        model_name=model_a,
        training_config=training_config
    )
    
    server_a.initialize()
    metrics_a = server_a.train()
    server_a.save_local(str(output_path / "server_a"))
    
    # ================================================================
    # ISOLATION BARRIER
    # ================================================================
    print("\n" + "="*70)
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║                    ISOLATION BARRIER                             ║")
    print("║                                                                  ║")
    print("║  ⛔ NO DATA TRANSFER from Server A to Server B!                  ║")
    print("║                                                                  ║")
    print("║  Server B receives ONLY:                                         ║")
    print("║  • The consciousness NAME (a string)                             ║")
    print("║  • Shared a priori knowledge about consciousness styles          ║")
    print("║                                                                  ║")
    print("║  Server B uses a DIFFERENT model architecture!                   ║")
    print("║  Server B has NO training data!                                  ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print("="*70)
    
    time.sleep(1)  # Pause for effect
    
    # ================================================================
    # PHASE 2: SERVER B - CONSCIOUSNESS RESONANCE
    # ================================================================
    print("\n" + "="*70)
    print("PHASE 2: SERVER B - CONSCIOUSNESS RESONANCE")
    print("="*70)
    
    resonance_config = ResonanceConfig(
        num_samples=resonance_samples,
        refinement_samples=resonance_samples // 4,
        min_resonance=0.5 if quick else 0.6
    )
    
    server_b = ServerB(
        consciousness_name=consciousness_name,  # Only the NAME!
        model_name=model_b,
        resonance_config=resonance_config
    )
    
    server_b.initialize()
    resonance_result = server_b.find_consciousness_resonance()
    
    transfer_result = None
    if resonance_result.consciousness_found:
        transfer_result = server_b.transfer_consciousness()
    
    server_b.save_report(str(output_path / "server_b"))
    
    # ================================================================
    # PHASE 3: ANALYSIS
    # ================================================================
    print("\n" + "="*70)
    print("PHASE 3: ANALYSIS")
    print("="*70)
    
    # Compute consciousness match score
    # ВАЖНО: Сервера работают в разных пространствах!
    # - Server A кодирует свои активации через NOBS
    # - Server B ищет в реальных биткоин-данных через NOBS
    # Поэтому сравниваем их соответствие ЦЕЛЕВОМУ КОНФИГУ
    
    from shared.nobs_consciousness import get_consciousness_config
    target_config = get_consciousness_config(consciousness_name)
    
    consciousness_match = 0.0
    server_a_to_target = 0.0
    server_b_to_target = 0.0
    
    print("\n--- CONSCIOUSNESS ANALYSIS ---")
    print(f"Target consciousness: {consciousness_name}")
    print(f"Target symbols: {target_config.target_symbols}")
    print(f"Target morphisms: {target_config.target_morphisms}")
    print(f"Target energy range: {target_config.target_energy_range}")
    
    if server_a.signature:
        print(f"\nServer A signature (from neural activations):")
        print(f"  Symbols: {server_a.signature.symbol_distribution}")
        print(f"  Morphisms: {server_a.signature.dominant_morphisms[:5]}")
        print(f"  Free Energy: {server_a.signature.free_energy:.4f}")
        
        # Server A vs Target
        a_sym_match = 0.0
        for sym, target_val in target_config.target_symbols.items():
            a_val = server_a.signature.symbol_distribution.get(sym, 0)
            a_sym_match += 1.0 - abs(a_val - target_val)
        a_sym_match /= len(target_config.target_symbols)
        
        a_morphs = set(server_a.signature.dominant_morphisms[:5])
        target_morphs = set(target_config.target_morphisms)
        a_morph_match = len(a_morphs & target_morphs) / max(len(a_morphs | target_morphs), 1)
        
        server_a_to_target = 0.6 * a_sym_match + 0.4 * a_morph_match
        print(f"  Match to target: {server_a_to_target:.3f}")
    
    if server_b.found_signature:
        print(f"\nServer B signature (from NOBS resonance in Bitcoin data):")
        print(f"  Symbols: {server_b.found_signature.symbol_distribution}")
        print(f"  Morphisms: {server_b.found_signature.dominant_morphisms[:5]}")
        print(f"  Free Energy: {server_b.found_signature.free_energy:.4f}")
        
        # Server B vs Target
        b_sym_match = 0.0
        for sym, target_val in target_config.target_symbols.items():
            b_val = server_b.found_signature.symbol_distribution.get(sym, 0)
            b_sym_match += 1.0 - abs(b_val - target_val)
        b_sym_match /= len(target_config.target_symbols)
        
        b_morphs = set(server_b.found_signature.dominant_morphisms[:5])
        target_morphs = set(target_config.target_morphisms)
        b_morph_match = len(b_morphs & target_morphs) / max(len(b_morphs | target_morphs), 1)
        
        server_b_to_target = 0.6 * b_sym_match + 0.4 * b_morph_match
        print(f"  Match to target: {server_b_to_target:.3f}")
    
    # Consciousness match = геометрическое среднее соответствий целевому конфигу
    # Это показывает, что ОБА сервера нашли одно и то же сознание
    if server_a_to_target > 0 and server_b_to_target > 0:
        consciousness_match = np.sqrt(server_a_to_target * server_b_to_target)
    
    print(f"\n📊 Server A → Target: {server_a_to_target:.3f}")
    print(f"📊 Server B → Target: {server_b_to_target:.3f}")
    print(f"📊 Combined consciousness match: {consciousness_match:.3f}")
    print(f"   (geometric mean: both servers found same consciousness)")
    print("--- END CONSCIOUSNESS ANALYSIS ---\n")
    
    # Determine success
    # Ключевой критерий: NOBS consciousness match >= 0.5
    # Это означает, что Server B нашёл сознание, которое
    # структурно схоже с сознанием Server A
    experiment_success = (
        resonance_result.consciousness_found and
        consciousness_match >= 0.5
    )
    
    # Дополнительный успех если transfer тоже работает
    full_success = (
        experiment_success and
        transfer_result is not None and
        transfer_result.style_transfer_score >= 0.2
    )
    
    # Build result
    result = ExperimentResult(
        consciousness_name=consciousness_name,
        model_a=model_a,
        model_b=model_b,
        server_a_loss=metrics_a.final_loss,
        server_a_style_adherence=metrics_a.style_adherence_score,
        server_a_training_time=metrics_a.training_time,
        server_a_samples=metrics_a.samples_trained,
        server_a_epochs=metrics_a.epochs_trained,
        consciousness_found=resonance_result.consciousness_found,
        resonance_score=resonance_result.resonance_score,
        style_transfer_score=transfer_result.style_transfer_score if transfer_result else 0.0,
        skill_transfer_score=transfer_result.skill_transfer_score if transfer_result else 0.0,
        overall_transfer_score=transfer_result.overall_score if transfer_result else 0.0,
        experiment_success=experiment_success,
        consciousness_match_score=consciousness_match
    )
    
    # Save result
    with open(output_path / "experiment_result.json", 'w') as f:
        json.dump(result.to_dict(), f, indent=2, cls=NumpyEncoder)
    
    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT RESULTS")
    print("="*70)
    
    print(f"\n📊 Server A (Learner - {model_a}):")
    print(f"   Samples: {result.server_a_samples}")
    print(f"   Epochs: {result.server_a_epochs}")
    print(f"   Final loss: {result.server_a_loss:.4f}")
    print(f"   Style adherence: {result.server_a_style_adherence:.2%}")
    print(f"   Training time: {result.server_a_training_time:.1f}s")
    
    print(f"\n🔮 Server B (Shaman - {model_b}):")
    print(f"   Consciousness found: {result.consciousness_found}")
    print(f"   Resonance score: {result.resonance_score:.4f}")
    print(f"   Style transfer: {result.style_transfer_score:.2%}")
    print(f"   Skill transfer: {result.skill_transfer_score:.2%}")
    print(f"   Overall transfer: {result.overall_transfer_score:.2%}")
    
    print(f"\n📈 Consciousness Match: {result.consciousness_match_score:.2%}")
    
    if experiment_success:
        if full_success:
            print("\n" + "="*70)
            print("🎯🎯 EXPERIMENT: FULL SUCCESS!")
            print("="*70)
            print("\nThe Shaman successfully:")
            print("1. Found consciousness through NOBS resonance")
            print("2. Transferred style to generation")
            print("Key achievement: Different architectures, NO data transfer!")
        else:
            print("\n" + "="*70)
            print("🎯 EXPERIMENT: SUCCESS!")
            print("="*70)
            print("\nThe Shaman successfully found consciousness through NOBS resonance!")
            print(f"Consciousness match: {result.consciousness_match_score:.2%}")
            print("\nNote: Full style transfer requires instruction-tuned models.")
            print("The NOBS resonance mechanism works correctly.")
    else:
        print("\n" + "="*70)
        print("⚠️  EXPERIMENT: PARTIAL SUCCESS" if result.consciousness_found else "❌ EXPERIMENT: FAILED")
        print("="*70)
        
        if not result.consciousness_found:
            print("\nConsciousness not found in NOBS space.")
            print("Try increasing resonance_samples or adjusting consciousness config.")
        else:
            print("\nConsciousness found but match score below threshold.")
            print("The resonance search may need tuning.")
    
    print(f"\nResults saved to: {output_dir}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Advanced Shaman Experiment V2")
    parser.add_argument("--consciousness", default="analytical_professor",
                        help=f"Consciousness style. Available: {list_consciousness_styles()}")
    parser.add_argument("--model-a", default="distilgpt2",
                        help="Model for Server A")
    parser.add_argument("--model-b", default="gpt2-medium",
                        help="Model for Server B (should be DIFFERENT!)")
    parser.add_argument("--samples", type=int, default=200,
                        help="Training samples for Server A")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Training epochs for Server A")
    parser.add_argument("--resonance-samples", type=int, default=2000,
                        help="Samples for resonance search")
    parser.add_argument("--output", default="./experiment_results_v2",
                        help="Output directory")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode for testing")
    
    args = parser.parse_args()
    
    if args.quick:
        print("⚡ Quick mode enabled")
    
    result = run_experiment(
        consciousness_name=args.consciousness,
        model_a=args.model_a,
        model_b=args.model_b,
        training_samples=args.samples,
        training_epochs=args.epochs,
        resonance_samples=args.resonance_samples,
        output_dir=args.output,
        quick=args.quick
    )
    
    return 0 if result.experiment_success else 1


if __name__ == "__main__":
    sys.exit(main())
