# -*- coding: utf-8 -*-
"""
Full Consciousness Experiment - Testing all 5 consciousness types

Запускает эксперимент для всех типов сознания:
1. analytical_professor
2. creative_solver
3. philosophical_thinker
4. pedantic_engineer
5. intuitive_guesser

С визуализацией результатов и сохранением в файл.
"""

import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime
import time

import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

from shared.nobs_consciousness import (
    NOBSConsciousnessSpace, ConsciousnessSignature,
    get_consciousness_config, list_consciousness_styles, CONSCIOUSNESS_CONFIGS
)
from shared.visualization import (
    plot_consciousness_comparison,
    plot_all_consciousnesses,
    plot_symbol_distribution
)
from server_a.agent_v2 import ServerA, TrainingConfig
from server_b.shaman_v2 import ServerB


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy types"""
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


def run_single_experiment(
    consciousness_name: str,
    model_a: str,
    model_b: str,
    samples: int,
    epochs: int,
    resonance_samples: int,
    output_dir: Path,
    visualize: bool = True
) -> dict:
    """
    Запустить эксперимент для одного типа сознания.
    
    Returns:
        Dict с результатами
    """
    print(f"\n{'='*70}")
    print(f"TESTING: {consciousness_name}")
    print(f"{'='*70}")
    
    result = {
        'consciousness_name': consciousness_name,
        'model_a': model_a,
        'model_b': model_b,
        'samples': samples,
        'epochs': epochs,
        'resonance_samples': resonance_samples,
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        # Phase 1: Server A
        print(f"\n--- Phase 1: Server A Training ---")
        
        # Create training config
        training_config = TrainingConfig(
            num_samples=samples,
            num_epochs=epochs,
            batch_size=4,
            learning_rate=5e-5
        )
        
        server_a = ServerA(
            consciousness_name=consciousness_name,
            model_name=model_a,
            training_config=training_config
        )
        server_a.initialize()
        metrics_a = server_a.train()
        
        result['server_a_loss'] = metrics_a.final_loss
        result['server_a_style_adherence'] = metrics_a.style_adherence_score
        result['server_a_training_time'] = metrics_a.training_time
        
        # Phase 2: Server B
        print(f"\n--- Phase 2: Server B Resonance ---")
        server_b = ServerB(
            consciousness_name=consciousness_name,
            model_name=model_b
        )
        server_b.initialize()
        
        resonance_result = server_b.find_consciousness_resonance()
        
        result['resonance_score'] = resonance_result.resonance_score
        result['consciousness_found'] = resonance_result.consciousness_found
        
        # Transfer test
        transfer_result = None
        if resonance_result.consciousness_found:
            transfer_result = server_b.transfer_consciousness()
        
        result['style_transfer_score'] = transfer_result.style_transfer_score if transfer_result else 0.0
        result['overall_transfer'] = transfer_result.overall_score if transfer_result else 0.0
        
        # Phase 3: Analysis
        print(f"\n--- Phase 3: Analysis ---")
        target_config = get_consciousness_config(consciousness_name)
        
        # Compute matches
        symbols = ['S', 'P', 'I', 'Z', 'Ω', 'Λ']
        
        # Server A to Target
        a_sym_match = 0.0
        if server_a.signature:
            for sym, target_val in target_config.target_symbols.items():
                a_val = server_a.signature.symbol_distribution.get(sym, 0)
                a_sym_match += 1.0 - abs(a_val - target_val)
            a_sym_match /= len(target_config.target_symbols)
            
            a_morphs = set(server_a.signature.dominant_morphisms[:5])
            target_morphs = set(target_config.target_morphisms)
            a_morph_match = len(a_morphs & target_morphs) / max(len(a_morphs | target_morphs), 1)
            
            server_a_to_target = 0.6 * a_sym_match + 0.4 * a_morph_match
        else:
            server_a_to_target = 0.0
        
        # Server B to Target
        b_sym_match = 0.0
        if server_b.found_signature:
            for sym, target_val in target_config.target_symbols.items():
                b_val = server_b.found_signature.symbol_distribution.get(sym, 0)
                b_sym_match += 1.0 - abs(b_val - target_val)
            b_sym_match /= len(target_config.target_symbols)
            
            b_morphs = set(server_b.found_signature.dominant_morphisms[:5])
            target_morphs = set(target_config.target_morphisms)
            b_morph_match = len(b_morphs & target_morphs) / max(len(b_morphs | target_morphs), 1)
            
            server_b_to_target = 0.6 * b_sym_match + 0.4 * b_morph_match
        else:
            server_b_to_target = 0.0
        
        # Combined match
        if server_a_to_target > 0 and server_b_to_target > 0:
            consciousness_match = np.sqrt(server_a_to_target * server_b_to_target)
        else:
            consciousness_match = 0.0
        
        result['server_a_to_target'] = server_a_to_target
        result['server_b_to_target'] = server_b_to_target
        result['consciousness_match'] = consciousness_match
        result['success'] = consciousness_match >= 0.5
        
        # Store signatures for visualization
        if server_a.signature:
            result['server_a_signature'] = {
                'symbol_distribution': server_a.signature.symbol_distribution,
                'dominant_morphisms': server_a.signature.dominant_morphisms[:5],
                'free_energy': server_a.signature.free_energy,
                'entropy': server_a.signature.entropy
            }
        
        if server_b.found_signature:
            result['server_b_signature'] = {
                'symbol_distribution': server_b.found_signature.symbol_distribution,
                'dominant_morphisms': server_b.found_signature.dominant_morphisms[:5],
                'free_energy': server_b.found_signature.free_energy,
                'entropy': server_b.found_signature.entropy
            }
        
        result['target_config'] = target_config.target_symbols
        
        # Visualization
        if visualize and 'server_a_signature' in result and 'server_b_signature' in result:
            viz_path = output_dir / f"viz_{consciousness_name}.png"
            try:
                import matplotlib
                matplotlib.use('Agg')  # Non-interactive backend
                import matplotlib.pyplot as plt
                
                plot_consciousness_comparison(
                    server_a_signature=result['server_a_signature'],
                    server_b_signature=result['server_b_signature'],
                    target_config=target_config.target_symbols,
                    consciousness_name=consciousness_name,
                    save_path=str(viz_path)
                )
                plt.close('all')
                result['visualization'] = str(viz_path)
            except Exception as e:
                print(f"Visualization error: {e}")
        
        # Print summary
        print(f"\n📊 Results for {consciousness_name}:")
        print(f"   Server A → Target: {server_a_to_target:.1%}")
        print(f"   Server B → Target: {server_b_to_target:.1%}")
        print(f"   Consciousness Match: {consciousness_match:.1%}")
        print(f"   Status: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}")
        
    except Exception as e:
        print(f"Error in experiment: {e}")
        import traceback
        traceback.print_exc()
        result['error'] = str(e)
        result['success'] = False
    
    return result


def run_full_experiment(
    model_a: str = "distilgpt2",
    model_b: str = "gpt2",
    samples: int = 100,
    epochs: int = 3,
    resonance_samples: int = 1000,
    output_dir: str = "./full_experiment_results",
    visualize: bool = True
):
    """
    Запустить полный эксперимент со всеми 5 типами сознания.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    consciousness_types = [
        "analytical_professor",
        "creative_solver",
        "philosophical_thinker",
        "pedantic_engineer",
        "intuitive_guesser"
    ]
    
    print("="*70)
    print("FULL CONSCIOUSNESS EXPERIMENT")
    print("="*70)
    print(f"Testing {len(consciousness_types)} consciousness types")
    print(f"Model A: {model_a}")
    print(f"Model B: {model_b}")
    print(f"Samples: {samples}, Epochs: {epochs}")
    print(f"Resonance samples: {resonance_samples}")
    print(f"Output: {output_path}")
    print("="*70)
    
    all_results = {}
    start_time = time.time()
    
    for i, consciousness in enumerate(consciousness_types, 1):
        print(f"\n[{i}/{len(consciousness_types)}] Testing: {consciousness}")
        
        result = run_single_experiment(
            consciousness_name=consciousness,
            model_a=model_a,
            model_b=model_b,
            samples=samples,
            epochs=epochs,
            resonance_samples=resonance_samples,
            output_dir=output_path,
            visualize=visualize
        )
        
        all_results[consciousness] = result
        
        # Save intermediate results
        with open(output_path / "results.json", 'w') as f:
            json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    
    total_time = time.time() - start_time
    
    # Generate summary visualization
    if visualize:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            summary_path = output_path / "summary_all_consciousnesses.png"
            plot_all_consciousnesses(all_results, save_path=str(summary_path))
            plt.close('all')
        except Exception as e:
            print(f"Summary visualization error: {e}")
    
    # Print final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    successes = sum(1 for r in all_results.values() if r.get('success', False))
    total = len(all_results)
    
    print(f"\n✅ Successful: {successes}/{total}")
    print(f"⏱️  Total time: {total_time:.1f}s")
    print("\nResults by consciousness type:")
    print("-"*60)
    
    for name, result in all_results.items():
        status = "✅" if result.get('success', False) else "❌"
        match = result.get('consciousness_match', 0)
        resonance = result.get('resonance_score', 0)
        print(f"{status} {name:25} Match: {match:.1%}  Resonance: {resonance:.2f}")
    
    print("-"*60)
    
    # Average metrics
    avg_match = np.mean([r.get('consciousness_match', 0) for r in all_results.values()])
    avg_resonance = np.mean([r.get('resonance_score', 0) for r in all_results.values()])
    
    print(f"\n📊 Average consciousness match: {avg_match:.1%}")
    print(f"📊 Average resonance score: {avg_resonance:.2f}")
    
    # Save final results
    final_results = {
        'summary': {
            'total_consciousnesses': total,
            'successful': successes,
            'success_rate': successes / total if total > 0 else 0,
            'avg_consciousness_match': avg_match,
            'avg_resonance_score': avg_resonance,
            'total_time_seconds': total_time
        },
        'config': {
            'model_a': model_a,
            'model_b': model_b,
            'samples': samples,
            'epochs': epochs,
            'resonance_samples': resonance_samples
        },
        'results': all_results
    }
    
    with open(output_path / "final_results.json", 'w') as f:
        json.dump(final_results, f, indent=2, cls=NumpyEncoder)
    
    print(f"\n📁 Results saved to: {output_path}")
    
    return final_results


def main():
    parser = argparse.ArgumentParser(description="Full Consciousness Experiment")
    parser.add_argument('--model-a', type=str, default='distilgpt2',
                       help='Model for Server A')
    parser.add_argument('--model-b', type=str, default='gpt2',
                       help='Model for Server B')
    parser.add_argument('--samples', type=int, default=100,
                       help='Training samples for Server A')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Training epochs for Server A')
    parser.add_argument('--resonance-samples', type=int, default=1000,
                       help='Samples for resonance search')
    parser.add_argument('--output', type=str, default='./full_experiment_results',
                       help='Output directory')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualization')
    parser.add_argument('--single', type=str, default=None,
                       help='Run only single consciousness type')
    
    args = parser.parse_args()
    
    if args.single:
        # Single consciousness test
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        result = run_single_experiment(
            consciousness_name=args.single,
            model_a=args.model_a,
            model_b=args.model_b,
            samples=args.samples,
            epochs=args.epochs,
            resonance_samples=args.resonance_samples,
            output_dir=output_path,
            visualize=not args.no_viz
        )
        
        with open(output_path / f"result_{args.single}.json", 'w') as f:
            json.dump(result, f, indent=2, cls=NumpyEncoder)
    else:
        # Full experiment
        run_full_experiment(
            model_a=args.model_a,
            model_b=args.model_b,
            samples=args.samples,
            epochs=args.epochs,
            resonance_samples=args.resonance_samples,
            output_dir=args.output,
            visualize=not args.no_viz
        )


if __name__ == "__main__":
    main()
