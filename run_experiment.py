#!/usr/bin/env python3
"""
Shaman MVP - Главный скрипт запуска эксперимента

Запуск:
    python run_experiment.py
    python run_experiment.py --config config/custom.yaml
    python run_experiment.py --quick  # быстрый тест
"""

import argparse
import sys
import os

# Добавляем src в path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.experiment import Experiment, ExperimentConfig, run_experiment


def parse_args():
    parser = argparse.ArgumentParser(
        description="Shaman MVP - Resonance Transfer Experiment"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode (reduced iterations)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of experiment runs"
    )
    
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for logs and results"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*60)
    print("SHAMAN MVP - Resonance Transfer Experiment")
    print("="*60)
    
    # Создаём конфигурацию
    if args.quick:
        # Быстрый режим для тестирования
        config = ExperimentConfig(
            name="shaman_quick_test",
            seed=args.seed,
            device=args.device,
            calibration_samples=10,
            shaman_iterations=20,
            shaman_batch_size=4,
            test_samples=10,
            num_runs=1,
            log_dir=args.log_dir
        )
        print("\n[QUICK MODE] Reduced iterations for testing")
    else:
        config = ExperimentConfig(
            name="shaman_mvp",
            seed=args.seed,
            device=args.device,
            calibration_samples=50,
            shaman_iterations=100,
            shaman_batch_size=8,
            test_samples=30,
            num_runs=args.runs,
            log_dir=args.log_dir
        )
    
    print(f"\nConfiguration:")
    print(f"  Device: {config.device}")
    print(f"  Seed: {config.seed}")
    print(f"  Calibration samples: {config.calibration_samples}")
    print(f"  Shaman iterations: {config.shaman_iterations}")
    print(f"  Test samples: {config.test_samples}")
    print(f"  Number of runs: {config.num_runs}")
    
    # Запускаем эксперимент
    try:
        results = run_experiment(config)
        
        # Возвращаем код в зависимости от результата
        if results.is_successful():
            print("\n✓ Experiment completed successfully!")
            return 0
        else:
            print("\n⚠ Experiment completed but results inconclusive")
            return 1
            
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")
        return 130
    except Exception as e:
        print(f"\n\nExperiment failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
