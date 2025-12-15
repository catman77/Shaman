"""
Протокол эксперимента - основной цикл

Реализует три фазы:
1. Калибровка - формирование s_A и s_B до изоляции
2. Изоляция + работа шамана - резонансный поиск без доступа к A
3. Тестирование - измерение d_P(s_A, s_B) после работы шамана
"""

import os
import json
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from tqdm import tqdm

from .agents import Agent, AgentConfig, AgentPair, create_default_agents
from .encoders import PhiEncoder
from .shaman import Shaman, ShamanConfig, create_shaman
from .tasks import TaskGenerator, TaskDataset, create_default_dataset
from .metrics import compute_d_P, MetricsTracker
from .sym_extractor import SymExtractor


@dataclass
class ExperimentConfig:
    """Конфигурация эксперимента."""
    name: str = "shaman_mvp"
    seed: int = 42
    device: str = "cuda"
    
    # Фаза 1: калибровка
    calibration_samples: int = 50
    
    # Фаза 2: работа шамана
    shaman_iterations: int = 100
    shaman_batch_size: int = 8
    
    # Фаза 3: тестирование
    test_samples: int = 30
    
    # Число повторов эксперимента
    num_runs: int = 3
    
    # Логирование
    log_dir: str = "logs"
    save_checkpoints: bool = True


@dataclass
class ExperimentResults:
    """Результаты эксперимента."""
    config: Dict
    
    # Метрики до и после
    d_P_before: float
    d_P_after: float
    d_P_reduction: float
    d_P_reduction_percent: float
    
    # Статистика по запускам
    d_P_before_runs: List[float]
    d_P_after_runs: List[float]
    
    # Метрики шамана
    shaman_metrics: Dict
    
    # Временные метки
    start_time: str
    end_time: str
    duration_seconds: float
    
    def is_successful(self, threshold: float = 0.1) -> bool:
        """Проверяет, считается ли эксперимент успешным."""
        return self.d_P_reduction_percent > threshold * 100


class Experiment:
    """
    Основной класс эксперимента.
    
    Координирует все фазы и компоненты.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = config.device
        
        # Устанавливаем seed
        np.random.seed(config.seed)
        
        # Создаём директорию для логов
        os.makedirs(config.log_dir, exist_ok=True)
        
        # Инициализируем компоненты
        print("Initializing experiment components...")
        
        # Энкодер для латентного пространства
        self.phi = PhiEncoder(device=self.device)
        
        # Экстрактор симструктур
        self.sym_extractor = SymExtractor()
        
        # Агенты будут созданы в run()
        self.agents: Optional[AgentPair] = None
        
        # Шаман будет создан в run()
        self.shaman: Optional[Shaman] = None
        
        # Датасет задач
        self.dataset: Optional[TaskDataset] = None
        
        # Трекер метрик
        self.metrics = MetricsTracker()
        
        # Хранилище результатов
        self.results: Optional[ExperimentResults] = None
    
    def _create_agents(self):
        """Создаёт пару агентов."""
        print("Creating agents A and B...")
        self.agents = create_default_agents(device=self.device)
    
    def _create_shaman(self):
        """Создаёт шамана."""
        print("Creating Shaman B_ш...")
        shaman_config = ShamanConfig(
            learning_rate=1e-4,
            batch_size=self.config.shaman_batch_size
        )
        self.shaman = create_shaman(config=shaman_config, device=self.device)
    
    def _create_dataset(self):
        """Создаёт датасет задач."""
        print("Creating task dataset...")
        self.dataset = create_default_dataset(
            seed=self.config.seed,
            calibration_size=self.config.calibration_samples,
            test_size=self.config.test_samples,
            shaman_pool_size=self.config.shaman_iterations * self.config.shaman_batch_size
        )
    
    def phase1_calibration(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Фаза 1: Калибровка агентов.
        
        Собираем ответы A и B на калибровочные задачи,
        формируем кластеры s_A и s_B.
        
        Returns:
            cluster_A: эмбеддинги ответов A
            cluster_B: эмбеддинги ответов B
        """
        print("\n" + "="*50)
        print("PHASE 1: Calibration")
        print("="*50)
        
        tasks = self.dataset.get_calibration_tasks()
        
        embeddings_A = []
        embeddings_B = []
        
        for task in tqdm(tasks, desc="Calibrating agents"):
            # Генерация от обоих агентов
            result_a = self.agents.generate_a(task.text)
            result_b = self.agents.generate_b(task.text)
            
            # Эмбеддинги через PhiEncoder
            emb_a = self.phi.encode_text(result_a.response)
            emb_b = self.phi.encode_text(result_b.response)
            
            embeddings_A.append(emb_a)
            embeddings_B.append(emb_b)
        
        cluster_A = np.stack(embeddings_A)
        cluster_B = np.stack(embeddings_B)
        
        # Вычисляем начальное d_P
        d_P_initial = compute_d_P(cluster_A, cluster_B)
        
        print(f"\nCalibration complete:")
        print(f"  Cluster A size: {len(cluster_A)}")
        print(f"  Cluster B size: {len(cluster_B)}")
        print(f"  Initial d_P(s_A, s_B): {d_P_initial:.4f}")
        
        self.metrics.log({"d_P_initial": d_P_initial})
        
        return cluster_A, cluster_B
    
    def phase2_shaman_work(self, cluster_A: np.ndarray):
        """
        Фаза 2: Работа шамана (изолированно от A).
        
        Шаман работает только с агентом B, не имея доступа к A.
        Цель: направить B к формированию инвариантов, близких к s_A.
        
        Args:
            cluster_A: кластер A (используется ТОЛЬКО для установки anchor,
                      шаман не видит конкретных ответов A)
        """
        print("\n" + "="*50)
        print("PHASE 2: Shaman Resonance Search")
        print("="*50)
        print("Note: Shaman has NO access to Agent A responses")
        print("Only anchor (centroid of s_A) is used for ρ computation")
        
        # Устанавливаем anchor как центроид s_A
        # Это единственная "утечка" информации - общее направление
        anchor = np.mean(cluster_A, axis=0)
        self.shaman.set_anchor(anchor)
        
        # Сбрасываем итератор задач
        self.dataset.reset_shaman_iterator()
        
        batch_rewards = []
        batch_log_probs = []
        
        for iteration in tqdm(range(self.config.shaman_iterations), desc="Shaman working"):
            # Получаем задачу
            task = self.dataset.get_shaman_task()
            
            # Шаман выбирает модификацию (не используется напрямую в MVP,
            # но показывает механизм)
            task_mods = self.shaman.select_task_modification()
            
            # Агент B генерирует ответ
            result_b = self.agents.generate_b(task.text)
            
            # Эмбеддинг ответа
            emb_b = self.phi.encode_text(result_b.response)
            
            # Шаман оценивает эпизод
            metrics = self.shaman.evaluate_episode(
                task_text=task.text,
                response_text=result_b.response,
                agent_embedding=emb_b
            )
            
            batch_rewards.append(metrics["reward"])
            
            # Получаем log_prob для policy update
            state = self.shaman.build_state()
            _, log_prob = self.shaman.policy.sample(state)
            batch_log_probs.append(log_prob.squeeze())
            
            # Обновляем policy каждые batch_size шагов
            if (iteration + 1) % self.config.shaman_batch_size == 0:
                self.shaman.update_policy(batch_rewards, batch_log_probs)
                batch_rewards = []
                batch_log_probs = []
            
            # Логируем каждые 10 итераций
            if (iteration + 1) % 10 == 0:
                summary = self.shaman.get_summary()
                avg_reward = summary["memory_stats"].get("mean_reward", 0)
                self.metrics.log({
                    "shaman_iteration": iteration + 1,
                    "shaman_avg_reward": avg_reward
                })
        
        print(f"\nShaman work complete:")
        summary = self.shaman.get_summary()
        print(f"  Total episodes: {summary['memory_stats']['size']}")
        print(f"  Mean reward: {summary['memory_stats'].get('mean_reward', 0):.4f}")
    
    def phase3_testing(self, cluster_A: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Фаза 3: Тестирование переноса.
        
        Измеряем d_P(s_A, s_B) после работы шамана.
        
        Args:
            cluster_A: эталонный кластер A
            
        Returns:
            d_P_final: финальное расстояние
            cluster_B_final: новый кластер B
        """
        print("\n" + "="*50)
        print("PHASE 3: Testing Transfer")
        print("="*50)
        
        tasks = self.dataset.get_test_tasks()
        
        embeddings_B = []
        
        for task in tqdm(tasks, desc="Testing agent B"):
            result_b = self.agents.generate_b(task.text)
            emb_b = self.phi.encode_text(result_b.response)
            embeddings_B.append(emb_b)
        
        cluster_B_final = np.stack(embeddings_B)
        
        # Вычисляем финальное d_P
        d_P_final = compute_d_P(cluster_A, cluster_B_final)
        
        print(f"\nTesting complete:")
        print(f"  Final d_P(s_A, s_B): {d_P_final:.4f}")
        
        return d_P_final, cluster_B_final
    
    def run_single(self) -> Tuple[float, float]:
        """
        Запускает один прогон эксперимента.
        
        Returns:
            d_P_before, d_P_after
        """
        # Пересоздаём компоненты для чистого запуска
        self._create_agents()
        self._create_shaman()
        self._create_dataset()
        
        # Фаза 1: калибровка
        cluster_A, cluster_B = self.phase1_calibration()
        d_P_before = compute_d_P(cluster_A, cluster_B)
        
        # Фаза 2: работа шамана
        self.phase2_shaman_work(cluster_A)
        
        # Фаза 3: тестирование
        d_P_after, _ = self.phase3_testing(cluster_A)
        
        return d_P_before, d_P_after
    
    def run(self) -> ExperimentResults:
        """
        Запускает полный эксперимент с несколькими прогонами.
        
        Returns:
            ExperimentResults со статистикой
        """
        start_time = datetime.now()
        
        print("\n" + "#"*60)
        print("# SHAMAN MVP EXPERIMENT")
        print(f"# Runs: {self.config.num_runs}")
        print("#"*60)
        
        d_P_before_runs = []
        d_P_after_runs = []
        
        for run_idx in range(self.config.num_runs):
            print(f"\n{'='*60}")
            print(f"RUN {run_idx + 1}/{self.config.num_runs}")
            print("="*60)
            
            # Меняем seed для каждого прогона
            np.random.seed(self.config.seed + run_idx)
            
            d_P_before, d_P_after = self.run_single()
            
            d_P_before_runs.append(d_P_before)
            d_P_after_runs.append(d_P_after)
            
            print(f"\nRun {run_idx + 1} results:")
            print(f"  d_P before: {d_P_before:.4f}")
            print(f"  d_P after:  {d_P_after:.4f}")
            print(f"  Reduction:  {(d_P_before - d_P_after):.4f} ({(d_P_before - d_P_after) / d_P_before * 100:.1f}%)")
        
        end_time = datetime.now()
        
        # Агрегируем результаты
        mean_before = float(np.mean(d_P_before_runs))
        mean_after = float(np.mean(d_P_after_runs))
        reduction = mean_before - mean_after
        reduction_percent = (reduction / mean_before * 100) if mean_before > 0 else 0
        
        self.results = ExperimentResults(
            config=asdict(self.config),
            d_P_before=mean_before,
            d_P_after=mean_after,
            d_P_reduction=reduction,
            d_P_reduction_percent=reduction_percent,
            d_P_before_runs=d_P_before_runs,
            d_P_after_runs=d_P_after_runs,
            shaman_metrics=self.shaman.get_summary() if self.shaman else {},
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration_seconds=(end_time - start_time).total_seconds()
        )
        
        # Выводим финальные результаты
        self._print_final_results()
        
        # Сохраняем результаты
        self._save_results()
        
        return self.results
    
    def _print_final_results(self):
        """Выводит финальные результаты."""
        print("\n" + "#"*60)
        print("# FINAL RESULTS")
        print("#"*60)
        
        r = self.results
        
        print(f"\nExperiment: {self.config.name}")
        print(f"Duration: {r.duration_seconds:.1f} seconds")
        print(f"Number of runs: {self.config.num_runs}")
        
        print(f"\n{'Metric':<25} {'Mean':>10} {'Std':>10}")
        print("-" * 45)
        print(f"{'d_P before shaman':<25} {r.d_P_before:>10.4f} {np.std(r.d_P_before_runs):>10.4f}")
        print(f"{'d_P after shaman':<25} {r.d_P_after:>10.4f} {np.std(r.d_P_after_runs):>10.4f}")
        print(f"{'Reduction':<25} {r.d_P_reduction:>10.4f}")
        print(f"{'Reduction %':<25} {r.d_P_reduction_percent:>10.1f}%")
        
        print("\n" + "="*45)
        if r.is_successful():
            print("✓ EXPERIMENT SUCCESSFUL: d_P reduced significantly")
        else:
            print("✗ EXPERIMENT INCONCLUSIVE: d_P reduction below threshold")
        print("="*45)
    
    def _save_results(self):
        """Сохраняет результаты в файл."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(
            self.config.log_dir,
            f"results_{self.config.name}_{timestamp}.json"
        )
        
        with open(filename, 'w') as f:
            json.dump(asdict(self.results), f, indent=2, default=str)
        
        print(f"\nResults saved to: {filename}")


def run_experiment(
    config: Optional[ExperimentConfig] = None
) -> ExperimentResults:
    """
    Запускает эксперимент с дефолтной или заданной конфигурацией.
    """
    if config is None:
        config = ExperimentConfig()
    
    experiment = Experiment(config)
    return experiment.run()
