"""
Искусственный Шаман B_ш - модуль резонансного поиска

Шаман - это надстройка над агентом B, которая:
1. Генерирует задачи для B (управляет exploration)
2. Оценивает метрики: E_B, ρ_B, cycle_persistence, σ
3. Оптимизирует свою policy через REINFORCE для максимизации J
4. Ищет устойчивые смысловые инварианты в S
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

from .encoders import PhiEncoder, LatentTrajectory
from .tda import TDAModule
from .sym_extractor import SymExtractor, SymStructure
from .metrics import (
    compute_energy_E,
    compute_rho,
    compute_shaman_reward,
    compute_d_P,
    MetricsTracker
)


@dataclass
class ShamanConfig:
    """Конфигурация шамана."""
    # Веса reward'а J
    w_energy: float = -1.0
    w_rho: float = 2.0
    w_cycle: float = 1.0
    w_rarity: float = 0.5
    w_struct: float = 0.1
    
    # Обучение
    learning_rate: float = 1e-4
    batch_size: int = 8
    
    # Exploration
    num_augmentations: int = 5
    exploration_noise: float = 0.3
    
    # Память
    memory_size: int = 1000
    
    # TDA
    tda_window_size: int = 50


@dataclass
class ShamanMemory:
    """Память шамана - хранит историю взаимодействий."""
    max_size: int = 1000
    
    # Очереди для разных типов данных
    embeddings: deque = field(default_factory=lambda: deque(maxlen=1000))
    rewards: deque = field(default_factory=lambda: deque(maxlen=1000))
    episodes: deque = field(default_factory=lambda: deque(maxlen=1000))
    sym_structures: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def __post_init__(self):
        self.embeddings = deque(maxlen=self.max_size)
        self.rewards = deque(maxlen=self.max_size)
        self.episodes = deque(maxlen=self.max_size)
        self.sym_structures = deque(maxlen=self.max_size)
    
    def add(
        self,
        embedding: np.ndarray,
        reward: float,
        episode: Dict,
        sym_structure: Optional[SymStructure] = None
    ):
        """Добавляет запись в память."""
        self.embeddings.append(embedding)
        self.rewards.append(reward)
        self.episodes.append(episode)
        if sym_structure:
            self.sym_structures.append(sym_structure)
    
    def get_recent_embeddings(self, n: int) -> np.ndarray:
        """Возвращает последние n эмбеддингов."""
        recent = list(self.embeddings)[-n:]
        if not recent:
            return np.zeros((0, 384))  # default dim
        return np.stack(recent)
    
    def get_stats(self) -> Dict[str, float]:
        """Возвращает статистику по памяти."""
        rewards = list(self.rewards)
        if not rewards:
            return {"mean_reward": 0, "size": 0}
        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "max_reward": float(np.max(rewards)),
            "size": len(rewards)
        }


class ShamanPolicy(nn.Module):
    """
    Простая policy network для шамана.
    
    Принимает агрегированное состояние и выдаёт параметры
    для выбора/модификации задач.
    """
    
    def __init__(
        self,
        state_dim: int = 128,
        hidden_dim: int = 64,
        num_task_params: int = 16
    ):
        super().__init__()
        
        self.state_dim = state_dim
        
        # Энкодер состояния
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Голова для параметров задач (mean, log_std)
        self.task_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_task_params * 2)  # mean + log_std
        )
        
        self.num_task_params = num_task_params
    
    def forward(
        self,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            state: [batch, state_dim]
            
        Returns:
            mean, log_std: [batch, num_task_params]
        """
        h = self.state_encoder(state)
        params = self.task_head(h)
        
        mean = params[:, :self.num_task_params]
        log_std = params[:, self.num_task_params:]
        log_std = torch.clamp(log_std, -5, 2)
        
        return mean, log_std
    
    def sample(
        self,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Сэмплирует параметры задачи.
        
        Returns:
            action: [batch, num_task_params]
            log_prob: [batch]
        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        # Reparameterization trick
        eps = torch.randn_like(mean)
        action = mean + std * eps
        
        # Log probability
        log_prob = -0.5 * (
            ((action - mean) / (std + 1e-8)) ** 2 +
            2 * log_std +
            np.log(2 * np.pi)
        ).sum(dim=-1)
        
        return action, log_prob


class Shaman:
    """
    Искусственный Шаман B_ш.
    
    Управляет exploration агента B в пространстве смыслов S,
    ищет устойчивые инварианты через резонансный поиск.
    """
    
    def __init__(
        self,
        config: ShamanConfig,
        phi_encoder: PhiEncoder,
        tda_module: TDAModule,
        sym_extractor: SymExtractor,
        device: str = "cuda"
    ):
        self.config = config
        self.phi = phi_encoder
        self.tda = tda_module
        self.sym_extractor = sym_extractor
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Policy network
        self.policy = ShamanPolicy(
            state_dim=128,
            hidden_dim=64,
            num_task_params=16
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=config.learning_rate
        )
        
        # Память
        self.memory = ShamanMemory(max_size=config.memory_size)
        
        # Трекер метрик
        self.metrics_tracker = MetricsTracker()
        
        # Траектория в S
        self.trajectory = LatentTrajectory(embedding_dim=self.phi.embedding_dim)
        
        # Anchor (центр s_я) - устанавливается при калибровке
        self.anchor_embedding: Optional[np.ndarray] = None
    
    def build_state(self) -> torch.Tensor:
        """
        Строит состояние для policy из текущей памяти.
        
        Агрегирует недавние эмбеддинги и метрики.
        """
        # Получаем недавние эмбеддинги
        recent_emb = self.memory.get_recent_embeddings(10)
        
        if len(recent_emb) == 0:
            # Инициализация нулями
            return torch.zeros(1, 128, device=self.device)
        
        # Агрегация: mean + std последних эмбеддингов
        mean_emb = np.mean(recent_emb, axis=0)
        std_emb = np.std(recent_emb, axis=0) if len(recent_emb) > 1 else np.zeros_like(mean_emb)
        
        # Конкатенация и сжатие до state_dim
        aggregated = np.concatenate([mean_emb[:64], std_emb[:64]])
        
        state = torch.tensor(aggregated, dtype=torch.float32, device=self.device)
        return state.unsqueeze(0)  # [1, 128]
    
    def select_task_modification(self) -> Dict[str, float]:
        """
        Выбирает параметры модификации задачи.
        
        Возвращает словарь параметров, которые влияют на
        формулировку задачи для агента B.
        """
        state = self.build_state()
        
        with torch.no_grad():
            action, _ = self.policy.sample(state)
        
        action = action.squeeze(0).cpu().numpy()
        
        # Sigmoid function (NumPy не имеет встроенной)
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        
        # Интерпретируем action как параметры модификации
        return {
            "temperature_mod": float(np.tanh(action[0]) * 0.3),  # [-0.3, 0.3]
            "prompt_style": int(np.argmax(action[1:5])),  # 0-3
            "detail_level": float(sigmoid(action[5])),  # [0, 1]
            "formality": float(sigmoid(action[6])),  # [0, 1]
        }
    
    def evaluate_episode(
        self,
        task_text: str,
        response_text: str,
        agent_embedding: np.ndarray
    ) -> Dict[str, float]:
        """
        Оценивает эпизод и вычисляет все метрики.
        
        Args:
            task_text: текст задачи
            response_text: ответ агента B
            agent_embedding: эмбеддинг ответа от PhiEncoder
            
        Returns:
            словарь с метриками
        """
        # 1. Добавляем точку в траекторию
        self.trajectory.add_point(agent_embedding)
        
        # 2. Энергия E_B
        trajectory_points = self.trajectory.get_points_array()
        E_components = compute_energy_E(
            trajectory_points,
            response_lengths=[len(response_text.split())]
        )
        E_B = E_components.total
        
        # 3. Самоинвариантность ρ_B
        if self.anchor_embedding is not None:
            recent = self.memory.get_recent_embeddings(self.config.num_augmentations)
            if len(recent) > 0:
                rho_B = compute_rho(self.anchor_embedding, recent)
            else:
                rho_B = 1.0
        else:
            rho_B = 1.0
        
        # 4. Циклы через TDA
        if len(trajectory_points) >= 5:
            tda_results = self.tda.analyze_trajectory(
                trajectory_points,
                window_size=self.config.tda_window_size
            )
            cycle_persistence = tda_results['total_persistence_1']
        else:
            cycle_persistence = 0.0
        
        # 5. Симструктура σ
        sigma = self.sym_extractor.extract(response_text)
        rarity = self.sym_extractor.compute_rarity(sigma)
        
        # Структурный score из σ
        struct_score = (
            sigma.features.get('structuredness', 0) * 2 +
            sigma.features.get('analyticity', 0) +
            len(sigma.patterns_found) * 0.1
        )
        
        # 6. Вычисляем reward J
        reward = compute_shaman_reward(
            E_B=E_B,
            rho_B=rho_B,
            cycle_persistence=cycle_persistence,
            rarity=rarity,
            struct_score=struct_score,
            weights={
                "w_energy": self.config.w_energy,
                "w_rho": self.config.w_rho,
                "w_cycle": self.config.w_cycle,
                "w_rarity": self.config.w_rarity,
                "w_struct": self.config.w_struct
            }
        )
        
        metrics = {
            "E_B": E_B,
            "rho_B": rho_B,
            "cycle_persistence": cycle_persistence,
            "rarity": rarity,
            "struct_score": struct_score,
            "reward": reward.total
        }
        
        # Сохраняем в память
        self.memory.add(
            embedding=agent_embedding,
            reward=reward.total,
            episode={"task": task_text, "response": response_text},
            sym_structure=sigma
        )
        
        # Логируем
        self.metrics_tracker.log(metrics)
        
        return metrics
    
    def update_policy(self, batch_rewards: List[float], batch_log_probs: List[torch.Tensor]):
        """
        Обновляет policy через REINFORCE.
        
        Args:
            batch_rewards: список rewards за батч
            batch_log_probs: список log_prob за батч
        """
        if len(batch_rewards) == 0:
            return
        
        rewards = torch.tensor(batch_rewards, dtype=torch.float32, device=self.device)
        log_probs = torch.stack(batch_log_probs)
        
        # Нормализация rewards (baseline)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Policy gradient loss
        loss = -(log_probs * rewards).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
    
    def set_anchor(self, anchor_embedding: np.ndarray):
        """Устанавливает anchor (центр s_я) для оценки ρ."""
        self.anchor_embedding = anchor_embedding.copy()
    
    def reset_trajectory(self):
        """Сбрасывает траекторию."""
        self.trajectory.clear()
    
    def get_summary(self) -> Dict:
        """Возвращает сводку состояния шамана."""
        return {
            "memory_stats": self.memory.get_stats(),
            "metrics_summary": self.metrics_tracker.get_summary(),
            "trajectory_length": len(self.trajectory.points),
            "has_anchor": self.anchor_embedding is not None
        }


def create_shaman(
    config: Optional[ShamanConfig] = None,
    device: str = "cuda"
) -> Shaman:
    """Создаёт шамана с дефолтными компонентами."""
    if config is None:
        config = ShamanConfig()
    
    phi = PhiEncoder(device=device)
    tda = TDAModule(max_dimension=1, backend="ripser")
    sym = SymExtractor()
    
    return Shaman(
        config=config,
        phi_encoder=phi,
        tda_module=tda,
        sym_extractor=sym,
        device=device
    )
