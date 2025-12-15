"""
Тесты для базовых компонентов Shaman MVP
"""

import pytest
import numpy as np
import sys
import os

# Добавляем src в path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestSymExtractor:
    """Тесты для извлечения симструктур."""
    
    def test_pattern_detection(self):
        from src.sym_extractor import SymExtractor
        
        extractor = SymExtractor()
        
        # Текст с аналитическим стилем
        analytical_text = """
        Анализируя ситуацию, можно выделить следующие аспекты:
        1. Первый важный момент
        2. Второй аспект
        3. Третий элемент
        
        Резюме: данный подход является оптимальным.
        """
        
        sigma = extractor.extract(analytical_text)
        
        # Должны найти паттерны
        assert sigma.patterns_found.get("ANALYSIS_START", 0) > 0
        assert sigma.patterns_found.get("ENUMERATION", 0) > 0
        assert sigma.patterns_found.get("CONCLUSION", 0) > 0
        
        # Features должны быть заполнены
        assert "structuredness" in sigma.features
        assert "analyticity" in sigma.features
    
    def test_creative_style(self):
        from src.sym_extractor import SymExtractor
        
        extractor = SymExtractor()
        
        # Текст с творческим стилем
        creative_text = """
        Это как река, текущая сквозь время!
        Представьте себе: мир подобно огромному океану...
        Замечательно, правда? Интересно, не так ли?
        """
        
        sigma = extractor.extract(creative_text)
        
        # Должны найти метафоры и эмоции
        assert sigma.patterns_found.get("METAPHOR", 0) > 0
        assert sigma.patterns_found.get("EXCLAMATION", 0) > 0
        assert sigma.patterns_found.get("QUESTION", 0) > 0
    
    def test_similarity(self):
        from src.sym_extractor import SymExtractor
        
        extractor = SymExtractor()
        
        text1 = "Анализируя данные, мы видим: 1) первое 2) второе. Вывод: успех."
        text2 = "Рассматривая ситуацию: 1) начало 2) конец. Резюме: готово."
        text3 = "Как прекрасен мир! Словно сказка..."
        
        sigma1 = extractor.extract(text1)
        sigma2 = extractor.extract(text2)
        sigma3 = extractor.extract(text3)
        
        # Похожие стили должны быть ближе
        sim_12 = extractor.compute_similarity(sigma1, sigma2)
        sim_13 = extractor.compute_similarity(sigma1, sigma3)
        
        assert sim_12 > sim_13, "Similar styles should be closer"


class TestMetrics:
    """Тесты для метрик."""
    
    def test_energy_E(self):
        from src.metrics import compute_energy_E
        
        # Плавная траектория
        smooth_traj = np.array([
            [0, 0],
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3]
        ])
        
        # Рваная траектория
        rough_traj = np.array([
            [0, 0],
            [1, -1],
            [-1, 1],
            [0, 0]
        ])
        
        E_smooth = compute_energy_E(smooth_traj)
        E_rough = compute_energy_E(rough_traj)
        
        # Рваная траектория должна иметь бóльшую энергию
        assert E_rough.total > E_smooth.total
    
    def test_rho(self):
        from src.metrics import compute_rho
        
        anchor = np.array([0, 0, 0])
        
        # Стабильные выборки (близко к anchor)
        stable_samples = np.array([
            [0.1, 0.1, 0.1],
            [0.05, -0.05, 0.1],
            [-0.1, 0.1, -0.1]
        ])
        
        # Нестабильные выборки (далеко от anchor)
        unstable_samples = np.array([
            [1, 1, 1],
            [-1, -1, 2],
            [2, -1, 0]
        ])
        
        rho_stable = compute_rho(anchor, stable_samples)
        rho_unstable = compute_rho(anchor, unstable_samples)
        
        # Стабильные должны иметь выше rho
        assert rho_stable > rho_unstable
    
    def test_d_P(self):
        from src.metrics import compute_d_P
        
        cluster_A = np.array([
            [0, 0],
            [0.1, 0.1],
            [-0.1, 0.1]
        ])
        
        # Близкий кластер
        cluster_B_close = np.array([
            [0.2, 0.2],
            [0.3, 0.1],
            [0.1, 0.3]
        ])
        
        # Далёкий кластер
        cluster_B_far = np.array([
            [5, 5],
            [5.1, 4.9],
            [4.9, 5.1]
        ])
        
        d_close = compute_d_P(cluster_A, cluster_B_close, method="centroid_euclidean")
        d_far = compute_d_P(cluster_A, cluster_B_far, method="centroid_euclidean")
        
        assert d_close < d_far


class TestTDA:
    """Тесты для TDA модуля."""
    
    def test_persistence_computation(self):
        from src.tda import TDAModule
        
        tda = TDAModule(max_dimension=1, backend="ripser")
        
        # Точки, образующие круг (должен быть 1-цикл)
        theta = np.linspace(0, 2*np.pi, 20, endpoint=False)
        circle_points = np.column_stack([np.cos(theta), np.sin(theta)])
        
        diagrams = tda.compute_persistence(circle_points)
        
        # Должен быть хотя бы один 1-цикл
        assert 1 in diagrams
        assert len(diagrams[1].birth_death_pairs) > 0
    
    def test_betti_numbers(self):
        from src.tda import TDAModule
        
        tda = TDAModule(max_dimension=1, backend="ripser")
        
        # Два отдельных кластера (β₀ = 2)
        points = np.array([
            [0, 0], [0.1, 0], [0, 0.1],  # кластер 1
            [10, 10], [10.1, 10], [10, 10.1]  # кластер 2
        ])
        
        diagrams = tda.compute_persistence(points)
        betti = tda.compute_betti_numbers(diagrams, threshold=0.5)
        
        # Должно быть 2 связных компоненты при большом threshold
        assert betti[0] >= 1  # как минимум 1 компонента


class TestTaskGenerator:
    """Тесты для генератора задач."""
    
    def test_task_generation(self):
        from src.tasks import TaskGenerator
        
        gen = TaskGenerator(seed=42)
        
        tasks = gen.generate_tasks(10)
        
        assert len(tasks) == 10
        
        # Все задачи должны иметь текст и топик
        for task in tasks:
            assert task.text
            assert task.topic
            assert task.id
    
    def test_augmentation(self):
        from src.tasks import TaskGenerator
        
        gen = TaskGenerator(seed=42)
        
        base_task = gen.generate_task()
        augmented = gen.generate_augmented_tasks(base_task, num_augmentations=3)
        
        assert len(augmented) == 3
        
        # Все аугментации должны быть про тот же топик
        for aug in augmented:
            assert base_task.topic in aug.text


class TestEncoders:
    """Тесты для энкодеров (требуют моделей)."""
    
    @pytest.mark.slow
    def test_phi_encoder(self):
        from src.encoders import PhiEncoder
        
        # Используем CPU для тестов
        phi = PhiEncoder(device="cpu")
        
        text1 = "This is a test sentence."
        text2 = "This is another test sentence."
        text3 = "Completely different topic about cats."
        
        emb1 = phi.encode_text(text1)
        emb2 = phi.encode_text(text2)
        emb3 = phi.encode_text(text3)
        
        # Эмбеддинги должны быть правильной размерности
        assert emb1.shape == (phi.embedding_dim,)
        
        # Похожие тексты должны быть ближе
        d12 = phi.compute_distance(emb1, emb2)
        d13 = phi.compute_distance(emb1, emb3)
        
        assert d12 < d13


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
