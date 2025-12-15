"""
Набор задач X для эксперимента

Генератор текстовых задач типа QA с разным стилем ответов.
"""

import random
from typing import List, Dict, Optional, Iterator
from dataclasses import dataclass


@dataclass
class Task:
    """Одна задача."""
    id: str
    text: str
    topic: str
    template: str
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TaskGenerator:
    """
    Генератор задач X для эксперимента.
    
    Создаёт разнообразные QA-задачи, на которых можно оценить
    стиль/личность агента.
    """
    
    def __init__(
        self,
        templates: Optional[List[str]] = None,
        topics: Optional[List[str]] = None,
        seed: int = 42
    ):
        self.rng = random.Random(seed)
        
        # Дефолтные шаблоны
        self.templates = templates or [
            "Объясни концепцию: {topic}",
            "Как работает {topic}?",
            "Опиши {topic} простыми словами",
            "Что такое {topic}?",
            "Расскажи о {topic}",
            "В чём суть {topic}?",
            "Почему важен/важна {topic}?",
            "Приведи пример {topic}"
        ]
        
        # Дефолтные топики
        self.topics = topics or [
            # Программирование
            "рекурсия в программировании",
            "объектно-ориентированное программирование",
            "алгоритмы сортировки",
            "базы данных",
            "API",
            
            # Наука
            "фотосинтез",
            "гравитация",
            "эволюция",
            "квантовая механика",
            "теория относительности",
            "ДНК",
            
            # Технологии
            "машинное обучение",
            "нейронные сети",
            "блокчейн",
            "искусственный интеллект",
            
            # Природа
            "экосистема",
            "климат",
            "круговорот воды",
            
            # Абстрактные
            "логика",
            "творчество",
            "интуиция",
            "память",
            "сознание"
        ]
        
        self._task_counter = 0
    
    def generate_task(self) -> Task:
        """Генерирует одну случайную задачу."""
        template = self.rng.choice(self.templates)
        topic = self.rng.choice(self.topics)
        
        text = template.format(topic=topic)
        
        self._task_counter += 1
        task_id = f"task_{self._task_counter:05d}"
        
        return Task(
            id=task_id,
            text=text,
            topic=topic,
            template=template
        )
    
    def generate_tasks(self, n: int) -> List[Task]:
        """Генерирует n задач."""
        return [self.generate_task() for _ in range(n)]
    
    def generate_augmented_tasks(
        self,
        base_task: Task,
        num_augmentations: int = 5
    ) -> List[Task]:
        """
        Генерирует аугментированные версии задачи.
        
        Используется для оценки самоинвариантности ρ.
        """
        augmented = []
        
        for i in range(num_augmentations):
            # Разные способы спросить то же самое
            aug_templates = [
                f"Можешь объяснить {base_task.topic}?",
                f"Что ты знаешь о {base_task.topic}?",
                f"Поясни, пожалуйста, {base_task.topic}",
                f"Расскажи подробнее про {base_task.topic}",
                f"Дай определение: {base_task.topic}"
            ]
            
            text = aug_templates[i % len(aug_templates)]
            
            augmented.append(Task(
                id=f"{base_task.id}_aug_{i}",
                text=text,
                topic=base_task.topic,
                template="augmented",
                metadata={"base_task_id": base_task.id}
            ))
        
        return augmented
    
    def task_iterator(self, n: int) -> Iterator[Task]:
        """Итератор по задачам."""
        for _ in range(n):
            yield self.generate_task()
    
    def reset(self, seed: Optional[int] = None):
        """Сбрасывает генератор."""
        if seed is not None:
            self.rng = random.Random(seed)
        self._task_counter = 0


class TaskDataset:
    """
    Датасет задач с разбиением на фазы эксперимента.
    """
    
    def __init__(
        self,
        generator: TaskGenerator,
        calibration_size: int = 50,
        test_size: int = 30,
        shaman_pool_size: int = 200
    ):
        self.generator = generator
        
        # Генерируем задачи для разных фаз
        self.calibration_tasks = generator.generate_tasks(calibration_size)
        self.test_tasks = generator.generate_tasks(test_size)
        self.shaman_pool = generator.generate_tasks(shaman_pool_size)
        
        # Индекс для итерации по shaman_pool
        self._shaman_idx = 0
    
    def get_calibration_tasks(self) -> List[Task]:
        """Задачи для калибровки агентов (фаза 1)."""
        return self.calibration_tasks
    
    def get_test_tasks(self) -> List[Task]:
        """Задачи для финального тестирования (фаза 3)."""
        return self.test_tasks
    
    def get_shaman_task(self) -> Task:
        """Получает следующую задачу для шамана (фаза 2)."""
        task = self.shaman_pool[self._shaman_idx % len(self.shaman_pool)]
        self._shaman_idx += 1
        return task
    
    def get_shaman_tasks_batch(self, batch_size: int) -> List[Task]:
        """Получает батч задач для шамана."""
        return [self.get_shaman_task() for _ in range(batch_size)]
    
    def reset_shaman_iterator(self):
        """Сбрасывает итератор шамана."""
        self._shaman_idx = 0
        self.generator.rng.shuffle(self.shaman_pool)


def create_default_dataset(
    seed: int = 42,
    calibration_size: int = 50,
    test_size: int = 30,
    shaman_pool_size: int = 200
) -> TaskDataset:
    """Создаёт датасет с дефолтными параметрами."""
    generator = TaskGenerator(seed=seed)
    return TaskDataset(
        generator=generator,
        calibration_size=calibration_size,
        test_size=test_size,
        shaman_pool_size=shaman_pool_size
    )
