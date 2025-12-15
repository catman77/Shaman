"""
Агенты A и B - LLM-обёртки с разными "личностями"

Для MVP используем DistilGPT2 (~82M параметров) - работает на RTX 3060Ti.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig
)


@dataclass
class AgentConfig:
    """Конфигурация агента."""
    name: str
    model_name: str = "distilgpt2"
    personality_prompt: str = ""
    temperature: float = 0.7
    max_length: int = 256
    top_p: float = 0.9
    top_k: int = 50
    device: str = "cuda"


@dataclass
class GenerationResult:
    """Результат генерации."""
    prompt: str
    response: str
    full_text: str
    logits: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    response_length: int = 0


class Agent:
    """
    LLM-агент с заданной "личностью".
    
    Личность задаётся через system prompt и влияет на стиль ответов.
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Загружаем модель и токенизатор
        print(f"Loading model {config.model_name} for agent {config.name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
        ).to(self.device)
        
        # GPT2 не имеет pad_token, используем eos_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        
        # Конфигурация генерации
        self.generation_config = GenerationConfig(
            max_new_tokens=config.max_length,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
    
    def _build_prompt(self, task: str) -> str:
        """Строит полный промпт с учётом личности."""
        if self.config.personality_prompt:
            return f"{self.config.personality_prompt}\n\nЗадача: {task}\n\nОтвет:"
        return f"Задача: {task}\n\nОтвет:"
    
    @torch.no_grad()
    def generate(
        self,
        task: str,
        return_logits: bool = False,
        return_hidden: bool = False
    ) -> GenerationResult:
        """
        Генерирует ответ на задачу.
        
        Args:
            task: текст задачи
            return_logits: возвращать ли logits
            return_hidden: возвращать ли hidden states
            
        Returns:
            GenerationResult с ответом и опциональными тензорами
        """
        prompt = self._build_prompt(task)
        
        # Токенизация
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        prompt_len = inputs.input_ids.shape[1]
        
        # Генерация
        outputs = self.model.generate(
            **inputs,
            generation_config=self.generation_config,
            output_scores=return_logits,
            output_hidden_states=return_hidden,
            return_dict_in_generate=True
        )
        
        # Декодирование
        full_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        response = full_text[len(prompt):].strip()
        
        # Logits (если запрошены)
        logits = None
        if return_logits and hasattr(outputs, 'scores') and outputs.scores:
            # scores - tuple of tensors [batch, vocab] для каждого шага
            logits = torch.stack(outputs.scores, dim=1)  # [batch, seq, vocab]
        
        # Hidden states (если запрошены)
        hidden = None
        if return_hidden and hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            # Берём последний слой последнего токена
            last_hidden = outputs.hidden_states[-1][-1]  # [batch, hidden_dim]
            hidden = last_hidden
        
        return GenerationResult(
            prompt=prompt,
            response=response,
            full_text=full_text,
            logits=logits,
            hidden_states=hidden,
            response_length=len(response.split())
        )
    
    @torch.no_grad()
    def generate_batch(
        self,
        tasks: List[str],
        return_logits: bool = False
    ) -> List[GenerationResult]:
        """Генерирует ответы на батч задач."""
        # Для простоты - последовательная генерация
        # (батчевая генерация сложнее из-за разной длины)
        results = []
        for task in tasks:
            result = self.generate(task, return_logits=return_logits)
            results.append(result)
        return results
    
    @torch.no_grad()
    def get_response_embedding(
        self,
        text: str
    ) -> torch.Tensor:
        """
        Получает внутреннее представление текста.
        
        Используем среднее по hidden states последнего слоя.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        outputs = self.model(
            **inputs,
            output_hidden_states=True
        )
        
        # Среднее по последовательности последнего слоя
        last_hidden = outputs.hidden_states[-1]  # [batch, seq, hidden]
        embedding = last_hidden.mean(dim=1).squeeze(0)  # [hidden]
        
        return embedding.cpu()
    
    def update_personality(self, new_prompt: str):
        """Обновляет personality prompt."""
        self.config.personality_prompt = new_prompt
    
    def get_config(self) -> Dict:
        """Возвращает текущую конфигурацию."""
        return {
            "name": self.config.name,
            "model_name": self.config.model_name,
            "personality_prompt": self.config.personality_prompt,
            "temperature": self.config.temperature
        }


class AgentPair:
    """
    Пара агентов A (источник) и B (целевой) для эксперимента.
    
    Логически изолированы - не имеют доступа к данным друг друга.
    """
    
    def __init__(
        self,
        config_a: AgentConfig,
        config_b: AgentConfig,
        shared_model: bool = False
    ):
        """
        Args:
            config_a: конфигурация агента A (источник)
            config_b: конфигурация агента B (целевой)
            shared_model: использовать одну модель для обоих (экономия памяти)
        """
        self.shared_model = shared_model
        
        # Создаём агента A
        self.agent_a = Agent(config_a)
        
        if shared_model:
            # Для экономии памяти - общая модель, разные промпты
            self.agent_b = Agent.__new__(Agent)
            self.agent_b.config = config_b
            self.agent_b.device = self.agent_a.device
            self.agent_b.tokenizer = self.agent_a.tokenizer
            self.agent_b.model = self.agent_a.model
            self.agent_b.generation_config = GenerationConfig(
                max_new_tokens=config_b.max_length,
                temperature=config_b.temperature,
                top_p=config_b.top_p,
                top_k=config_b.top_k,
                do_sample=True,
                pad_token_id=self.agent_a.tokenizer.pad_token_id,
                eos_token_id=self.agent_a.tokenizer.eos_token_id
            )
            # Копируем методы
            self.agent_b._build_prompt = lambda task: Agent._build_prompt(self.agent_b, task)
            self.agent_b.generate = lambda *args, **kwargs: Agent.generate(self.agent_b, *args, **kwargs)
            self.agent_b.get_response_embedding = lambda text: Agent.get_response_embedding(self.agent_b, text)
        else:
            # Отдельные модели (больше памяти, но полная изоляция)
            self.agent_b = Agent(config_b)
    
    def generate_a(self, task: str, **kwargs) -> GenerationResult:
        """Генерация от агента A."""
        return self.agent_a.generate(task, **kwargs)
    
    def generate_b(self, task: str, **kwargs) -> GenerationResult:
        """Генерация от агента B."""
        return self.agent_b.generate(task, **kwargs)
    
    def generate_both(self, task: str, **kwargs) -> Tuple[GenerationResult, GenerationResult]:
        """Генерация от обоих агентов для сравнения."""
        result_a = self.generate_a(task, **kwargs)
        result_b = self.generate_b(task, **kwargs)
        return result_a, result_b


def create_default_agents(device: str = "cuda") -> AgentPair:
    """
    Создаёт пару агентов с дефолтными личностями для эксперимента.
    
    Агент A - аналитический стиль (структурированный, логичный)
    Агент B - творческий стиль (образный, свободный)
    """
    config_a = AgentConfig(
        name="Agent_A_Analytical",
        model_name="distilgpt2",
        personality_prompt="""Ты - аналитический помощник. Твой стиль:
- Всегда начинай ответ с "Анализируя ситуацию..."
- Используй нумерованные списки для структурирования
- Заканчивай выводом "Резюме: ..."
- Предпочитай логические аргументы эмоциональным
- Будь кратким и точным""",
        temperature=0.6,
        device=device
    )
    
    config_b = AgentConfig(
        name="Agent_B_Creative",
        model_name="distilgpt2",
        personality_prompt="""Ты - творческий помощник. Твой стиль:
- Используй метафоры и образные сравнения
- Ответы могут быть свободной формы
- Предпочитай интуитивные объяснения
- Будь выразительным и живым
- Можешь использовать восклицания""",
        temperature=0.9,
        device=device
    )
    
    # Используем shared_model для экономии памяти на 3060Ti
    return AgentPair(config_a, config_b, shared_model=True)
