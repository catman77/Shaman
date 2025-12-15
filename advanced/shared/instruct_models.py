# -*- coding: utf-8 -*-
"""
Instruction-Tuned Models Support

Поддержка instruction-tuned моделей для лучшего style transfer:
- TinyLlama-1.1B-Chat
- Qwen1.5-0.5B-Chat
- Qwen1.5-1.8B-Chat
"""

import os
import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# HuggingFace token
HF_TOKEN = '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'
os.environ['HF_TOKEN'] = HF_TOKEN


@dataclass
class InstructModelConfig:
    """Configuration for instruction-tuned model."""
    name: str
    hf_name: str
    hidden_size: int
    num_layers: int
    parameters_m: int  # in millions
    chat_template: str  # 'chatml', 'llama', 'qwen'
    system_prompt: str = "You are a helpful assistant."


# Available instruction-tuned models
INSTRUCT_MODELS: Dict[str, InstructModelConfig] = {
    'tinyllama': InstructModelConfig(
        name='TinyLlama-1.1B-Chat',
        hf_name='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        hidden_size=2048,
        num_layers=22,
        parameters_m=1100,
        chat_template='llama',
        system_prompt="You are a helpful assistant that solves math problems."
    ),
    'qwen-0.5b': InstructModelConfig(
        name='Qwen1.5-0.5B-Chat',
        hf_name='Qwen/Qwen1.5-0.5B-Chat',
        hidden_size=1024,
        num_layers=24,
        parameters_m=464,
        chat_template='qwen',
        system_prompt="You are a helpful assistant that solves math problems."
    ),
    'qwen-1.8b': InstructModelConfig(
        name='Qwen1.5-1.8B-Chat',
        hf_name='Qwen/Qwen1.5-1.8B-Chat',
        hidden_size=2048,
        num_layers=24,
        parameters_m=1800,
        chat_template='qwen',
        system_prompt="You are a helpful assistant that solves math problems."
    ),
}


class InstructModel:
    """
    Wrapper for instruction-tuned models.
    
    Поддерживает:
    - Генерацию с системным промптом
    - Форматирование в стиле chat
    - Извлечение активаций
    """
    
    def __init__(
        self, 
        model_key: str = 'qwen-0.5b',
        device: str = 'auto',
        dtype: torch.dtype = torch.float16,
        load_in_8bit: bool = False
    ):
        if model_key not in INSTRUCT_MODELS:
            raise ValueError(f"Unknown model: {model_key}. Available: {list(INSTRUCT_MODELS.keys())}")
        
        self.config = INSTRUCT_MODELS[model_key]
        self.device = device
        self.dtype = dtype
        
        print(f"Loading {self.config.name}...")
        
        # Quantization config for 8-bit
        quantization_config = None
        if load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.hf_name,
            token=HF_TOKEN,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.hf_name,
            token=HF_TOKEN,
            torch_dtype=dtype,
            device_map=device,
            quantization_config=quantization_config,
            trust_remote_code=True
        )
        
        self.model.eval()
        print(f"✅ {self.config.name} loaded ({self.config.parameters_m}M parameters)")
    
    def format_prompt(
        self, 
        user_message: str, 
        system_prompt: Optional[str] = None
    ) -> str:
        """Format prompt using chat template."""
        system = system_prompt or self.config.system_prompt
        
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message}
        ]
        
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception:
            # Fallback for models without chat template
            prompt = f"System: {system}\n\nUser: {user_message}\n\nAssistant:"
        
        return prompt
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        do_sample: bool = True,
        return_activations: bool = False
    ) -> Tuple[str, Optional[torch.Tensor]]:
        """
        Generate response.
        
        Args:
            prompt: Input prompt (already formatted or raw)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Use sampling vs greedy
            return_activations: Return hidden state activations
            
        Returns:
            (response_text, activations or None)
        """
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        
        # Generation config
        gen_kwargs = {
            'max_new_tokens': max_new_tokens,
            'do_sample': do_sample,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs['temperature'] = temperature
            gen_kwargs['top_p'] = 0.9
        
        activations = None
        
        if return_activations:
            # Generate with output hidden states
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **gen_kwargs,
                    output_hidden_states=True,
                    return_dict_in_generate=True
                )
                
                # Extract activations from last layer
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                    # hidden_states is tuple of tuples
                    # Each generation step has hidden states
                    last_hidden = []
                    for step_hidden in outputs.hidden_states:
                        if step_hidden:
                            # Take last layer
                            last_hidden.append(step_hidden[-1])
                    if last_hidden:
                        activations = torch.cat(last_hidden, dim=1)
                
                sequences = outputs.sequences
        else:
            with torch.no_grad():
                sequences = self.model.generate(**inputs, **gen_kwargs)
        
        # Decode
        response = self.tokenizer.decode(sequences[0], skip_special_tokens=True)
        
        # Extract only assistant's response
        response = self._extract_assistant_response(response, prompt)
        
        return response, activations
    
    def _extract_assistant_response(self, full_response: str, prompt: str) -> str:
        """Extract only the assistant's response from full output."""
        # Try to find assistant marker
        markers = ['<|assistant|>', 'assistant\n', 'Assistant:', '<|im_start|>assistant']
        
        for marker in markers:
            if marker in full_response:
                parts = full_response.split(marker)
                if len(parts) > 1:
                    response = parts[-1].strip()
                    # Remove end tokens
                    for end in ['<|im_end|>', '<|endoftext|>', '</s>']:
                        response = response.split(end)[0]
                    return response.strip()
        
        # Fallback: remove prompt from response
        if prompt in full_response:
            return full_response[len(prompt):].strip()
        
        return full_response.strip()
    
    def generate_with_style(
        self,
        question: str,
        style_prompt: str,
        max_new_tokens: int = 200,
        return_activations: bool = False
    ) -> Tuple[str, Optional[torch.Tensor]]:
        """
        Generate response in specific style.
        
        Args:
            question: The question to answer
            style_prompt: System prompt defining the style
            max_new_tokens: Max tokens
            return_activations: Return activations
            
        Returns:
            (response, activations or None)
        """
        prompt = self.format_prompt(question, system_prompt=style_prompt)
        return self.generate(
            prompt, 
            max_new_tokens=max_new_tokens,
            return_activations=return_activations
        )
    
    def get_hidden_size(self) -> int:
        """Get hidden state dimension."""
        return self.config.hidden_size
    
    def get_num_layers(self) -> int:
        """Get number of layers."""
        return self.config.num_layers
    
    def cleanup(self):
        """Free GPU memory."""
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()


def test_instruct_model(model_key: str = 'qwen-0.5b'):
    """Test instruction-tuned model."""
    print(f"\n{'='*60}")
    print(f"Testing {model_key}")
    print('='*60)
    
    model = InstructModel(model_key)
    
    # Test basic generation
    question = "Tom has 7 apples. Sarah gives him 5 more. How many apples does Tom have?"
    
    # Test with analytical style
    style = """You are an analytical professor. When solving problems:
1. State the problem clearly
2. Break it down step by step
3. Show all calculations
4. End with "Therefore, the answer is X. Q.E.D."
"""
    
    print(f"\nQuestion: {question}")
    print(f"\nStyle: Analytical Professor")
    
    response, activations = model.generate_with_style(
        question, 
        style,
        return_activations=True
    )
    
    print(f"\nResponse:\n{response}")
    
    if activations is not None:
        print(f"\nActivations shape: {activations.shape}")
    
    # Check style markers
    markers = ['step', 'therefore', 'calculate', 'Q.E.D']
    found = [m for m in markers if m.lower() in response.lower()]
    print(f"\nStyle markers found: {found}")
    print(f"Style adherence: {len(found)/len(markers)*100:.0f}%")
    
    model.cleanup()
    
    return len(found) / len(markers)


if __name__ == '__main__':
    # Test available models
    for model_key in ['qwen-0.5b', 'tinyllama']:
        try:
            score = test_instruct_model(model_key)
            print(f"\n✅ {model_key}: Style score = {score*100:.0f}%")
        except Exception as e:
            print(f"\n❌ {model_key} failed: {e}")
