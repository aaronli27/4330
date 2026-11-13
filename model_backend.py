from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional, Protocol


@dataclass(frozen=True)
class GenerationResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ModelBackend(Protocol):
    def count_tokens(self, text: str) -> int:
        ...

    def generate(self, prompt: str, max_new_tokens: int) -> GenerationResult:
        ...


class FakeBackend:
    """Deterministic backend for pipeline tests without model dependencies."""

    def __init__(
        self,
        responder: Optional[Callable[[str], str]] = None,
        tokens_per_word: int = 1,
    ) -> None:
        self.responder = responder or self._default_responder
        self.tokens_per_word = tokens_per_word

    @staticmethod
    def _default_responder(prompt: str) -> str:
        if "TASK: DECOMPOSE" in prompt:
            return '{"subquestions": ["What is the first required fact?", "What is the second required fact?"]}'
        if "TASK: GLOBAL_GATE" in prompt:
            return '{"sufficient": true, "confidence": 0.9, "reason": "The evidence appears sufficient."}'
        if "TASK: LOCAL_GATE" in prompt:
            return '{"sufficient": true, "confidence": 0.9, "covered_subquestions": [true, true], "reason": "Covered."}'
        return '{"status": "answer", "answer": "synthetic answer", "evidence_ids": []}'

    def count_tokens(self, text: str) -> int:
        return max(1, len(text.split()) * self.tokens_per_word)

    def generate(self, prompt: str, max_new_tokens: int) -> GenerationResult:
        started = time.perf_counter()
        text = self.responder(prompt)
        output_tokens = min(self.count_tokens(text), max_new_tokens)
        if output_tokens < self.count_tokens(text):
            words = text.split()
            text = " ".join(words[:output_tokens])
        elapsed = (time.perf_counter() - started) * 1000.0
        return GenerationResult(
            text=text,
            input_tokens=self.count_tokens(prompt),
            output_tokens=output_tokens,
            latency_ms=elapsed,
        )


class QwenBackend:
    """Optional Transformers adapter; imports heavy dependencies lazily."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        load_in_4bit: bool = True,
        device_map: str = "auto",
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "QwenBackend requires torch and transformers. "
                "Install them in the execution environment before model runs."
            ) from exc

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_kwargs = {
            "device_map": device_map,
            "dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        }
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "4-bit loading requires transformers BitsAndBytesConfig "
                    "and bitsandbytes support. Set load_in_4bit=False to use "
                    "the model's normal loading path."
                ) from exc
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.model.eval()

    def _chat_input_ids(self, prompt: str):
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            prepared = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            )
            if hasattr(prepared, "input_ids"):
                return prepared.input_ids
            if isinstance(prepared, dict) and "input_ids" in prepared:
                return prepared["input_ids"]
            return prepared
        return self.tokenizer(prompt, return_tensors="pt").input_ids

    def count_tokens(self, text: str) -> int:
        return int(self._chat_input_ids(text).shape[-1])

    def generate(self, prompt: str, max_new_tokens: int) -> GenerationResult:
        started = time.perf_counter()
        inputs = self._chat_input_ids(prompt)
        device = self.model.device
        inputs = inputs.to(device)
        with self._torch.inference_mode():
            generated = self.model.generate(
                input_ids=inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = generated[:, inputs.shape[1] :]
        text = self.tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()
        elapsed = (time.perf_counter() - started) * 1000.0
        return GenerationResult(
            text=text,
            input_tokens=int(inputs.shape[1]),
            output_tokens=int(new_tokens.shape[1]),
            latency_ms=elapsed,
        )

