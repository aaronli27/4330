from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model_backend import QwenBackend  # noqa: E402


def main() -> int:
    backend = QwenBackend(
        model_name=str(
            PROJECT_ROOT / "models" / "Qwen2.5-1.5B-Instruct"
        ),
        load_in_4bit=True,
    )
    result = backend.generate(
        'Return JSON only: {"status":"answer","answer":"ok","evidence_ids":[]}',
        max_new_tokens=64,
    )
    print(
        {
            "text": result.text,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "latency_ms": result.latency_ms,
            "device": str(backend.model.device),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
