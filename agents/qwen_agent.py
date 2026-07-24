"""Local HuggingFace Qwen backend for the LLM-policy baseline.

Use --agent qwen3b with --model pointing at either:
- a local HuggingFace model directory, or
- a HuggingFace repo id already present in your local HF cache.

The model is loaded lazily on the first action so importing the baseline
does not require torch/transformers unless this agent is actually used.
"""
import os
from pathlib import Path
from typing import Optional

from agents.llm_agent import LLMPolicyAgent
from agents.registry import register_agent
from utils.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-3B-Instruct"


@register_agent("qwen3b")
class QwenAgent(LLMPolicyAgent):
    name = "qwen3b"
    model_name = DEFAULT_QWEN_MODEL

    def __init__(self, seed=None, checkpoint_path=None, model_name: Optional[str] = None, **kwargs):
        super().__init__(seed=seed, checkpoint_path=checkpoint_path, model_name=model_name, **kwargs)
        self.model_name = model_name or os.environ.get("QWEN_MODEL_PATH") or self.model_name
        self.device_map = os.environ.get("QWEN_DEVICE_MAP", "auto")
        self.torch_dtype = os.environ.get("QWEN_TORCH_DTYPE", "auto")
        self.max_new_tokens = int(os.environ.get("QWEN_MAX_NEW_TOKENS", "80"))
        self.local_files_only = os.environ.get("QWEN_LOCAL_FILES_ONLY", "1") != "0"
        self._tokenizer = None
        self._model = None

    def _resolve_model_name(self) -> str:
        """Accept either a repo id, a normal model dir, or a HF cache repo dir."""
        path = Path(self.model_name).expanduser()
        if not path.exists() or not path.is_dir():
            return self.model_name

        if (path / "config.json").exists():
            return str(path)

        snapshots_dir = path / "snapshots"
        if snapshots_dir.is_dir():
            snapshots = sorted(
                (snapshot for snapshot in snapshots_dir.iterdir() if snapshot.is_dir()),
                key=lambda snapshot: snapshot.stat().st_mtime,
                reverse=True,
            )
            for snapshot in snapshots:
                if (snapshot / "config.json").exists():
                    logger.info("Resolved HuggingFace cache dir %s to snapshot %s", path, snapshot)
                    return str(snapshot)

        return str(path)

    def _load_model(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Local Qwen requires torch and transformers. Install them in this "
                "environment, then rerun with --agent qwen3b."
            ) from exc

        dtype = self.torch_dtype
        if dtype != "auto":
            try:
                dtype = getattr(torch, dtype)
            except AttributeError as exc:
                raise RuntimeError(
                    f"Unknown QWEN_TORCH_DTYPE={self.torch_dtype!r}; try auto, float16, "
                    "bfloat16, or float32."
                ) from exc

        logger.info(
            "Loading Qwen model from %s (device_map=%s, dtype=%s, local_files_only=%s)",
            self.model_name,
            self.device_map,
            self.torch_dtype,
            self.local_files_only,
        )
        resolved_model = self._resolve_model_name()
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                resolved_model,
                trust_remote_code=True,
                local_files_only=self.local_files_only,
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not load the Qwen tokenizer. In your active Python env, install "
                "the tokenizer helpers with: pip install tokenizers tiktoken sentencepiece. "
                f"Model path used: {resolved_model}"
            ) from exc

        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                resolved_model,
                torch_dtype=dtype,
                device_map=self.device_map,
                trust_remote_code=True,
                local_files_only=self.local_files_only,
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not load the Qwen model weights. If you passed a HuggingFace cache "
                "directory, try passing its snapshots/<hash> directory directly, or set "
                "QWEN_LOCAL_FILES_ONLY=0 to let Transformers fetch missing files."
            ) from exc
        self._model.eval()

    def _call_backend(self, prompt: str) -> str:
        self._load_model()

        messages = [
            {"role": "user", "content": prompt},
        ]
        if hasattr(self._tokenizer, "apply_chat_template"):
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = f"{self.system_prompt}\n\n{prompt}\n"

        inputs = self._tokenizer(text, return_tensors="pt")
        model_device = getattr(self._model, "device", None)
        if model_device is not None:
            inputs = {key: value.to(model_device) for key, value in inputs.items()}

        do_sample = self.temperature > 0
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.eos_token_id or getattr(self._model.config, "eos_token_id", None),
        }
        if do_sample:
            generation_kwargs.update(
                {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                }
            )

        import torch

        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, **generation_kwargs)

        input_length = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0][input_length:]
        return self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
