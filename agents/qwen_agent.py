"""Template for adding a new model as a baseline agent.

This is the whole diff needed to add "qwen3b" as a --agent option:
this one file, registered via @register_agent. baseline.py, eval/runner.py,
and every other agent file are untouched.

TODO: point `_call_backend` at however Xinbo's checkpoints get served
(local HF `generate`, vLLM server, etc.) — this stub currently just
raises so the shared LLMPolicyAgent fallback (random legal action)
kicks in, exactly like DeepSeekAgent does when no API key is set.
"""
from agents.llm_agent import LLMPolicyAgent
from agents.registry import register_agent


@register_agent("qwen3b")
class QwenAgent(LLMPolicyAgent):
    name = "qwen3b"
    model_name = "Qwen2.5-3B-Instruct"

    def _call_backend(self, prompt: str) -> str:
        raise NotImplementedError(
            "Wire this up to your Qwen serving endpoint (local HF pipeline, "
            "vLLM, etc.) — see DeepSeekAgent for the pattern of an HTTP "
            "backend, or call a local `model.generate(...)` here instead."
        )
