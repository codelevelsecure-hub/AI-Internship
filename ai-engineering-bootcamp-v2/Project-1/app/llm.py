"""OpenAI chat calls and their cost."""

from functools import lru_cache

from openai import OpenAI

from app.schemas import Answer

DEFAULT_MODEL = "gpt-4o"

# Per-1K-token input/output USD (from OpenAI list prices).
MODEL_PRICES_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "o3-mini": (0.0011, 0.0044),
}


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    """One shared client, created on first use (so importing the app needs no API key)."""

    return OpenAI()  # Reads OPENAI_API_KEY from the environment; never hardcode keys.


def compute_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Turn real usage into dollars. Unknown models are priced as the default model."""

    input_per_1k, output_per_1k = MODEL_PRICES_PER_1K.get(model, MODEL_PRICES_PER_1K[DEFAULT_MODEL])
    return (prompt_tokens / 1000 * input_per_1k) + (completion_tokens / 1000 * output_per_1k)


def call_model_structured(messages: list[dict[str, str]], model: str) -> tuple[Answer, int, int, int]:
    """
    Structured output forces exactly the Answer schema.
    Returns (parsed answer, total tokens, prompt tokens, completion tokens).
    Raises ValueError if the model returns nothing parseable.
    """

    completion = _client().chat.completions.parse(model=model, messages=messages, response_format=Answer)

    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise ValueError("Model returned no parseable structured output")

    usage = completion.usage
    total = usage.total_tokens if usage else 0
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    return parsed, total, prompt_tokens, completion_tokens
