import os
from typing import List, Dict, Any
from typing import Optional, Union

import tiktoken
from crewai.cli.constants import DEFAULT_LLM_MODEL, ENV_VARS, LITELLM_PARAMS
from crewai.llm import LLM, BaseLLM


def create_llm(
    llm_value: Union[str, LLM, Any, None] = None,
) -> Optional[LLM | BaseLLM]:
    """
    Creates or returns an LLM instance based on the given llm_value.

    Args:
        llm_value (str | BaseLLM | Any | None):
            - str: The model name (e.g., "gpt-4").
            - BaseLLM: Already instantiated BaseLLM (including LLM), returned as-is.
            - Any: Attempt to extract known attributes like model_name, temperature, etc.
            - None: Use environment-based or fallback default model.

    Returns:
        A BaseLLM instance if successful, or None if something fails.
    """

    # 1) If llm_value is already a BaseLLM or LLM object, return it directly
    if isinstance(llm_value, LLM) or isinstance(llm_value, BaseLLM):
        return llm_value

    # 2) If llm_value is a string (model name)
    if isinstance(llm_value, str):
        try:
            created_llm = LLM(model=llm_value)
            return created_llm
        except Exception as e:
            print(f"Failed to instantiate LLM with model='{llm_value}': {e}")
            return None

    # 3) If llm_value is None, parse environment variables or use default
    if llm_value is None:
        return _llm_via_environment_or_fallback()

    # 4) Otherwise, attempt to extract relevant attributes from an unknown object
    try:
        # Extract attributes with explicit types
        model = (
            getattr(llm_value, "model", None)
            or getattr(llm_value, "model_name", None)
            or getattr(llm_value, "deployment_name", None)
            or str(llm_value)
        )
        temperature: Optional[float] = getattr(llm_value, "temperature", None)
        max_tokens: Optional[int] = getattr(llm_value, "max_tokens", None)
        logprobs: Optional[int] = getattr(llm_value, "logprobs", None)
        timeout: Optional[float] = getattr(llm_value, "timeout", None)
        api_key: Optional[str] = getattr(llm_value, "api_key", None)
        base_url: Optional[str] = getattr(llm_value, "base_url", None)
        api_base: Optional[str] = getattr(llm_value, "api_base", None)

        created_llm = LLM(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=logprobs,
            timeout=timeout,
            api_key=api_key,
            base_url=base_url,
            api_base=api_base,
        )
        return created_llm
    except Exception as e:
        print(f"Error instantiating LLM from unknown object type: {e}")
        return None


def _llm_via_environment_or_fallback() -> Optional[LLM]:
    """
    Helper function: if llm_value is None, we load environment variables or fallback default model.
    """
    model_name = (
        os.environ.get("MODEL")
        or os.environ.get("MODEL_NAME")
        or os.environ.get("OPENAI_MODEL_NAME")
        or DEFAULT_LLM_MODEL
    )

    # Initialize parameters with correct types
    model: str = model_name
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    logprobs: Optional[int] = None
    timeout: Optional[float] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    api_version: Optional[str] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    logit_bias: Optional[Dict[int, float]] = None
    response_format: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None
    top_logprobs: Optional[int] = None
    callbacks: List[Any] = []

    # Optional base URL from env
    base_url = (
        os.environ.get("BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
        or os.environ.get("OPENAI_BASE_URL")
    )

    api_base = os.environ.get("API_BASE") or os.environ.get("AZURE_API_BASE")

    # Synchronize base_url and api_base if one is populated and the other is not
    if base_url and not api_base:
        api_base = base_url
    elif api_base and not base_url:
        base_url = api_base

    # Initialize llm_params dictionary
    llm_params: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_completion_tokens": max_completion_tokens,
        "logprobs": logprobs,
        "timeout": timeout,
        "api_key": api_key,
        "base_url": base_url,
        "api_base": api_base,
        "api_version": api_version,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "top_p": top_p,
        "n": n,
        "stop": stop,
        "logit_bias": logit_bias,
        "response_format": response_format,
        "seed": seed,
        "top_logprobs": top_logprobs,
        "callbacks": callbacks,
    }

    UNACCEPTED_ATTRIBUTES = [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_REGION_NAME",
    ]
    set_provider = model_name.partition("/")[0] if "/" in model_name else "openai"

    if set_provider in ENV_VARS:
        env_vars_for_provider = ENV_VARS[set_provider]
        if isinstance(env_vars_for_provider, (list, tuple)):
            for env_var in env_vars_for_provider:
                key_name = env_var.get("key_name")
                if key_name and key_name not in UNACCEPTED_ATTRIBUTES:
                    env_value = os.environ.get(key_name)
                    if env_value:
                        # Map environment variable names to recognized parameters
                        param_key = _normalize_key_name(key_name.lower())
                        llm_params[param_key] = env_value
                elif isinstance(env_var, dict):
                    if env_var.get("default", False):
                        for key, value in env_var.items():
                            if key not in ["prompt", "key_name", "default"]:
                                llm_params[key.lower()] = value
                else:
                    print(
                        f"Expected env_var to be a dictionary, but got {type(env_var)}"
                    )

    # Remove None values
    llm_params = {k: v for k, v in llm_params.items() if v is not None}

    # Try creating the LLM
    try:
        new_llm = LLM(**llm_params)
        return new_llm
    except Exception as e:
        print(
            f"Error instantiating LLM from environment/fallback: {type(e).__name__}: {e}"
        )
        return None


def _normalize_key_name(key_name: str) -> str:
    """
    Maps environment variable names to recognized litellm parameter keys,
    using patterns from LITELLM_PARAMS.
    """
    for pattern in LITELLM_PARAMS:
        if pattern in key_name:
            return pattern
    return key_name



def truncate_messages_to_token_limit(
    messages: List[Dict[str, Any]],
    model: str = "gpt-5",
    max_tokens: int = 20_000,
) -> List[Dict[str, Any]]:
    """
    Keep the most recent messages whose *content* fits within max_tokens.
    - Walks from the end (most recent) backwards.
    - If adding a message would push the sum over max_tokens, stop there and
      drop that message and everything older.
    - Returns the truncated list in the original chronological order.
    """
    # robust encoding fallback
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")

    def extract_text(msg: Dict[str, Any]) -> str:
        """
        Handles OpenAI-style message content:
        - If content is str -> use it
        - If content is list of parts -> join text parts
        - Else -> best-effort string cast
        """
        c = msg.get("content", "")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            parts = []
            for part in c:
                if isinstance(part, dict):
                    # prefer 'text' field if present, else stringify
                    if "text" in part and isinstance(part["text"], str):
                        parts.append(part["text"])
                    elif "content" in part and isinstance(part["content"], str):
                        parts.append(part["content"])
                    else:
                        parts.append(str(part))
                else:
                    parts.append(str(part))
            return "\n".join(parts)
        return str(c)

    total = 0
    kept_reversed = []

    # iterate from most recent to oldest
    for msg in reversed(messages):
        text = extract_text(msg)
        token_count = len(enc.encode(text))
        if total + token_count > max_tokens:
            break  # drop this one and everything older
        kept_reversed.append(msg)
        total += token_count

    # put back to chronological order
    return list(reversed(kept_reversed))
