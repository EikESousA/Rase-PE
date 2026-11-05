"""Construcao do cliente Ollama com as mesmas opcoes em todos os niveis."""

import os
from typing import Any, Dict, List, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def build_llm(
    model_id: str,
    seed: int | None,
    num_predict: int,
    keep_alive: str = "30s",
    stop: List[str] | None = None,
    output_format: str | None = None,
) -> Tuple[OllamaLLM, Dict[str, Any]]:
    """Retorna o cliente e o dicionario de opcoes de amostragem usado nele.

    GEN_NUM_CTX fixa a janela de contexto (o padrao do Ollama, 2048 tokens em
    versoes antigas, inclui a saida e pode truncar os prompts do N3).
    GEN_TEMPERATURE permite zerar a temperatura para reproducao exata.
    """
    options: Dict[str, Any] = {
        "temperature": _env_float("GEN_TEMPERATURE", 0.1),
        "top_p": 0.9,
        "repeat_penalty": 1.1,
        "num_predict": num_predict,
        "num_ctx": _env_int("GEN_NUM_CTX", 4096),
    }
    if seed is not None:
        options["seed"] = seed
    if stop:
        options["stop"] = stop

    llm_kwargs: Dict[str, Any] = {
        "model": model_id,
        "client_kwargs": {"timeout": 600},
        **options,
    }
    if keep_alive:
        llm_kwargs["keep_alive"] = keep_alive
    if output_format:
        llm_kwargs["format"] = output_format
    return OllamaLLM(**llm_kwargs), options


def chain_for_attempt(prompt: ChatPromptTemplate, llm: OllamaLLM, options: Dict[str, Any], attempt: int):
    """Na primeira tentativa usa a semente fixa; nas seguintes varia a semente.

    Repetir a chamada com a mesma semente e temperatura baixa devolve a mesma
    saida, o que tornava as novas tentativas inuteis.
    """
    if attempt <= 1 or options.get("seed") is None:
        return prompt | llm
    retry_options = {**options, "seed": options["seed"] + attempt - 1}
    return prompt | llm.bind(options=retry_options)
