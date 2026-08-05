from typing import Any

from models.multimodel_api.base import BaseApiLlmEvaluator
from models.multimodel_api.anthropic import AnthropicApiLlmEvaluator
from models.multimodel_api.gemini import GoogleApiLlmEvaluator
from models.multimodel_api.openai import OpenAiApiLlmEvaluator
from utils.config_loader import Config

REGISTRY_API: dict[str, dict[str, type[Any]]] = {
    "anthropic": {
        "api": AnthropicApiLlmEvaluator,
        "checkpoint_name": "claude_results_checkpoint.csv",
        "final_filename": "claude_final_evaluation.csv",
        "tqdm_desc": "Comitê FLC - Avaliando com Claude (Anthropic)"
    },
    "google": {
        "api": GoogleApiLlmEvaluator,
        "checkpoint_name": "gemini_results_checkpoint.csv",
        "final_filename": "gemini_final_evaluation.csv",
        "tqdm_desc": "Comitê FLC - Avaliando com Gemini (Google)"
    },
    "openai": {
        "api": OpenAiApiLlmEvaluator,
        "checkpoint_name": "gpt_results_checkpoint.csv",
        "final_filename": "gpt_final_evaluation.csv",
        "tqdm_desc": "Comitê FLC - Avaliando com GPT (OpenAI)"
    }
}


def load_platform_api(apis: list[str], config: Config, logger, results_dir) -> list[tuple[str, str, str, BaseApiLlmEvaluator]]:
    logger.info("Carregando API das Plataformas...")
    registry_list: list
    if "all" in apis:
        registry_list = list(REGISTRY_API.items())
    elif set(apis).issubset(REGISTRY_API):
        REGISTRY_FILTRATION = {k: v for k, v in REGISTRY_API.items() if k in apis}
        registry_list = list(REGISTRY_FILTRATION.items())
    else:
        raise ValueError(f"Unknown API '{apis}'. Choose from: {list(REGISTRY_API)}")

    list_apis = []
    for platform_name, platform_value in registry_list:
        ClazzApiLlmEvaluator = platform_value["api"]
        platform_tuple = (platform_name,
                          platform_value["tqdm_desc"],
                          platform_value["final_filename"],
                          ClazzApiLlmEvaluator(config=config, logger=logger, checkpoint_path=results_dir / platform_value["checkpoint_name"]))
        list_apis.append(platform_tuple)
    return list_apis