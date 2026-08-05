from __future__ import annotations

import time
from typing import Any, Dict, Tuple
import json

from google import genai
from google.genai import types

from models.multimodel_api.base import BaseApiLlmEvaluator, EvaluationApiLlmResult


class GoogleApiLlmEvaluator(BaseApiLlmEvaluator):
    """
    Implementação da BaseApiLlmEvaluator para a API do Google Gemini.

    Responsável apenas por:
      - configurar o client/gen_config/safety
      - fazer a chamada para UM par (_single_api_call)
      - definir quais erros merecem retry
      - implementar a auditoria de um par
    """

    def __init__(self, config, logger, checkpoint_path):
        # Lê configs específicas do Gemini
        temperature = config["models.gemini.temperature"]
        max_output_tokens = config["models.gemini.max_output_tokens"]
        model_name = config["models.gemini.model_name"]
        api_key = config["models.gemini.api_key"]
        delay_time = config["models.gemini.delay_time_between_response"]

        # pode ser parametrizado depois, se quiser
        super().__init__(config=config, logger=logger, checkpoint_path=checkpoint_path, delay_time=delay_time, max_retries=5)

        self.model_name = model_name

        # Client da API Gemini
        self.client = genai.Client(api_key=api_key)

        # Safety settings iguais aos dos notebooks
        self.safety_settings = [
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
        ]

        # Schema de resposta:
        # Vamos usar JSON livre com response_mime_type="application/json"
        # e parse manual via json.loads(), como já é feito em _single_api_call.
        self.generation_config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
            response_schema=EvaluationApiLlmResult,
            system_instruction=config["prompting.system_instruction"],
            safety_settings=self.safety_settings,
        )

    # -------------------------------------------------------------------------
    # Hooks de retry específicos do Gemini
    # -------------------------------------------------------------------------

    def is_retryable_error(self, exc: Exception) -> bool:
        """
        Define quais erros da API Gemini devem gerar retry.
        Reproduz o comportamento dos notebooks:
          - status 503
          - status 429
          - mensagem contendo 'UNAVAILABLE'
        """
        erro_str = str(exc).upper()
        return (
                "503" in erro_str
                or "429" in erro_str
                or "UNAVAILABLE" in erro_str
        )

    def compute_backoff_seconds(self, attempt_index: int) -> float:
        """
        Mesmo padrão de backoff progressivo usado no notebook do Gemini:
          (tentativa + 2) * 5  →  10, 15, 20, 25, 30...
        """
        return (attempt_index + 2) * 5.0

    # -------------------------------------------------------------------------
    # Implementação da chamada de API para UM par (sem retry)
    # -------------------------------------------------------------------------

    def _single_api_call(self, pair_id: int, img1_pil, img2_pil, prompt_text: str, system_instruction: str
    ) -> Tuple[EvaluationApiLlmResult, Dict[str, Any]]:
        """
        Executa UMA chamada à API Gemini para um par de imagens.
        Não faz retry aqui; o retry é responsabilidade de infer_pair().
        """
        # A system_instruction está na generation_config, mas mantemos a
        # assinatura consistente com a base. Se precisar diferenciar por chamada,
        # podemos clonar generation_config aqui no futuro.
        start_time = time.time()
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[img1_pil, img2_pil, prompt_text],
            config=self.generation_config,
        )
        end_time = time.time()

        raw_text = response.text or ""
        raw_text = raw_text.strip()

        # Remove possíveis fences de markdown se o modelo devolver ```json ... ```
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:-3].strip()
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:-3].strip()

        if not raw_text:
            raise ValueError("A resposta da API veio vazia (sem texto).")

        try:
            dados = json.loads(raw_text)
        except json.JSONDecodeError as e:
            self.logger.error(f"Falha ao fazer json.loads() da resposta do Gemini no par {pair_id}: {e}. Texto bruto: {response.text!r}")
            raise

        # Monta objeto de resultado normalizado
        fraude = bool(dados.get("fraude"))
        confianca = float(dados.get("confianca"))
        justificativa = str(dados.get("justificativa"))
        similaridade = float(dados.get("similaridade"))

        resultado = EvaluationApiLlmResult(
            fraude=fraude,
            confianca=confianca,
            justificativa=justificativa,
            similaridade=similaridade,
        )

        # Metadata de tokens / custo
        usage = getattr(response, "usage_metadata", None)

        metadata: Dict[str, Any] = {
            "tempo_inferencia_s": round(end_time - start_time, 4),
            "raw_response_text": response.text,
            "input_tokens": getattr(usage, "prompt_token_count", None),
            "output_tokens": getattr(usage, "candidates_token_count", None),
            "thoughts_tokens": getattr(usage, "thoughts_token_count", None),
            "total_tokens": getattr(usage, "total_token_count", None),
        }

        return resultado, metadata

    # -------------------------------------------------------------------------
    # Auditoria de um par (tokens, etc.)
    # -------------------------------------------------------------------------

    def audit_single_pair(
            self,
            img1_pil,
            img2_pil,
            prompt_text: str,
            system_instruction: str,
    ) -> None:
        """
        Executa uma chamada de auditoria para inspecionar uso de tokens,
        tempo de resposta e o JSON gerado.
        """
        # Para evitar custos desnecessários, fazemos um thumbnail
        img1_safe = img1_pil.copy()
        img1_safe.thumbnail((512, 512))
        img2_safe = img2_pil.copy()
        img2_safe.thumbnail((512, 512))

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[img1_safe, img2_safe, prompt_text],
            config=self.generation_config,
        )

        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", None)
        completion_tokens = getattr(usage, "candidates_token_count", None)
        thoughts_tokens = getattr(usage, "thoughts_token_count", None)

        print("--- ANATOMIA DOS TOKENS (Gemini) ---")
        print(f"Tokens de Entrada (Prompt): {prompt_tokens}")
        print(f"Tokens de Saída (Candidato/Texto): {completion_tokens}")
        if thoughts_tokens is not None:
            print(f"Tokens de Raciocínio Oculto (Thoughts): {thoughts_tokens}")

        print("--- TEXTO GERADO (JSON bruto) ---")
        print(str(response.text))

        print("--- RESPONSE RETORNADO ---")
        print(str(response))