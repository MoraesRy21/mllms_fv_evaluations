from __future__ import annotations

import base64
import io
import time
from typing import Any, Dict, Tuple

from openai import OpenAI
from pydantic import BaseModel, Field

from models.multimodel_api.base import BaseApiLlmEvaluator, EvaluationApiLlmResult


class _OpenAIAvaliacaoBiometrica(BaseModel):
    """
    Schema Pydantic apenas para parse da resposta da OpenAI via .parse().

    Mantém o mesmo contrato lógico dos notebooks:
      - fraude: True se imagens de pessoas DIFERENTES; False se MESMA pessoa.
      - confianca: grau de certeza [0.0, 1.0].
      - justificativa: texto curto justificando a decisão facial.
    """

    fraude: bool = Field(
        description="True se as imagens forem de pessoas DIFERENTES; False se forem da MESMA pessoa."
    )
    confianca: float = Field(
        description="Grau de certeza de 0.0 a 1.0."
    )
    justificativa: str = Field(
        description="Máximo de 30 palavras justificando a decisão facial."
    )
    similaridade: float = Field(
        description="Grau de similaridade entre as duas imagens de 0.0 a 1.0."
    )


class OpenAiApiLlmEvaluator(BaseApiLlmEvaluator):
    """
    Implementação da BaseApiLlmEvaluator para a API da OpenAI (ChatGPT),
    equivalente ao fluxo do notebook 02_chatgpt_pair_evaluation.ipynb.

    Responsável por:
      - configurar o client/modelo da OpenAI
      - converter PIL -> Base64
      - chamar a API para UM par (_single_api_call)
      - definir erros que merecem retry
      - implementar auditoria de um par
    """

    def __init__(self, config, logger, checkpoint_path):
        # Configs específicas da OpenAI
        api_key = config["models.openai.api_key"]
        model_name = config["models.openai.model_name"]
        temperature = config["models.openai.temperature"]
        max_output_tokens = config["models.openai.max_output_tokens"]
        delay_time = config["models.openai.delay_time_between_response"]

        super().__init__(
            config=config,
            logger=logger,
            checkpoint_path=checkpoint_path,
            delay_time=delay_time,
            max_retries=5,  # segue padrão do notebook (ajustável depois se preciso)
        )

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    # -------------------------------------------------------------------------
    # Utilidades internas
    # -------------------------------------------------------------------------

    @staticmethod
    def _pil_to_base64(img) -> str:
        """
        Converte um objeto PIL.Image para uma string Base64 limpa (JPEG).
        """
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    # -------------------------------------------------------------------------
    # Hooks de retry específicos da OpenAI
    # -------------------------------------------------------------------------

    def is_retryable_error(self, exc: Exception) -> bool:
        """
        Define quais erros da OpenAI devem gerar retry.

        Reproduz a lógica do notebook:
          - mensagens contendo 'rate_limit'
          - códigos '503'
        """
        erro_str = str(exc).lower()
        return ("rate_limit" in erro_str) or ("503" in erro_str)

    # -------------------------------------------------------------------------
    # Implementação da chamada de API para UM par (sem retry)
    # -------------------------------------------------------------------------

    def _single_api_call(
            self, pair_id: int, img1_pil, img2_pil, prompt_text: str, system_instruction: str,
    ) -> Tuple[EvaluationApiLlmResult, Dict[str, Any]]:
        """
        Executa UMA chamada à API OpenAI (modelo de chat multimodal) para um par de imagens.
        Não faz retry aqui; o retry é responsabilidade de infer_pair().
        """

        img1_b64 = self._pil_to_base64(img1_pil)
        img2_b64 = self._pil_to_base64(img2_pil)

        messages = [
            {"role": "system", "content": system_instruction},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img1_b64}",
                            "detail": "low",  # trava de custo, como no notebook
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img2_b64}",
                            "detail": "low",
                        },
                    },
                ],
            },
        ]

        start_time = time.time()
        response = self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=messages,
            response_format=_OpenAIAvaliacaoBiometrica,
            temperature=self.temperature,
            max_completion_tokens=self.max_output_tokens,
        )

        end_time = time.time()
        tempo_inferencia_s = round(end_time - start_time, 4)

        parsed = response.choices[0].message.parsed

        # Monta resultado normalizado para o comitê
        result = EvaluationApiLlmResult(
            fraude=bool(parsed.fraude),
            confianca=float(parsed.confianca),
            justificativa=str(parsed.justificativa),
            similaridade=float(parsed.similaridade),
        )

        # Uso de tokens
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
        raw_text = response.choices[0].message

        metadata: Dict[str, Any] = {
            "tempo_inferencia_s": tempo_inferencia_s,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "raw_response_text": raw_text,
        }

        self.logger.debug(f"[COST] Par {pair_id} | In: {input_tokens} | Out: {output_tokens} | Total: {total_tokens}")

        return result, metadata

    # -------------------------------------------------------------------------
    # Auditoria de um par (tokens, etc.)
    # -------------------------------------------------------------------------

    def audit_single_pair(self, img1_pil, img2_pil, prompt_text: str, system_instruction: str) -> None:
        """
        Executa uma chamada de auditoria de um par, registrando:
          - tokens de entrada / saída / total
          - tempo de resposta
          - conteúdo parseado
        """
        print("Preparando requisição de auditoria para o ChatGPT (OpenAI)...")

        img1_b64 = self._pil_to_base64(img1_pil)
        img2_b64 = self._pil_to_base64(img2_pil)

        messages_debug = [
            {"role": "system", "content": system_instruction},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img1_b64}",
                            "detail": "low",
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img2_b64}",
                            "detail": "low",
                        },
                    },
                ],
            },
        ]

        start_time_debug = time.time()
        response_debug = self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=messages_debug,
            response_format=_OpenAIAvaliacaoBiometrica,
            temperature=self.temperature,
            max_completion_tokens=self.max_output_tokens,
        )
        end_time_debug = time.time()

        parsed = response_debug.choices[0].message.parsed
        usage = response_debug.usage

        print("--- ANATOMIA DOS TOKENS (OpenAI) ---")
        print(f"Tokens de Entrada (Prompt + Imagens): {usage.prompt_tokens}")
        print(f"Tokens de Saída (Completion): {usage.completion_tokens}")
        print(f"Total de Tokens: {usage.total_tokens}")

        print("--- PERFORMANCE ---")
        print(f"Tempo de resposta da API: {end_time_debug - start_time_debug:.2f} segundos")

        print("--- TEXTO GERADO (OBJETO PARSEADO) ---")
        print(f"Fraude (Pessoas Diferentes?): {parsed.fraude}")
        print(f"Confiança: {parsed.confianca}")
        print(f"Justificativa: {parsed.justificativa}")
        print(f"Similaridade: {parsed.similaridade}")

        print("--- RESPONSE RETORNADO ---")
        print(str(response_debug))