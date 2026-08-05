from __future__ import annotations

import base64
import io
import time
from typing import Any, Dict, Tuple

from anthropic import Anthropic
from pydantic import BaseModel, Field

from models.multimodel_api.base import BaseApiLlmEvaluator, EvaluationApiLlmResult


class _ClaudeAvaliacaoBiometrica(BaseModel):
    """
    Schema lógico equivalente aos outros providers.
    Será usado apenas para validar/normalizar o JSON retornado pelo Claude.
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


class AnthropicApiLlmEvaluator(BaseApiLlmEvaluator):
    """
    Implementação da BaseApiLlmEvaluator para a API da Anthropic (Claude),
    equivalente ao fluxo do notebook 03_claude_pair_evaluation.ipynb.

    Responsável por:
      - configurar client/modelo do Claude
      - converter PIL -> Base64 (ou formato aceito pelo client)
      - chamar a API para UM par (_single_api_call)
      - definir erros que merecem retry
      - implementar auditoria de um par
    """

    def __init__(self, config, logger, checkpoint_path):
        api_key = config["models.anthropic.api_key"]
        model_name = config["models.anthropic.model_name"]
        temperature = config["models.anthropic.temperature"]
        max_output_tokens = config["models.anthropic.max_output_tokens"]
        delay_time = config["models.anthropic.delay_time_between_response"]

        super().__init__(
            config=config,
            logger=logger,
            checkpoint_path=checkpoint_path,
            delay_time=delay_time,
            max_retries=5,
        )

        self.client = Anthropic(api_key=api_key)
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
    # Hooks de retry específicos da Anthropic
    # -------------------------------------------------------------------------

    def is_retryable_error(self, exc: Exception) -> bool:
        """
        Define quais erros do Claude devem gerar retry.
        Ajuste aqui conforme o padrão real de erros da Anthropic.
        """
        erro_str = str(exc).lower()
        return (
                "rate_limit" in erro_str
                or "timeout" in erro_str
                or "temporary_unavailable" in erro_str
                or "503" in erro_str
        )

    # -------------------------------------------------------------------------
    # Implementação da chamada de API para UM par (sem retry)
    # -------------------------------------------------------------------------

    def _single_api_call(self, pair_id: int, img1_pil, img2_pil, prompt_text: str, system_instruction: str
    ) -> Tuple[EvaluationApiLlmResult, Dict[str, Any]]:
        """
        Executa UMA chamada à API Claude multimodal para um par de imagens.
        Não faz retry aqui; o retry é responsabilidade de infer_pair().
        """
        img1_b64 = self._pil_to_base64(img1_pil)
        img2_b64 = self._pil_to_base64(img2_pil)

        # Estrutura de mensagens compatível com Anthropic Messages API multimodal.
        #
        # Documentação (resumida): cada mensagem tem um "role" (user/assistant)
        # e "content" é uma lista de partes ("type": "text" | "image").
        #
        # Aqui enviamos:
        #   - o texto do usuário contendo o prompt
        #   - duas imagens base64 (JPEG)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{system_instruction}\n\n{prompt_text}",
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img1_b64,
                        },
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img2_b64,
                        },
                    },
                ],
            }
        ]

        start_time = time.time()
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_output_tokens,
            temperature=self.temperature,
            messages=messages,
        )
        end_time = time.time()

        tempo_inferencia_s = round(end_time - start_time, 4)

        # Claude tipicamente devolve o conteúdo em response.content[0].text
        # (ou em múltiplos blocos). Aqui concatenamos todos os blocos "text".
        textos = []
        for block in getattr(response, "content", []):
            if getattr(block, "type", None) == "text":
                textos.append(getattr(block, "text", ""))

        raw_text = "\n".join(t.strip() for t in textos if t is not None).strip()
        if not raw_text:
            raise ValueError("A resposta da API Claude veio vazia (sem texto).")

        # Esperamos um JSON compatível com _ClaudeAvaliacaoBiometrica.
        # Caso o modelo devolva texto “solto”, será necessário ajustar prompt
        # para forçar JSON puro.
        try:
            parsed = _ClaudeAvaliacaoBiometrica.model_validate_json(raw_text)
        except Exception as e:
            self.logger.error(f"Falha ao parsear JSON do Claude no par {pair_id}: {e}. Texto bruto: {raw_text!r}")
            raise

        result = EvaluationApiLlmResult(
            fraude=bool(parsed.fraude),
            confianca=float(parsed.confianca),
            justificativa=str(parsed.justificativa),
            similaridade=float(parsed.similaridade),
        )

        # Uso de tokens (se disponível na resposta)
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        total_tokens = int(output_tokens) + int(input_tokens)

        metadata: Dict[str, Any] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "tempo_inferencia_s": tempo_inferencia_s,
            "raw_response_text": raw_text,
        }

        self.logger.debug(f"[COST] Par {pair_id} | In: {input_tokens} | Out: {output_tokens}")

        return result, metadata

    # -------------------------------------------------------------------------
    # Auditoria de um par (tokens, etc.)
    # -------------------------------------------------------------------------

    def audit_single_pair(self, img1_pil, img2_pil, prompt_text: str, system_instruction: str
    ) -> None:
        """
        Executa uma chamada de auditoria para um par, registrando:
          - uso de tokens
          - tempo de resposta
          - conteúdo JSON gerado
        """
        print("Preparando requisição de auditoria para o Claude (Anthropic)...")

        img1_b64 = self._pil_to_base64(img1_pil)
        img2_b64 = self._pil_to_base64(img2_pil)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{system_instruction}\n\n{prompt_text}",
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img1_b64,
                        },
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img2_b64,
                        },
                    },
                ],
            }
        ]

        start_time_debug = time.time()
        response_debug = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_output_tokens,
            temperature=self.temperature,
            messages=messages,
        )
        end_time_debug = time.time()

        textos = []
        for block in getattr(response_debug, "content", []):
            if getattr(block, "type", None) == "text":
                textos.append(getattr(block, "text", ""))

        raw_text = "\n".join(t.strip() for t in textos if t is not None).strip()

        usage = getattr(response_debug, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)

        print("--- ANATOMIA DOS TOKENS (Anthropic / Claude) ---")
        print(f"Tokens de Entrada (Prompt + Imagens): {input_tokens}")
        print(f"Tokens de Saída (Completion): {output_tokens}")
        print(f"Tokens totais: {int(output_tokens) + int(input_tokens)}")

        print("--- PERFORMANCE ---")
        print(f"Tempo de resposta da API: {end_time_debug - start_time_debug:.2f} segundos")

        print("--- TEXTO GERADO (BRUTO) ---")
        print(raw_text)

        print("--- RESPONSE RETORNADO ---")
        print(response_debug)