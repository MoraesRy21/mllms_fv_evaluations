from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Tuple

import time
import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

class EvaluationApiLlmResult(BaseModel):
    fraude: bool
    confianca: float
    justificativa: str
    similaridade: float


class BaseApiLlmEvaluator(ABC):
    """
    Classe base para avaliação de pares em APIs LLM multimodais.
    Centraliza:
      - loop sobre pares
      - retry/backoff
      - checkpoint
      - delay entre requisições

    Subclasses implementam apenas:
      - chamada de API para um único par
      - lógica de auditoria
      - regra de 'erro que merece retry'
    """

    def __init__(self, config, logger, checkpoint_path, delay_time: float = 0.0, max_retries: int = 5):
        self.config = config
        self.logger = logger
        self.checkpoint_path = checkpoint_path
        self.delay_time = delay_time
        self.max_retries = max_retries

    # -------------------------------------------------------------------------
    # Métodos que a subclasse PRECISA implementar
    # -------------------------------------------------------------------------

    @abstractmethod
    def _single_api_call( self, pair_id: int, img1_pil, img2_pil, prompt_text: str, system_instruction: str,
    ) -> Tuple["EvaluationApiLlmResult", Dict[str, Any]]:
        """
        Chamada de API para UM par (sem retry).
        Deve retornar:
          - EvaluationApiLlmResult
          - metadata: {input_tokens, output_tokens, tempo_inferencia_s, ...}
        Pode lançar exceções (tratadas por infer_pair()).
        """
        ...

    @abstractmethod
    def audit_single_pair( self, img1_pil, img2_pil, prompt_text: str, system_instruction: str) -> None:
        """
        Versão de auditoria (imprime/anota uso de tokens, tempos, etc.).
        """
        ...

    # -------------------------------------------------------------------------
    # Hooks opcionais para a subclasse
    # -------------------------------------------------------------------------

    def is_retryable_error(self, exc: Exception) -> bool:
        """
        Subclasse pode sobrescrever para indicar quais erros merecem retry.
        Ex.: checar por '429', '503', 'UNAVAILABLE', 'rate_limit', etc.
        """
        return False

    def compute_backoff_seconds(self, attempt_index: int) -> float:
        """
        Política de backoff padrão (pode ser sobrescrita pela subclasse).
        attempt_index é 0-based.
        """
        return (attempt_index + 1) * 5.0

    # -------------------------------------------------------------------------
    # Lógica comum de inferência de UM par (com retry)
    # -------------------------------------------------------------------------

    def infer_pair( self, pair_id: int, img1_pil, img2_pil, prompt_text: str, system_instruction: str,
    ) -> Tuple["EvaluationApiLlmResult", Dict[str, Any]]:
        """
        Chamada de alto nível:
          - aplica retry/backoff usando _single_api_call()
          - mede tempo total de inferência
        """
        last_exc: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                result, metadata = self._single_api_call(
                    pair_id=pair_id,
                    img1_pil=img1_pil,
                    img2_pil=img2_pil,
                    prompt_text=prompt_text,
                    system_instruction=system_instruction,
                )

                return result, metadata

            except Exception as e:
                last_exc = e
                if not self.is_retryable_error(e) or attempt == self.max_retries - 1:
                    # Sem retry ou esgotou tentativas
                    raise

                wait_time = self.compute_backoff_seconds(attempt)
                self.logger.warning(f"Erro retryable no par {pair_id} (tentativa {attempt + 1}/{self.max_retries}). Aguardando {wait_time:.1f}s... Erro: {e}")
                time.sleep(wait_time)

        # Teoricamente nunca chega aqui por causa do raise dentro do loop
        raise last_exc or RuntimeError("Falha desconhecida em infer_pair")

    # -------------------------------------------------------------------------
    # Lógica comum de LOOP + checkpoint
    # -------------------------------------------------------------------------

    def _load_checkpoint(self) -> Tuple[List[Dict[str, Any]], set[int]]:
        """
        Lê checkpoint CSV, se existir.
        Retorna:
          - lista de dicts (results já existentes)
          - conjunto de pair_ids já processados
        """
        results: List[Dict[str, Any]] = []
        processed_pair_ids: set[int] = set()

        if self.checkpoint_path is None or not self.checkpoint_path.exists():
            return results, processed_pair_ids

        df_checkpoint = pd.read_csv(self.checkpoint_path)
        if "pair_id" not in df_checkpoint.columns:
            self.logger.warning(
                f"Checkpoint {self.checkpoint_path} não possui coluna 'pair_id'. Ignorando."
            )
            return results, processed_pair_ids

        results = df_checkpoint.to_dict(orient="records")
        processed_pair_ids = set(df_checkpoint["pair_id"].tolist())

        self.logger.warning(
            f"Checkpoint encontrado em {self.checkpoint_path}. "
            f"Retomando a partir de {len(processed_pair_ids)} pares."
        )
        return results, processed_pair_ids

    def _save_checkpoint(self, results: List[Dict[str, Any]]) -> None:
        """
        Salva o checkpoint atual em CSV.
        """
        if self.checkpoint_path is None:
            return
        df = pd.DataFrame(results)
        df.to_csv(self.checkpoint_path, index=False)

    def run_pairs(self, pairs: Iterable[Dict[str, Any]], prompt_text: str, system_instruction: str,
                  tqdm_desc: str, checkpoint_every: int = 20,
    ) -> pd.DataFrame:
        """
        Loop genérico de avaliação sobre uma coleção de pares.

        Cada item de `pairs` deve ter pelo menos:
          - 'pair_id'
          - 'image1'
          - 'image2'
          - 'is_same'
        """
        results, processed_pair_ids = self._load_checkpoint()
        results_list: List[Dict[str, Any]] = list(results)

        tempo_inicio_experimento = time.time()

        progress = tqdm(pairs, desc=tqdm_desc)
        for item in progress:
            pair_id = int(item["pair_id"])

            if pair_id in processed_pair_ids:
                continue

            img1_pil = item["image1"]
            img2_pil = item["image2"]
            is_same = bool(item["is_same"])

            try:
                self.logger.info(f"Processando par {pair_id} - Ground Truth is_same = {is_same}")

                result, meta = self.infer_pair(
                    pair_id=pair_id,
                    img1_pil=img1_pil,
                    img2_pil=img2_pil,
                    prompt_text=prompt_text,
                    system_instruction=system_instruction,
                )

                self.logger.info(f"Predição par {pair_id} - Par é {not result.fraude} - Confiança: {result.confianca} - Similaridade: {result.similaridade}")
                self.logger.info(f"Justificativa: {result.justificativa}")

                linha = {
                    "pair_id": pair_id,
                    "ground_truth_is_same": is_same,
                    "pred_fraude": result.fraude,
                    "pred_confianca": result.confianca,
                    "justificativa": result.justificativa,
                    "similaridade": result.similaridade,
                    "tempo_inferencia_s": meta.get("tempo_inferencia_s"),
                    "input_tokens": meta.get("input_tokens"),
                    "output_tokens": meta.get("output_tokens"),
                    "total_tokens": meta.get("total_tokens"),
                    "status": "success",
                }

            except Exception as e:
                self.logger.error(f"Erro ao processar o par {pair_id}: {e}")
                linha = {
                    "pair_id": pair_id,
                    "ground_truth_is_same": is_same,
                    "pred_fraude": None,
                    "pred_confianca": None,
                    "justificativa": str(e),
                    "similaridade": None,
                    "tempo_inferencia_s": None,
                    "input_tokens": None,
                    "output_tokens": None,
                    "total_tokens": None,
                    "status": "error",
                }

            results_list.append(linha)

            if len(results_list) % checkpoint_every == 0:
                self._save_checkpoint(results_list)

            if self.delay_time > 0:
                time.sleep(self.delay_time)

        # fim do loop
        tempo_total_experimento = time.time() - tempo_inicio_experimento
        self.logger.info(
            f"Loop de avaliação concluído. Tempo total (incluindo delays): "
            f"{tempo_total_experimento / 60:.2f} minutos"
        )

        df_results = pd.DataFrame(results_list)
        return df_results
