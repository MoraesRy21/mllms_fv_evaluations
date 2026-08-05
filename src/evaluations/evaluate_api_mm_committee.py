"""
Multi‑model LLM committee evaluation over face pairs.

This script:
  - loads a binary face-pair dataset (face_eval_bin_pair-*.bin);
  - runs inference on one or more multimodal APIs (Google/Gemini, OpenAI, Anthropic);
  - computes binary classification metrics per API (fraud vs genuine);
  - builds and evaluates a committee (majority vote across APIs);
  - exports CSVs, JSON metrics and plots.

Typical usage (from project root):

  python evaluate_api_mm_committee.py --apis google openai anthropic
  python evaluate_api_mm_committee.py --apis all
  python evaluate_api_mm_committee.py --apis google --no-plot

Key options:
  --config
      YAML config path. Controls:
        - paths (results_dir, logs_dir, bin_subset, plot_dir, etc.);
        - models (API names, keys, limits);
        - prompting (system_instruction, user_prompt).
      Default: config.fc_pair_eval.yaml

  --apis
      Which platforms to evaluate. Allowed values:
        - "google"     → Google/Gemini API
        - "openai"     → OpenAI API
        - "anthropic"  → Anthropic/Claude API
        - "all"        → runs all configured APIs
      Examples:
        --apis google
        --apis google openai
        --apis all

  --no-plot
      Disable plot generation (CSV/JSON only), useful for batch runs.

  --debug-only / --debug-pair-id
      Run a single detailed audit (audit_single_pair) instead of full evaluation.

  --reuse-existing
      Reuses existing *_final_evaluation.csv files instead of calling the APIs again.

  --max-pairs
      Limits the evaluation to the first N pairs from the bin. Useful for smoke tests with low API cost.

  --shuffle-pairs
      If set together with --max-pairs, shuffles the pairs before selecting the first N. Useful to randomize which pairs are used in smoke tests.

Outputs (per run):
  - Results directory:
      <paths.results_dir> / <bin_suffix>

    where <bin_suffix> is derived from the configured face_eval_bin_pair-*.bin
    (see _build_logger_and_paths for details). This keeps results for different
    bin strategies separated automatically.

  Inside that directory:

    Per API:
      - <api>_..._final_evaluation.csv
          Raw per-pair results (status, prediction, confidence, timing, etc.)
      - metrics.json
          List of dicts with per-API metrics:
            Accuracy, Precision, Recall, F1, TN/FP/FN/TP, AUC/TAR@FAR (if scores)

    Committee:
      - committee_full_analysis.csv
          Full committee dataframe: votes, errors, unanimity/divergence flags
      - committee_analise_casos_polemicos.csv
          Subset of “hard” cases (errors or disagreement between models)
      - committee_metrics.json
          JSON with:
            per_model.gemini/chatgpt/claude/committee_majority
            per_model[*].metadata (time & token aggregates per API)
            concordance stats (unanimous vs divergent, etc.)

    Plots (if --no-plot is not used):
      - mm_committee_<API>_confusion_mtx_and_confidence.png
      - mm_committee_all_apis_confusion_mtx_side_by_side.png
      - mm_committee_comparative_accuracy_and_concordance.png
"""

import argparse
import datetime
import json
import random
from argparse import Namespace
from logging import Logger
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from dataset.pairs.bin_reader import FacePairBinReader
from evaluations.metrics import compute_binary_classification_metrics, to_native, plot_api_results, \
    plot_combined_confusion_matrices, compute_committee_metrics, plot_committee_results, plot_roc_curves
from models.multimodel_api import load_platform_api, BaseApiLlmEvaluator
from utils.config_loader import Config
from utils.log_utils import setup_hybrid_logger
from utils.paths import ProjectPathResolver

def parse_args():
    parser = argparse.ArgumentParser(description=("Evaluation of multimodal LLM APIs on face pairs, including committee (majority vote) analysis."))
    parser.add_argument(
        "--config", type=str, default="config.fc_pair_eval.yaml",
        help=("YAML config path. Defines paths, models and prompting. Default: config.fc_pair_eval.yaml"),
    )
    parser.add_argument("--apis", type=str, nargs="+", required=True,
        help=("List of APIs to run. Allowed values: 'google', 'openai', 'anthropic' or 'all'. Examples:\n"
            "  --apis google\n"
            "  --apis google openai\n"
            "  --apis all"
        ),
    )
    parser.add_argument("--no-plot", action="store_true", help="If set, do not generate plots (only CSV/JSON outputs).")

    # --- Debug/audit flags ---
    parser.add_argument("--debug-only", action="store_true",
        help=("Run only audit_single_pair for one pair per selected API, instead of evaluating the full dataset."))
    parser.add_argument("--debug-pair-id", type=int, default=None,
        help=("Specific pair_id to audit in --debug-only mode. If not provided, uses the first available pair."))

    # --- Cust control / reusing the outputs ---
    parser.add_argument("--reuse-existing", action="store_true",
        help=("If set, reuses existing *_final_evaluation.csv files instead of calling the APIs again. "
              "If a CSV for an API does not exist, inference will be executed normally."))

    # --- Optionally limit the number of pairs to be evaluated (smoke test / dry run) ---
    parser.add_argument("--max-pairs", type=int, default=None,
        help=("If set, limits the evaluation to the first N pairs from the bin. Useful for smoke tests with low API cost."))

    # --- shuffle pair when use max-pairs flag ---
    parser.add_argument("--shuffle-pairs", action="store_true",
        help=("If set together with --max-pairs, shuffles the pairs before selecting "
              "the first N. Useful to randomize which pairs are used in smoke tests."),
    )

    return parser.parse_args()

def _build_logger_and_paths(config: Config, path_resolver: ProjectPathResolver, debug_mode: bool):
    """
    Create logger, resolve the paths in the results/logs, and return common utilities.
    """

    filename_suffix = path_resolver['paths.bin_subset'].path.stem.replace('face_eval_bin_pair', 'pair')
    results_dir: Path = (path_resolver["paths.results_dir"] / filename_suffix).path
    plots_dir: Path = results_dir / "plots"
    logs_dir: Path = results_dir / "logs"

    if not debug_mode:
        results_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"evaluation_mm_committee"
    log_file_path = logs_dir / f"{log_filename}_{now}.log" if not debug_mode else None

    logger = setup_hybrid_logger(
        name=log_filename,
        level="INFO",
        log_file=log_file_path,
        stream_target="auto",
        notebook_friendly=False if not debug_mode else True,
        clear_handlers=True,
    )

    logger.info(f"Results dir: {results_dir}")
    logger.info(f"Logs dir: {logs_dir}")
    logger.info(f"Config loaded from: {config.path}")

    return logger, results_dir, plots_dir

def _load_face_pairs(logger: Logger, path_resolver: ProjectPathResolver, max_pairs: int, shuffle_pairs: bool) -> list[dict[str, Any]]:
    logger.info(f"Loading Faces Pairs... Files: {path_resolver['paths.bin_subset'].path.name}")
    bin_subset_path = path_resolver["paths.bin_subset"].path
    reader = FacePairBinReader(bin_subset_path)
    face_pairs = reader.load_pil_pairs()

    total_pairs = len(face_pairs)
    total_genuinos = sum(1 for p in face_pairs if bool(p["is_same"]))
    total_impostores = total_pairs - total_genuinos

    logger.info(f"Total of pair ready for evaluations: {total_pairs} → ({total_genuinos} Genuine / {total_impostores} Impostors)")

    if max_pairs is not None:
        if max_pairs <= 0:
            logger.error(f"--max-pairs={max_pairs} (value must be > 0), exiting the application.")
            raise ValueError("Value of the flag max-pair must be more than 0.")
        else:
            if shuffle_pairs:
                logger.info(f"[MAX-PAIRS] Flag --shuffle-pairs ativa. Shuffling pairs before select the firsts {max_pairs}.")
                random.shuffle(face_pairs)
            original_len = len(face_pairs)
            face_pairs = face_pairs[: max_pairs]
            logger.info(f"[MAX-PAIRS] Limiting the evalution to the fist {len(face_pairs)} pairs (out of a total of {original_len} pairs available in the bin).")
    return face_pairs

def _extract_and_append_metrics(all_metrics: list[dict[str, Any]], df_success: Series | DataFrame | Any, logger: Logger, platform_name: str):
    # 1 for different images and 0 for equals.
    y_true = (~df_success["ground_truth_is_same"]).astype(int).values
    # 1 for different images and 0 for equals.
    y_pred = df_success["pred_fraude"].astype(int).values
    y_score = 1 - df_success["similaridade"].astype(float).values

    metrics_dict = compute_binary_classification_metrics(
        model_name=platform_name,
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
    )

    all_metrics.append(metrics_dict)

    logger.info(f"Metrics of classification ({platform_name}): {metrics_dict}")

    return y_score, y_true


def _extract_committee_dataframe(api_success_dfs: dict[str, pd.DataFrame], logger: Logger) -> pd.DataFrame | None:
    """
    Constructs the committee DataFrame from the success DataFrames of each API.
    Constrói o DataFrame do comitê a partir dos DataFrames de sucesso de cada API.

    Expects keys: 'google', 'openai', 'anthropic', with columns:
        - pair_id
        - ground_truth_is_same
        - pred_fraude
        - justificativa
        - (optional for metadata) tempo_inferencia_s, input_tokens, output_tokens, total_tokens

    Retorna:
      - df_comite indexed by pair_id, or None if it cannot be constructed.
    """
    required_apis = {"google", "openai", "anthropic"}
    available_apis = set(api_success_dfs.keys())

    if not required_apis.issubset(available_apis):
        logger.info(
            "[COMMITTEE] The three APIs required for the committee are missing. "
            f"Needed: {sorted(required_apis)} | Available: {sorted(available_apis)}"
        )
        return None

    df_google = api_success_dfs["google"].copy()
    df_openai = api_success_dfs["openai"].copy()
    df_anthropic = api_success_dfs["anthropic"].copy()

    # Index by pair_id
    for name, df in [("google", df_google), ("openai", df_openai), ("anthropic", df_anthropic)]:
        if "pair_id" not in df.columns:
            logger.warning(f"[COMMITTEE] API DataFrame {name} does not have a 'pair_id' column; skipping committee.")
            return None

    df_google = df_google.set_index("pair_id")
    df_openai = df_openai.set_index("pair_id")
    df_anthropic = df_anthropic.set_index("pair_id")

    # Pairwise intersection (safe for scripts; in notebooks, indices are assumed to be equal)
    common_index = df_google.index.intersection(df_openai.index).intersection(df_anthropic.index)
    if common_index.empty:
        logger.warning("[COMMITTEE] No common pair_id across the three APIs; the committee will not be calculated.")
        return None

    df_google = df_google.loc[common_index]
    df_openai = df_openai.loc[common_index]
    df_anthropic = df_anthropic.loc[common_index]

    df_comite = pd.DataFrame(index=common_index)

    # --- 2. Ground truth and predictions (same as the notebook) ---
    df_comite["ground_truth_fraude"] = ~df_google["ground_truth_is_same"]

    df_comite["pred_gemini"] = df_google["pred_fraude"].astype(bool)
    df_comite["pred_chatgpt"] = df_openai["pred_fraude"].astype(bool)
    df_comite["pred_claude"] = df_anthropic["pred_fraude"].astype(bool)

    df_comite["justificativa_gemini"] = df_google.get("justificativa")
    df_comite["justificativa_chatgpt"] = df_openai.get("justificativa")
    df_comite["justificativa_claude"] = df_anthropic.get("justificativa")

    df_comite["similaridade_gemini"] = df_google.get("similaridade")
    df_comite["similaridade_chatgpt"] = df_openai.get("similaridade")
    df_comite["similaridade_claude"] = df_anthropic.get("similaridade")

    # --- 3. Committee logic ---
    df_comite["soma_votos_fraude"] = df_comite[["pred_gemini", "pred_chatgpt", "pred_claude"]].sum(axis=1)

    # Majority vote (>= 2 votes for fraud)
    df_comite["pred_maioria"] = df_comite["soma_votos_fraude"] >= 2

    # Level of agreement
    df_comite["unanimidade"] = (df_comite["soma_votos_fraude"] == 0) | (df_comite["soma_votos_fraude"] == 3)
    df_comite["divergencia"] = ~df_comite["unanimidade"]

    # Calculation of correct predictions by model and by the committee
    df_comite["gemini_acertou"] = df_comite["pred_gemini"] == df_comite["ground_truth_fraude"]
    df_comite["chatgpt_acertou"] = df_comite["pred_chatgpt"] == df_comite["ground_truth_fraude"]
    df_comite["claude_acertou"] = df_comite["pred_claude"] == df_comite["ground_truth_fraude"]
    df_comite["comite_acertou"] = df_comite["pred_maioria"] == df_comite["ground_truth_fraude"]

    # --- 4. API metadata (time/tokens), if available ---
    for src_df, prefix in [
        (df_google, "gemini"),
        (df_openai, "chatgpt"),
        (df_anthropic, "claude"),
    ]:
        for col in ["tempo_inferencia_s", "input_tokens", "output_tokens", "total_tokens"]:
            if col in src_df.columns:
                df_comite[f"{prefix}_{col}"] = src_df[col]

    return df_comite

def _debug_audit_inference(args: Namespace, face_pairs: list[dict[str, Any]], logger: Logger,
                           platform_to_run: list[tuple[str, str, str, BaseApiLlmEvaluator]], prompt_text,
                           system_instruction):
    logger.info("DEBUG mode enabled: performing audit for only one pair per selected API.")

    def _select_debug_pair(face_pairs, debug_pair_id: int | None):
        """
        Select a pair for auditing:
          - if debug_pair_id is provided, attempt to find that pair_id
          - otherwise, simply use the first pair in the list
        """
        if not face_pairs:
            raise ValueError("Empty pair list; no pair available for debugging.")

        if debug_pair_id is None:
            return face_pairs[0]

        for item in face_pairs:
            # assume estrutura dict com chave 'pair_id'; adapte se for tupla
            if int(item["pair_id"]) == debug_pair_id:
                return item

        raise ValueError(f"Pair with pair_id={debug_pair_id} not found in the evaluation subset.")

    # Select the target pair for debugging
    debug_item = _select_debug_pair(face_pairs, args.debug_pair_id)

    # If face_pairs is list of dicts:
    img1_pil = debug_item["image1"]
    img2_pil = debug_item["image2"]
    pair_id = debug_item["pair_id"]
    is_same = debug_item["is_same"]

    logger.info(f"Par selecionado para auditoria (pair_id={pair_id}) - is_same={is_same}")

    for platform_name, tqdm_desc, final_filename, evaluator in platform_to_run:
        logger.info(f"[DEBUG] Auditoria com API: {platform_name.upper()}")
        evaluator.audit_single_pair(
            img1_pil=img1_pil,
            img2_pil=img2_pil,
            prompt_text=prompt_text,
            system_instruction=system_instruction,
        )

    logger.info(f"Audit completed for the APIs: {[api_run[0] for api_run in platform_to_run]}")


def main():
    args = parse_args()

    # -------------------------------------------------------------------------
    # 1. Load configuration and pairs once
    # -------------------------------------------------------------------------
    config = Config(path=args.config)
    path_resolver = ProjectPathResolver(config)

    logger, results_dir, plots_dir = _build_logger_and_paths(config, path_resolver, args.debug_only)

    platform_to_run = load_platform_api(args.apis, config, logger, results_dir)

    face_pairs = _load_face_pairs(logger, path_resolver, args.max_pairs, args.shuffle_pairs)

    prompt_text = config["prompting.user_prompt"]
    system_instruction = config["prompting.system_instruction"]

    # -------------------------------------------------------------------------
    # 1.1 Interactive confirmation
    # -------------------------------------------------------------------------
    if not args.debug_only:
        total_pairs = len(face_pairs)
        api_names = [name for (name, _, _, _) in platform_to_run]

        print("\n==== EXECUTION CONFIRMATION ====")
        print(f"APIs selected : {', '.join(api_names)}")
        print(f"Nº of pairs   : {total_pairs}")
        if args.max_pairs is not None:
            print(f"(Limit via --max-pairs = {args.max_pairs}, shuffle={args.shuffle_pairs})")
        print(f"Results dir    : {results_dir}")
        print("=================================")
        resp = input("Continue executing the inferences? [y/N]: ").strip().lower()

        if resp not in ("y", "yes", "s", "sim"):
            logger.info("Execution cancelled by the user before inferences.")
            print("Execution cancelled.")
            return

    # -------------------------------------------------------------------------
    # 2. DEBUG mode: audit_single_pair for a pair
    # -------------------------------------------------------------------------
    if args.debug_only:
        _debug_audit_inference(args, face_pairs, logger, platform_to_run, prompt_text, system_instruction)
        return

    # -------------------------------------------------------------------------
    # 3. Loop over the selected APIs (standard inference)
    # -------------------------------------------------------------------------
    all_metrics: list[dict[str, Any]] = []
    api_success_dfs: dict[str, DataFrame] = {}
    roc_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for platform_name, tqdm_desc, final_filename, evaluator in platform_to_run:
        logger.info(f"Starting evaluation with API: {platform_name.upper()}")

        final_path = results_dir / final_filename
        if args.reuse_existing and final_path.exists():
            logger.info(f"[REUSE] Flag --reuse-existing is active and file found for {platform_name}: {final_path}. "
                "Skipping inference and reusing existing results.")
            df_results = pd.read_csv(final_path)
        else:
            df_results = evaluator.run_pairs(
                pairs=face_pairs,
                prompt_text=prompt_text,
                system_instruction=system_instruction,
                tqdm_desc=tqdm_desc,
                checkpoint_every=20,
            )
            df_results.to_csv(final_path, index=False)
            logger.info(f"Results file saved to: {final_path}")

        df_success = df_results[df_results["status"] == "success"].copy()
        if df_success.empty:
            logger.warning(f"No pair with status=success for API {platform_name}; metrics will not be calculated.")
            continue

        api_success_dfs[platform_name] = df_success

        # --- metrics + data for ROC ---
        scores, labels = _extract_and_append_metrics(all_metrics, df_success, logger, platform_name)
        unique_labels = np.unique(labels)
        if unique_labels.size < 2:
            logger.warning(f"[ROC] API {platform_name}: only one class present in y_true ({unique_labels.tolist()}); the ROC curve will be ignored for this API.")
        else:
            roc_data[platform_name] = (scores, labels)

        # ----------------- Gráficos -----------------
        if not args.no_plot:
            plot_api_results(platform_name, df_results, plots_dir, logger)

    # -------------------------------------------------------------------------
    # 3.1 Combined confusion matrix chart (all APIs side-by-side)
    # -------------------------------------------------------------------------
    df_all_metrics = pd.DataFrame(all_metrics)
    if not args.no_plot:
        plot_combined_confusion_matrices(df_metrics=df_all_metrics, plots_dir=plots_dir, logger=logger)

        if roc_data:
            roc_path = plots_dir / "mm_committee_all_apis_roc_curves.png"
            plot_roc_curves(results=roc_data, title="ROC Curves – Fraud vs Genuine (All APIs)", save_path=str(roc_path))
            logger.info(f"[PLOTS] ROC curves for all APIs saved at: {roc_path}")
        elif not roc_data:
            logger.info("[PLOTS] ROC curves not generated: no ROC data available (no 'success' pairs).")

    # -------------------------------------------------------------------------
    # 3.2 Committee Analysis (Gemini + OpenAI + Anthropic)
    # -------------------------------------------------------------------------

    df_committee = _extract_committee_dataframe(api_success_dfs, logger)
    if df_committee is None or df_committee.empty:
        logger.info("[COMMITTEE] Committee DataFrame is missing or empty; committee analyses and charts will be skipped.")
    else:
        committee_full_path = results_dir / "committee_full_analysis.csv"
        df_committee.to_csv(committee_full_path, index_label="pair_id")
        logger.info(f"[COMMITTEE] Análise completa do comitê salva em: {committee_full_path}")

        df_polemicos = df_committee[(~df_committee["comite_acertou"]) | (df_committee["divergencia"])].copy()
        committee_hard_cases_path = results_dir / "committee_analise_casos_polemicos.csv"
        df_polemicos.to_csv(committee_hard_cases_path, index_label="pair_id")
        logger.info(f"[COMMITTEE] Casos polêmicos salvos em: {committee_hard_cases_path}")

        committee_metrics = compute_committee_metrics(df_committee=df_committee, logger=logger)
        json_path_cm = results_dir / "committee_metrics.json"
        json_path_cm_safe = to_native(committee_metrics)
        json_path_cm_text = json.dumps(json_path_cm_safe, indent=2)
        json_path_cm.write_text(json_path_cm_text)

        if not args.no_plot:
            plot_committee_results(df_committee=df_committee, plots_dir=plots_dir, logger=logger)

    # -------------------------------------------------------------------------
    # 4. Save consolidated metrics to JSON
    # -------------------------------------------------------------------------

    json_path = results_dir / "metrics.json"
    json_safe = to_native(all_metrics)
    json_text = json.dumps(json_safe, indent=2)
    json_path.write_text(json_text)

    logger.info(f"Consolidated metrics saved in: {json_path}")

    # -------------------------------------------------------------------------
    # 5. General closing message
    # -------------------------------------------------------------------------
    print(f"Execution completed for the APIs: {[api_run[0] for api_run in platform_to_run]}")



if __name__ == "__main__":
    main()