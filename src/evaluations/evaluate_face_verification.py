#!/usr/bin/env python3
"""
Full evaluation on Faces in Public Transport pairs.

Usage:
    python evaluate.py                        # all models, 10-fold split
    python evaluate.py --model arcface        # single model
    python evaluate.py --dataset-pair         # waht is the dataset pair to use, lfw, or public_transport
    python evaluate.py --split test           # quick test split (600 pairs)
    python evaluate.py --no-plot              # skip ROC plot
    python evaluate.py --output results/      # save metrics CSV + ROC plot
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
from numpy import ndarray
from pandas import DataFrame

from dataset.pairs.bin_reader import load_pairs_from_bin
from dataset.lfw import compute_similarities, load_lfw_pairs
from evaluations.metrics import compute_metrics, plot_roc_curves, print_results_table, to_native, plot_confusion_matrix_by_threshold
from models import REGISTRY, load_all_models, load_model, FaceEmbedder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate face models on LFW pairs.")
    p.add_argument("--model", choices=list(REGISTRY) + ["all"], default="all")
    p.add_argument("--dataset-pair", choices=["lfw", "public_transport"], default="public_transport")
    p.add_argument("--split", choices=["train", "test", "10fold"], default="10fold")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output", type=Path, default=None, help="Directory to save results")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()

def generate_final_prediction_all_models(df_pairs: DataFrame | None, labels: ndarray, metrics: dict, model: FaceEmbedder, pair_ids: list[int],
                                         scores: ndarray) -> DataFrame:
    # ------------------------------------------------------------------
    # NOVO: construir/atualizar o CSV unificado por par
    # ------------------------------------------------------------------
    # pair_ids veio do carregamento do dataset (real no bin, índice na LFW)
    if len(pair_ids) != len(labels):
        raise ValueError(
            f"pair_ids e labels com tamanhos diferentes: "
            f"{len(pair_ids)} vs {len(labels)}"
        )

    ground_truth_is_same = (1 - labels).astype(bool)

    # Threshold ótimo retornado por compute_metrics
    thr = metrics["Threshold"]

    # Distância alta => pessoa DIFERENTE (fraude) → É FRAUDE
    pred_fraude = scores >= thr

    # Nome de colunas por modelo
    col_score = f"score_similaridade_{model.short_name}"
    col_pred = f"pred_fraude_{model.short_name}"

    if df_pairs is None:
        # Cria o DF base na primeira iteração
        df_pairs = pd.DataFrame(
            {
                "pair_id": pair_ids,
                "ground_truth_is_same": ground_truth_is_same,
                col_score: (1 - scores),
                col_pred: pred_fraude,
            }
        )
    else:
        # Garante que o número de pares bate
        if len(df_pairs) != len(scores):
            raise ValueError(
                f"Número de pares diferente entre modelos: "
                f"{len(df_pairs)} (já no DF) vs {len(scores)} (modelo {model.name})"
            )
        df_pairs[col_score] = (1 - scores)
        df_pairs[col_pred] = pred_fraude
    return df_pairs

def main() -> None:
    args = parse_args()

    output: Path = None
    if args.dataset_pair == "lfw":
        print(f"Loading LFW pairs (split='{args.split}')…")
        pairs = load_lfw_pairs(split=args.split, root=args.data_dir)
        print(f"  → {len(pairs)} pairs ({sum(l for _, _, l in pairs)} genuine / {sum(1 - l for _, _, l in pairs)} impostor)\n")
        if args.output:
            output = args.output
        # LFW não tem pair_id nativo – usamos simplesmente o índice
        pair_ids = list(range(len(pairs)))
    elif args.dataset_pair == "public_transport":
        print(f"Loading Pairs of Faces in Public Transport…")
        root_pairs_dir = "/path/to/pair/directory/"
        # pair_filename = "face_eval_bin_pair-gt_labat-cropped_image-warpe-112x112-len1180.bin" # Alias paper - FCW112-PAIR-SET
        pair_filename = "face_eval_bin_pair-gt_labat-full_image-default_480x640-len1180.bin" # Alias paper - FFIDR-PAIR-SET
        pairs = load_pairs_from_bin(root_pairs_dir + pair_filename)
        pair_ids = list(range(len(pairs)))
        print(f"  → {len(pairs)} pairs ({sum(l for _, _, l in pairs)} genuine / {sum(1 - l for _, _, l in pairs)} impostor)\n")
        print(f"Pair filename: {pair_filename}")
        print(pair_ids)
        if args.output:
            dir_name = pair_filename.replace('face_eval_bin_pair', 'pair').removesuffix('.bin')
            output = args.output / dir_name
    else:
        raise ValueError(f"Invalid dataset_pair: {args.dataset_pair}")

    if args.model == "all":
        models = load_all_models(device=args.device)
    else:
        models = {args.model: load_model(args.model, device=args.device)}

    all_metrics = []
    roc_data = {}
    df_pairs = None

    for name, model in models.items():
        print(f"Evaluating {model.name}…")
        start = time.perf_counter()
        scores, labels = compute_similarities(model, pairs, desc=model.name)
        # --- métricas de tempo ---
        elapsed = time.perf_counter() - start  # em segundos
        inf_time_per_pair_ms = (elapsed / len(pairs)) * 1000.0

        labels = 1 - labels.astype(int) # Fraudes = 1 e Genuinus = 0
        scores = 1.0 - scores.astype(float)
        metrics = compute_metrics(scores, labels, model_name=model.name)
        metrics["InferenceTimeSeconds"] = elapsed
        metrics["InferenceTimePerPairMs"] = inf_time_per_pair_ms
        all_metrics.append(metrics)
        roc_data[model.name] = (scores, labels)
        print(f"  AUC={metrics['AUC']:.4f}  Acc={metrics['Accuracy']:.4f}\n")
        if not args.no_plot:
            plot_confusion_matrix_by_threshold(scores=scores, labels=labels, metrics=metrics, model_name=model.name, output=output,)

        df_pairs = generate_final_prediction_all_models(df_pairs, labels, metrics, model, pair_ids, scores)
        print(df_pairs.head(5))

    print("\n── Results ─────────────────────────────────────────────")
    print_results_table(all_metrics)

    if output:
        output.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(all_metrics)
        csv_path = output / "metrics.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nMetrics saved → {csv_path}")

        if df_pairs is not None:
            pairs_csv_path = output / "pairs_predictions_all_models.csv"
            df_pairs.to_csv(pairs_csv_path, index=False)
            print(f"Per‑pair predictions saved → {pairs_csv_path}")

        json_path = output / "metrics.json"
        json_safe = to_native(all_metrics)
        json_path.write_text(json.dumps(json_safe, indent=2))

    if not args.no_plot:
        save_path = str(output / "roc_curves.png") if output else None
        fig = plot_roc_curves(roc_data, save_path=save_path)
        if save_path:
            print(f"ROC plot saved → {save_path}")
        else:
            import matplotlib.pyplot as plt
            plt.show()



if __name__ == "__main__":
    main()
