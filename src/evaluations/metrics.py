"""
Face verification metrics:
  - AUC (Area Under the ROC Curve)
  - Best accuracy (threshold sweep)
  - TAR @ FAR=0.1%, 1%, 10%
  - ROC curve plot (single or multi-model)
"""
from __future__ import annotations

from logging import Logger
from pathlib import Path

import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, confusion_matrix


def tar_at_far(
        scores: np.ndarray,
        labels: np.ndarray,
        far_targets: tuple[float, ...] = (0.001, 0.01, 0.1),
) -> dict[str, float]:
    """True Accept Rate at given False Accept Rate thresholds."""
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    result = {}
    for target in far_targets:
        # Largest TPR where FPR ≤ target
        idx = np.searchsorted(fpr, target, side="right") - 1
        idx = max(0, min(idx, len(tpr) - 1))
        result[f"TAR@FAR={target:.1%}"] = float(tpr[idx])
    return result


def best_accuracy(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Sweep thresholds and return (best_acc, best_threshold)."""
    thresholds = np.linspace(scores.min(), scores.max(), 400)
    best_acc, best_thr = 0.0, 0.0
    for thr in thresholds:
        preds = (scores >= thr).astype(int)
        acc = float((preds == labels).mean())
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    return best_acc, best_thr


def compute_metrics(
        scores: np.ndarray,
        labels: np.ndarray,
        model_name: str = "",
) -> dict:
    """Return a dict with AUC, best accuracy, threshold, and TAR@FAR values."""
    auc = roc_auc_score(labels, scores)
    acc, thr = best_accuracy(scores, labels)
    tar_far = tar_at_far(scores, labels)

    y_true = labels.astype(int)
    y_pred = (scores >= thr).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Classe positiva = 1 (mesma pessoa / genuíno) – ajuste se sua convenção for outra
    prec_den = tp + fp
    rec_den = tp + fn
    precision = float(tp / prec_den) if prec_den > 0 else 0.0
    recall = float(tp / rec_den) if rec_den > 0 else 0.0
    f1_den = precision + recall
    f1 = float(2 * precision * recall / f1_den) if f1_den > 0 else 0.0

    return {
        "model": model_name,
        "AUC": round(auc, 4),
        "Accuracy": round(acc, 4),
        "Threshold": round(thr, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        **{k: round(v, 4) for k, v in tar_far.items()},
    }


def plot_roc_curves(
        results: dict[str, tuple[np.ndarray, np.ndarray]],
        title: str = "ROC Curves – LFW Face Verification",
        save_path: str | None = None,
) -> plt.Figure:
    """
    Plot ROC curves for multiple models.

    results: {model_name: (scores, labels)}
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")

    for name, (scores, labels) in results.items():
        fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
        auc = roc_auc_score(labels, scores)
        ax.plot(fpr, tpr, lw=2, label=f"{name}  (AUC={auc:.4f})")

    # Fonte dos rótulos dos eixos
    ax.set_xlabel("False Accept Rate (FAR)", fontsize=16)
    ax.set_ylabel("True Accept Rate (TAR)", fontsize=16)

    # Fonte dos números dos eixos (ticks)
    ax.tick_params(axis="both", labelsize=14)

    # ax.set_title(title)
    ax.legend(loc="lower right", fontsize=14)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def print_results_table(metrics_list: list[dict]) -> None:
    """Pretty-print a comparison table to stdout."""
    if not metrics_list:
        return
    keys = list(metrics_list[0].keys())
    col_w = max(len(k) for k in keys) + 2
    header = "".join(k.ljust(col_w) for k in keys)
    print(header)
    print("-" * len(header))
    for m in metrics_list:
        row = "".join(str(v).ljust(col_w) for v in m.values())
        print(row)

def plot_confusion_matrix_by_threshold(
        scores,
        labels,
        metrics: dict,
        model_name: str,
        output: Path | None,
) -> None:
    """
    Plota a matriz de confusão usando o melhor limiar encontrado em compute_metrics.

    - scores: array de similaridades do modelo
    - labels: array de rótulos (0 = impostor, 1 = genuíno)
    - metrics: dicionário retornado por compute_metrics (usa 'Threshold')
    - model_name: nome amigável do modelo (para título/arquivo)
    - output: diretório base de saída; se None, apenas mostra o plot
    """
    thr = metrics["Threshold"]
    y_true = labels
    y_pred = (scores >= thr).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Impostor", "Genuíno"],
        # display_labels=["Genuíno", "Fraude"],
    )
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix – {model_name} (thr={thr:.4f})")
    plt.tight_layout()

    if output is not None:
        plots_dir = output / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        safe_name = _sanitize_filename(model_name)
        cm_path = plots_dir / f"confusion_matrix_{safe_name}.png"
        plt.savefig(cm_path, dpi=150)
        print(f"Confusion matrix saved → {cm_path}")
        plt.close()
    else:
        plt.show()

# -------------------------------------------------------------------------
# NOVO: métricas genéricas de classificação binária (fraude vs genuíno)
# -------------------------------------------------------------------------

def compute_binary_classification_metrics(
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: np.ndarray | None = None
) -> dict:
    """
    Métricas genéricas de classificação binária.

    Pensado para o comitê FLC (fraude vs genuíno), mas reutilizável.

    Parâmetros
    ----------
    y_true : array-like de shape (n_samples,)
        Labels verdadeiros (0 = classe negativa, 1 = classe positiva, ex: fraude).
    y_pred : array-like de shape (n_samples,)
        Predições binárias (0/1) do modelo para a mesma convenção de rótulo.
    y_score : array-like de shape (n_samples,), opcional
        Score contínuo para a classe positiva (ex.: probabilidade de fraude).
        Se informado, AUC e TAR@FAR são calculados.
    model_name : str
        Nome do modelo / API (para identificação em tabelas).

    Retorna
    -------
    dict com chaves:
      - model
      - Accuracy
      - Precision (classe 1)
      - Recall
      - F1
      - TN, FP, FN, TP
      - (opcional) AUC, TAR@FAR=...
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    # Matriz de confusão (fixa ordem: [0, 1])
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    acc = accuracy_score(y_true, y_pred)

    # Classe positiva = 1 (fraude, por exemplo)
    prec_den = tp + fp
    rec_den = tp + fn
    precision = float(tp / prec_den) if prec_den > 0 else 0.0
    recall = float(tp / rec_den) if rec_den > 0 else 0.0

    f1_den = precision + recall
    f1 = float(2 * precision * recall / f1_den) if f1_den > 0 else 0.0

    metrics = {
        "model": model_name,
        "Accuracy": round(float(acc), 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

    # Opcional: AUC + TAR@FAR, se tivermos score contínuo
    if y_score is not None:
        y_score = np.asarray(y_score).astype(float)
        try:
            auc_val = roc_auc_score(y_true, y_score)
            tar_far = tar_at_far(y_score, y_true)
            metrics["AUC"] = round(float(auc_val), 4)
            metrics.update({k: round(float(v), 4) for k, v in tar_far.items()})
        except ValueError:
            # Casos degenerados: apenas uma classe nos labels, etc.
            metrics["AUC"] = float("nan")

    return metrics

def compute_committee_metrics(df_committee: pd.DataFrame, logger: Logger) -> dict:
    """
    Calcula métricas consolidadas do comitê a partir de um DataFrame já pronto.

    Espera colunas mínimas:
      - ground_truth_fraude (bool)
      - pred_gemini, pred_chatgpt, pred_claude (bool)
      - pred_maioria (bool)
      - unanimidade (bool)
      - divergencia (bool)
      - comite_acertou (bool)

    Metadados opcionais agregados (se existirem):
      - {gemini|chatgpt|claude}_tempo_inferencia_s
      - {gemini|chatgpt|claude}_input_tokens
      - {gemini|chatgpt|claude}_output_tokens
      - {gemini|chatgpt|claude}_total_tokens

    Retorna um dicionário pronto para ser serializado em JSON.
    """
    if df_committee is None or df_committee.empty:
        return {}

    df = df_committee.copy()

    y_true = df["ground_truth_fraude"].astype(int).values

    # Métricas por modelo e pelo comitê (voto da maioria)
    per_model = {}

    acc_gemini = accuracy_score(df_committee["ground_truth_fraude"], df_committee["pred_gemini"])
    acc_chatgpt = accuracy_score(df_committee["ground_truth_fraude"], df_committee["pred_chatgpt"])
    acc_claude = accuracy_score(df_committee["ground_truth_fraude"], df_committee["pred_claude"])
    acc_comite = accuracy_score(df_committee["ground_truth_fraude"], df_committee["pred_maioria"])

    per_model["gemini"] = compute_binary_classification_metrics(
        model_name="Gemini 2.5 Flash",
        y_true=y_true,
        y_pred=df["pred_gemini"].astype(int).values,
        y_score=1 - df["similaridade_gemini"].astype(float).values,
    )
    per_model["chatgpt"] = compute_binary_classification_metrics(
        model_name="ChatGPT 4o-mini",
        y_true=y_true,
        y_pred=df["pred_chatgpt"].astype(int).values,
        y_score=1 - df["similaridade_chatgpt"].astype(float).values,
    )
    per_model["claude"] = compute_binary_classification_metrics(
        model_name="Claude 4.5 Haiku",
        y_true=y_true,
        y_pred=df["pred_claude"].astype(int).values,
        y_score=1 - df["similaridade_claude"].astype(float).values,
    )
    per_model["committee_majority"] = compute_binary_classification_metrics(
        model_name="Committee (Majority Vote)",
        y_true=y_true,
        y_pred=df["pred_maioria"].astype(int).values,
    )

    logger.info("\n[COMMITTEE] Acurácia dos modelos individuais e do comitê:")
    logger.info(f"  Gemini 2.5 Flash        : {per_model['gemini']['Accuracy']:.2%}")
    logger.info(f"  ChatGPT 4o-mini         : {per_model['chatgpt']['Accuracy']:.2%}")
    logger.info(f"  Claude 4.5 Haiku        : {per_model['claude']['Accuracy']:.2%}")
    logger.info(f"  COMITÊ (Voto da Maioria): {per_model['claude']['Accuracy']:.2%}")

    # Estatísticas de concordância
    total_pares = int(len(df))
    unanimidade = df["unanimidade"].astype(bool)
    divergencia = df["divergencia"].astype(bool)
    comite_acertou = df["comite_acertou"].astype(bool)

    unanimous_count = int(unanimidade.sum())
    divergent_count = int(divergencia.sum())

    unanimous_correct = int((unanimidade & comite_acertou).sum())
    unanimous_wrong = int((unanimidade & ~comite_acertou).sum())

    logger.info("\n[COMMITTEE] Análise de concordância:")
    logger.info(f"  Decisões Unânimes               : {unanimous_count}/{total_pares} ({unanimous_count/total_pares:.2%})")
    logger.info(f"  Decisões Divergentes (2 vs 1)   : {divergent_count}/{total_pares} ({divergent_count/total_pares:.2%})")
    logger.info(f"  'Alucinação coletiva' (todos erram): {unanimous_wrong}/{total_pares}")

    concordance = {
        "total_pairs": total_pares,
        "unanimous_count": unanimous_count,
        "divergent_count": divergent_count,
        "unanimous_correct": unanimous_correct,
        "unanimous_wrong": unanimous_wrong,
        "unanimous_ratio": float(unanimidade.mean()),
        "divergent_ratio": float(divergencia.mean()),
    }

    # Agregados de metadados (tempo e tokens) por API, acoplados em per_model
    def aggregate_meta(prefix: str) -> dict:
        meta: dict[str, float | int] = {}
        cols = {
            "tempo_inferencia_s": "inference_time_s",
            "input_tokens": "input_tokens",
            "output_tokens": "output_tokens",
            "total_tokens": "total_tokens",
        }

        for col_suffix, short_name in cols.items():
            col_name = f"{prefix}_{col_suffix}"
            if col_name not in df.columns:
                continue

            series = pd.to_numeric(df[col_name], errors="coerce").dropna()
            if series.empty:
                continue

            meta[short_name] = {
                "mean": float(series.mean()),
                "std": float(series.std()) if len(series) > 1 else 0.0,
                "min": float(series.min()),
                "max": float(series.max()),
                "sum": float(series.sum()),
                "count": int(series.count()),
            }

        return meta

    # Enriquecer apenas os modelos individuais com metadata (comitê não tem tokens/tempo direto)
    for key, prefix in [("gemini", "gemini"), ("chatgpt", "chatgpt"), ("claude", "claude")]:
        meta = aggregate_meta(prefix)
        if meta:
            per_model[key]["metadata"] = meta

    return {
        "per_model": per_model,
        "concordance": concordance,
    }


def plot_api_results(platform_name: str, df_results, plots_dir: Path, logger: Logger) -> None:
    """
    Gera e salva:
      - Matriz de confusão (fraude vs genuíno)
      - Distribuição de confiança (acertos vs erros)
    a partir de um DataFrame de resultados de uma API.
    """
    df_success = df_results[df_results["status"] == "success"].copy()
    if df_success.empty:
        logger.warning(f"[PLOTS] Nenhum resultado 'success' para API {platform_name}; gráficos não serão gerados.")
        return

    # Convenção: y_true_fraude = 1 se fraude (pessoas diferentes)
    df_plot = df_success.copy()
    df_plot["y_true_fraude"] = ~df_plot["ground_truth_is_same"]
    df_plot["y_pred_fraude"] = df_plot["pred_fraude"].astype(bool)
    df_plot["Acertou"] = df_plot["y_true_fraude"] == df_plot["y_pred_fraude"]

    # ----------------- Matriz de confusão -----------------
    cm = confusion_matrix(df_plot["y_true_fraude"], df_plot["y_pred_fraude"])
    labels_true = ["Genuíno (Real)", "Fraude (Real)"]
    labels_pred = ["Genuíno (Pred)", "Fraude (Pred)"]

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[0],
        cbar=False,
        xticklabels=labels_pred,
        yticklabels=labels_true,
    )
    axes[0].set_title(f"Matriz de Confusão – {platform_name}", fontsize=14, pad=15)
    axes[0].set_ylabel("Ground Truth")
    axes[0].set_xlabel("Previsão")

    # ----------------- Distribuição de confiança -----------------
    sns.boxplot(
        data=df_plot,
        x="Acertou",
        y="pred_confianca",
        ax=axes[1],
        palette=["#e74c3c", "#2ecc71"],
    )
    sns.stripplot(
        data=df_plot,
        x="Acertou",
        y="pred_confianca",
        ax=axes[1],
        color="black",
        alpha=0.3,
        jitter=True,
    )
    axes[1].set_title("Nível de Confiança: Acertos vs. Erros", fontsize=14, pad=15)
    axes[1].set_xticklabels(["Errou", "Acertou"])
    axes[1].set_ylabel("Grau de Confiança (0.0 a 1.0)")
    axes[1].set_xlabel("Resultado da Previsão")

    plt.tight_layout()

    plots_dir.mkdir(parents=True, exist_ok=True)
    filename = f"mm_committee_{platform_name}_confusion_mtx_and_confidence.png"
    save_path = plots_dir / filename
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    logger.info(f"[PLOTS] Gráfico gerado para API {platform_name}: {save_path}")

def plot_combined_confusion_matrices(df_metrics: pd.DataFrame, plots_dir: Path, logger: Logger) -> None:
    """
    Gera um único gráfico com as matrizes de confusão das APIs lado a lado,
    usando um DataFrame de métricas consolidadas.

    Parameters
    ----------
    df_metrics : pandas.DataFrame ou similar (com interface de DataFrame)
        Resultado da conversão de `all_metrics` para DataFrame.
        Esperado conter ao menos as colunas: ['model', 'TN', 'FP', 'FN', 'TP'].
    """

    valid_entries: list[tuple[str, np.ndarray]] = []

    for _, row in df_metrics.iterrows():
        model_name = str(row["model"])
        tn = int(row["TN"])
        fp = int(row["FP"])
        fn = int(row["FN"])
        tp = int(row["TP"])

        cm = np.array([[tn, fp],
                       [fn, tp]], dtype=int)
        valid_entries.append((model_name, cm))

    if not valid_entries:
        logger.warning("[PLOTS] Nenhuma linha válida em df_metrics; "
                       "matriz de confusão combinada não será gerada.")
        return

    n_apis = len(valid_entries)
    labels_true = ["Genuíno (Real)", "Fraude (Real)"]
    labels_pred = ["Genuíno (Pred)", "Fraude (Pred)"]

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(
        1,
        n_apis,
        figsize=(6 * n_apis, 6),
        squeeze=False,
    )
    axes_row = axes[0]

    for ax, (platform_name, cm) in zip(axes_row, valid_entries):
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            cbar=False,
            xticklabels=labels_pred,
            yticklabels=labels_true,
        )
        ax.set_title(f"Matriz de Confusão – {platform_name}", fontsize=12, pad=10)
        ax.set_ylabel("Ground Truth")
        ax.set_xlabel("Previsão")

    # Se sobrar algum eixo (teoricamente não sobra, mas por segurança)
    for ax in axes_row[len(valid_entries):]:
        ax.axis("off")

    plt.tight_layout()

    plots_dir.mkdir(parents=True, exist_ok=True)
    filename = "mm_committee_all_apis_confusion_mtx_side_by_side.png"
    save_path = plots_dir / filename
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    logger.info(f"[PLOTS] Matriz de confusão combinada gerada: {save_path}")

def plot_committee_results(df_committee: pd.DataFrame, plots_dir: Path, logger: Logger) -> None:
    """
    Gera os gráficos do comitê (barra de acurácia + pizza de concordância)
    a partir de um DataFrame de comitê já enriquecido.

    Espera colunas:
      - ground_truth_fraude
      - pred_gemini, pred_chatgpt, pred_claude
      - pred_maioria, unanimidade, divergencia, comite_acertou
    """
    if df_committee is None or df_committee.empty:
        logger.warning("[COMMITTEE/PLOTS] DataFrame do comitê vazio; gráficos não serão gerados.")
        return

    df = df_committee.copy()

    # Acurácias individuais e do comitê
    acc_gemini = accuracy_score(df["ground_truth_fraude"], df["pred_gemini"])
    acc_chatgpt = accuracy_score(df["ground_truth_fraude"], df["pred_chatgpt"])
    acc_claude = accuracy_score(df["ground_truth_fraude"], df["pred_claude"])
    acc_comite = accuracy_score(df["ground_truth_fraude"], df["pred_maioria"])

    total_pares = len(df)
    unanimes = int(df["unanimidade"].sum())
    unanimes_errados = int(
        df[(df["unanimidade"] == True) & (df["comite_acertou"] == False)].shape[0]  # noqa: E712
    )

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Gráfico 1: Comparativo de acurácia
    modelos = ["Gemini Flash", "ChatGPT Mini", "Claude Haiku", "Comitê (Maioria)"]
    acuracias = [acc_gemini, acc_chatgpt, acc_claude, acc_comite]
    cores = ["#4285F4", "#10a37f", "#d97757", "#8e44ad"]

    sns.barplot(x=modelos, y=acuracias, palette=cores, ax=axes[0])
    axes[0].set_title("Comparativo de Acurácia na Detecção de Fraude", fontsize=14, pad=15)
    axes[0].set_ylabel("Acurácia")
    axes[0].set_ylim(0, 1.0)
    for i, v in enumerate(acuracias):
        axes[0].text(i, v + 0.02, f"{v:.2%}", ha="center", fontweight="bold")

    # Gráfico 2: Dinâmica de concordância (pizza)
    pizza_labels = [
        "Unânime\ne correto",
        "Divergente\n(mai. acerta)",
        "Divergente\n(mai. erra)",
        "Alucinação coletiva\n(todos erram)",
    ]
    pizza_sizes = [
        len(df[(df["unanimidade"] == True) & (df["comite_acertou"] == True)]),    # noqa: E712
        len(df[(df["divergencia"] == True) & (df["comite_acertou"] == True)]),    # noqa: E712
        len(df[(df["divergencia"] == True) & (df["comite_acertou"] == False)]),   # noqa: E712
        unanimes_errados,
    ]
    pizza_cores = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]

    # Evita divisão por zero se, por acaso, não houver pares
    if total_pares == 0 or sum(pizza_sizes) == 0:
        logger.warning("[COMMITTEE/PLOTS] Nenhos casos válidos para pizza de concordância.")
    else:
        axes[1].pie(
            pizza_sizes,
            labels=pizza_labels,
            autopct="%1.1f%%",
            startangle=140,
            colors=pizza_cores,
            explode=(0.05, 0, 0, 0),
        )
        axes[1].set_title("Dinâmica de Decisão do Comitê", fontsize=14, pad=15)

    plt.tight_layout()
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig_path = plots_dir / "mm_committee_comparative_accuracy_and_concordance.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    logger.info(f"[COMMITTEE/PLOTS] Gráfico de acurácia + concordância salvo em: {fig_path}")

def to_native(obj):
    """
    Converte tipos numpy (float32, int32, etc.) para tipos Python nativos,
    recursivamente em listas/dicts, para serem serializáveis em JSON.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    return obj

def _sanitize_filename(name: str) -> str:
    """
    Torna um nome seguro para uso como parte de um filename.
    Remove caracteres estranhos e troca espaços por underscore.
    """
    # Substitui barras por hífen
    name = name.replace("/", "-")
    # Remove parênteses
    name = name.replace("(", "").replace(")", "")
    # Troca espaços por underscore
    name = name.replace(" ", "_")
    # Remove qualquer coisa muito fora do padrão
    name = re.sub(r"[^a-zA-Z0-9_.\-]", "", name)
    return name