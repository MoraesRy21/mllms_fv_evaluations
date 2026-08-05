# MLLMs Face Verification Evaluations

## Evaluating Multimodal Large Language Models for Face Verification in Public Transportatio

This project evaluates the performance of Multimodal Large Language Models (MLLMs) as a committee-based system for face verification, with applications in fraud detection scenarios such as public transport access control.

The codebase supports:
- Face pair dataset generation and preprocessing
- Evaluation of multiple MLLM APIs (Google Gemini, OpenAI GPT-4, Anthropic Claude)
- Committee-based decision aggregation
- Comprehensive metrics and visualization

---

## 🚀 Quick Start

### 1. Environment Setup

**Recommended:** Use a virtual environment with **Python 3.10+**

```bash
# Create conda environment
conda create -n mllms_fv_eval python=3.10
conda activate mllms_fv_eval

# Install dependencies
pip install -r requirements.txt

# Install project in editable mode
pip install -e .
```

**Main dependencies:** `torch`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `opencv-python`, `Pillow`, `insightface`, `pydantic`, `PyYAML`

**Optional (for notebooks):** `ipykernel`, `jupyterlab`

---

## ⚙️ Configuration

### 2.1. Dataset Configuration

Edit `configs/config.bins_pairs.yaml` to configure your dataset and pair generation:

```yaml
dataset:
  csv_path: "path/to/your/ground_truth_dataset.csv"

bins_pairs:
  parameters:
    max_pairs_per_class: 5
    seed: 42
    image_strategy:
      use_full_image: True
      resize_strategy: pad_black
      resize_size: [640, 640]

  results:
    root_path: "results/pairs"
    pair_images_dir: "results/pairs/images"
```

### 2.2. API Keys Configuration

**IMPORTANT:** Add your API keys to `configs/config.fc_pair_eval.yaml`

```yaml
models:
  gemini:
    model_name: "gemini-2.5-flash"
    temperature: 0.0
    api_key: "YOUR_GOOGLE_API_KEY_HERE"  # ← Add your key

  openai:
    model_name: "gpt-4o-mini"
    temperature: 0.0
    api_key: "YOUR_OPENAI_API_KEY_HERE"  # ← Add your key

  anthropic:
    model_name: "claude-haiku-4-5-20251001"
    temperature: 0.0
    api_key: "YOUR_ANTHROPIC_API_KEY_HERE"  # ← Add your key
```

**⚠️ Security Note:** Never commit API keys to version control. Keep them secure and private.

---

## 🎯 Usage

### 3.1. Generate Face Pairs Dataset

```bash
python -m src.dataset.pairs --config configs/config.bins_pairs.yaml
```

This will create a binary file (`.bin`) containing face pairs for evaluation.

### 3.2. Evaluate MLLM Committee

Run evaluation on one or more APIs:

```bash
# Single API
python -m src.evaluations.evaluate_api_mm_committee --apis google

# Multiple APIs
python -m src.evaluations.evaluate_api_mm_committee --apis google openai anthropic

# All configured APIs
python -m src.evaluations.evaluate_api_mm_committee --apis all
```

**Common options:**
- `--config`: Path to config file (default: `configs/config.fc_pair_eval.yaml`)
- `--max-pairs N`: Limit evaluation to first N pairs (useful for testing)
- `--shuffle-pairs`: Randomize pair selection when using `--max-pairs`
- `--no-plot`: Skip plot generation
- `--reuse-existing`: Reuse previous API responses from CSV files

### 3.3. Analysis Notebooks

Jupyter notebooks for result analysis are located in `notebooks/`:

```bash
jupyter lab
```

Available notebooks:
- `01_flc_committee_analyses.ipynb` - Foundation Language Committee analysis
- `02_fvb_committee_analyses.ipynb` - Face Verification Benchmark analysis
- `03_all_committee_analyses.ipynb` - Combined committee analysis
- `05_all_roc_curve.ipynb` - ROC curve visualization

---

## 📊 Evaluation Metrics

The evaluation pipeline computes:
- **Accuracy**, **Precision**, **Recall**, **F1-Score**
- **ROC-AUC** and **ROC Curves**
- **Confusion Matrices**
- **Committee Agreement** (majority vote across MLLMs)

Results are saved to:
- `results/flc_evaluation/` - Individual model results
- `results/flc_fv_aggregation/` - Committee aggregation results

---

## 🔬 Methodology

1. **Dataset Preparation**: Face pairs are generated from ground truth data with configurable filtering and preprocessing strategies.

2. **MLLM Evaluation**: Each configured MLLM API receives face pairs and returns fraud predictions with confidence scores.

3. **Committee Aggregation**: Multiple MLLM responses are aggregated using majority voting to improve robustness.

4. **Metrics Computation**: Standard binary classification metrics are computed and visualized.
