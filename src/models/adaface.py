"""
AdaFace embedder.

Backbone: IResNet-50 (same architecture used by InsightFace/ArcFace family).
Weights:  minchul/adaface-ir50-ms1mv2  on HuggingFace Hub.

Checkpoint format (saved with PyTorch Lightning):
  {'state_dict': {'model.<layer>': tensor, 'head.<layer>': tensor, ...}}
  → extract only 'model.*' keys and strip the 'model.' prefix.

Input:  112×112 RGB, normalized to [-1, 1] (mean=0.5, std=0.5 per channel).
Output: 512-d L2-normalized embedding.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .base import FaceEmbedder

# ──────────────────────────────────────────────
# IResNet architecture (InsightFace / AdaFace)
# ──────────────────────────────────────────────

def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)


def _conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False)


class _IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-5)
        self.conv1 = _conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-5)
        self.prelu = nn.PReLU(planes)
        self.conv2 = _conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-5)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return out + identity


class _IResNet(nn.Module):
    """IResNet backbone used in AdaFace / ArcFace."""

    def __init__(self, layers: list[int], num_features: int = 512, dropout: float = 0.0):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5)
        self.prelu = nn.PReLU(64)
        self.layer1 = self._make_layer(64, layers[0], stride=2)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.bn2 = nn.BatchNorm2d(512, eps=1e-5)
        self.dropout = nn.Dropout(p=dropout)
        # 112×112 input → 7×7 feature map after 4× stride-2 layers
        self.fc = nn.Linear(512 * 7 * 7, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-5)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes, stride),
                nn.BatchNorm2d(planes, eps=1e-5),
            )
        layers = [_IBasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(_IBasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prelu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)
        return x


def _iresnet50(**kw) -> _IResNet:
    return _IResNet([3, 4, 6, 3], **kw)


def _iresnet100(**kw) -> _IResNet:
    return _IResNet([3, 13, 30, 3], **kw)


# ──────────────────────────────────────────────
# AdaFace embedder
# ──────────────────────────────────────────────

_HF_MODELS = {
    "ir50-ms1mv2": ("minchul/adaface-ir50-ms1mv2", "adaface_ir50_ms1mv2.ckpt", _iresnet50),
    "ir100-ms1mv3": ("minchul/adaface-ir100-ms1mv3", "adaface_ir100_ms1mv3.ckpt", _iresnet100),
}

# InsightFace antelopev2 (IResNet-100, Glint360K) — fallback when HF auth is unavailable
_ANTELOPE_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip"
_ANTELOPE_ROOT = Path.home() / ".insightface" / "models" / "antelopev2"
_ANTELOPE_DIR = _ANTELOPE_ROOT / "antelopev2"  # zip extracts into a nested subdir

_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
_STD = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)


def _download_antelope_fallback() -> str:
    """Download InsightFace antelopev2 and return path to glintr100.onnx."""
    import urllib.request, zipfile
    onnx_path = _ANTELOPE_ROOT / "glintr100.onnx"
    if onnx_path.exists():
        return str(onnx_path)
    _ANTELOPE_ROOT.mkdir(parents=True, exist_ok=True)
    zip_path = _ANTELOPE_ROOT / "antelopev2.zip"
    print(f"Downloading antelopev2 (IResNet-100 fallback) from InsightFace…")
    urllib.request.urlretrieve(_ANTELOPE_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(_ANTELOPE_ROOT)
    zip_path.unlink(missing_ok=True)
    # Flatten if zip extracted into a nested subdirectory (antelopev2/antelopev2/*)
    nested = _ANTELOPE_ROOT / "antelopev2"
    if nested.is_dir():
        import shutil
        for f in nested.iterdir():
            shutil.move(str(f), str(_ANTELOPE_ROOT / f.name))
        nested.rmdir()
    return str(onnx_path)


class AdaFaceEmbedder(FaceEmbedder):
    """
    AdaFace embedder with two backends:
      1. PyTorch IResNet — requires HuggingFace authentication (set HF_TOKEN env var)
         to download weights from minchul/adaface-ir50-ms1mv2.
      2. ONNX fallback — InsightFace antelopev2 IResNet-100 (Glint360K), downloaded
         automatically from GitHub Releases when HF auth is not available.
    """

    def __init__(self, variant: str = "ir50-ms1mv2", device: str = "cuda"):
        import os
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._ort_session = None
        self._torch_model = None

        # Try HuggingFace PyTorch weights first
        if variant in _HF_MODELS:
            repo_id, filename, arch_fn = _HF_MODELS[variant]
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
            try:
                ckpt_path = self._hf_download(repo_id, filename, token)
                self._torch_model = arch_fn().to(self.device).eval()
                self._load_torch_weights(ckpt_path)
                self._variant = f"AdaFace PyTorch {variant}"
                return
            except Exception as e:
                if "401" in str(e) or "RepositoryNotFound" in type(e).__name__:
                    print(
                        f"[AdaFace] HuggingFace auth required for {repo_id}.\n"
                        f"  Set HF_TOKEN env var for proper AdaFace weights.\n"
                        f"  Falling back to InsightFace antelopev2 (IResNet-100 / Glint360K)…"
                    )
                else:
                    raise

        # ONNX fallback via InsightFace antelopev2
        import onnxruntime as ort
        from insightface.app import FaceAnalysis

        _download_antelope_fallback()  # ensures the pack is extracted
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        ctx_id = 0 if device == "cuda" else -1

        # Use FaceAnalysis with antelopev2 — handles detection + alignment + recognition
        self._fa = FaceAnalysis(name="antelopev2", providers=providers)
        self._fa.prepare(ctx_id=ctx_id, det_size=(640, 640))
        self._rec_onnx = self._fa.models.get("recognition")
        self._variant = "IResNet-100 / Glint360K (antelopev2 fallback)"

    @staticmethod
    def _hf_download(repo_id: str, filename: str, token) -> str:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=repo_id, filename=filename, token=token)

    def _load_torch_weights(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        raw = ckpt.get("state_dict", ckpt)
        state = {k[6:]: v for k, v in raw.items() if k.startswith("model.")}
        missing, _ = self._torch_model.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"Missing keys when loading AdaFace weights: {missing[:5]}")

    @property
    def name(self) -> str:
        return f"AdaFace ({self._variant})"

    @property
    def short_name(self) -> str:
        return f"adaface"

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """Returns NCHW float32 array normalized to [-1, 1] at 112×112."""
        img = image.convert("RGB").resize((112, 112), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
        return arr.transpose(2, 0, 1)[np.newaxis]  # NCHW

    def get_embedding(self, image: Image.Image) -> np.ndarray:
        if hasattr(self, "_fa"):
            # ONNX path: detect → align → recognize via FaceAnalysis
            import cv2
            img_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
            faces = self._fa.get(img_bgr)
            if faces:
                emb = faces[0].embedding
            else:
                # Fallback: center-crop to 112×112
                h, w = img_bgr.shape[:2]
                c = min(h, w)
                crop = img_bgr[(h - c) // 2:(h + c) // 2, (w - c) // 2:(w + c) // 2]
                crop = cv2.resize(crop, (112, 112))
                emb = self._rec_onnx.get_feat([crop])[0]  # get_feat expects list of BGR HWC images
        else:
            # PyTorch path
            nchw = self.preprocess(image)
            t = torch.from_numpy(nchw).to(self.device)
            with torch.no_grad():
                emb = self._torch_model(t).squeeze(0).cpu().numpy()

        norm = np.linalg.norm(emb)
        return emb / (norm + 1e-8)
