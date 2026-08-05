"""
ViT-based face embedder using a HuggingFace checkpoint fine-tuned for face recognition.

Default checkpoint: jayanta/google-vit-base-patch16-224-finetuned-face-recognition
The CLS token from the last hidden state is used as the face embedding.
"""
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, ViTModel

from .base import FaceEmbedder

# google/vit-base-patch16-224 — general-purpose ViT (ImageNet-21k pretrain).
# For a face-specific ViT, replace with a model fine-tuned with ArcFace/AdaFace loss,
# e.g. one available on HuggingFace Hub (requires HF_TOKEN if private).
_DEFAULT_MODEL = "google/vit-base-patch16-224"


class ViTFaceEmbedder(FaceEmbedder):
    def __init__(self, model_id: str = _DEFAULT_MODEL, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.extractor = AutoImageProcessor.from_pretrained(model_id)
        # Load backbone only (no classification head) to get raw embeddings
        self.model = ViTModel.from_pretrained(model_id).to(self.device).eval()
        self._name = f"ViT ({model_id.split('/')[-1]})"

    @property
    def name(self) -> str:
        return self._name

    @property
    def short_name(self) -> str:
        return "ViT-face"

    def preprocess(self, image: Image.Image) -> dict:
        image = image.convert("RGB")
        return self.extractor(images=image, return_tensors="pt")

    @torch.no_grad()
    def get_embedding(self, image: Image.Image) -> np.ndarray:
        inputs = self.preprocess(image)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        # CLS token → embedding
        cls = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
        norm = np.linalg.norm(cls)
        return cls / (norm + 1e-8)
