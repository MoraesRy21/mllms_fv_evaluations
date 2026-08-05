from abc import ABC, abstractmethod

import numpy as np
from PIL import Image


class FaceEmbedder(ABC):
    """Abstract base for face embedding models used in verification."""

    @abstractmethod
    def get_embedding(self, image: Image.Image) -> np.ndarray:
        """Return an L2-normalized 512-d embedding for a (pre-cropped) face image."""
        ...

    @abstractmethod
    def preprocess(self, image: Image.Image) -> np.ndarray:
        """Convert PIL image to model-ready tensor/array (before forward pass)."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    def similarity(self, img1: Image.Image, img2: Image.Image) -> float:
        """Cosine similarity ∈ [-1, 1]. Both embeddings are L2-normalized, so dot = cosine."""
        e1 = self.get_embedding(img1)
        e2 = self.get_embedding(img2)
        return float(np.dot(e1, e2))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
