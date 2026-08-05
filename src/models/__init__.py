from models.base import FaceEmbedder
from models.vit_face import ViTFaceEmbedder
from models.arcface import ArcFaceEmbedder
from models.adaface import AdaFaceEmbedder

REGISTRY: dict[str, type[FaceEmbedder]] = {
    "vit": ViTFaceEmbedder,
    "arcface": ArcFaceEmbedder,
    "adaface": AdaFaceEmbedder,
}


def load_model(name: str, device: str = "cuda") -> FaceEmbedder:
    """Load a face embedding model by name."""
    if name not in REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(REGISTRY)}")
    return REGISTRY[name](device=device)


def load_all_models(device: str = "cuda") -> dict[str, FaceEmbedder]:
    return {name: cls(device=device) for name, cls in REGISTRY.items()}