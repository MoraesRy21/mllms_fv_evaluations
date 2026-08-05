"""
ArcFace embedder using InsightFace's buffalo_l model pack.

buffalo_l contains:
  - det_10g.onnx  – RetinaFace-based detector
  - w600k_r50.onnx – ArcFace ResNet-50 recognition model (trained on WebFace600K)

For pre-aligned LFW images, face detection is attempted first.
If no face is found, the center-cropped image is fed directly to the recognizer.
"""
import cv2
import numpy as np
from PIL import Image

from .base import FaceEmbedder


class ArcFaceEmbedder(FaceEmbedder):
    def __init__(self, model_name: str = "buffalo_l", device: str = "cuda"):
        # Import here so the rest of the project doesn't break if insightface isn't installed
        import insightface
        from insightface.app import FaceAnalysis

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        ctx_id = 0 if device == "cuda" else -1

        self.app = FaceAnalysis(name=model_name, providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))

        # Keep a direct reference to the recognition sub-model for fallback
        self._rec = self.app.models.get("recognition")

    @property
    def name(self) -> str:
        return "ArcFace (buffalo_l / w600k_r50)"

    @property
    def short_name(self) -> str:
        return "arcface-buffalo_l"

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """Convert PIL → BGR numpy array (insightface convention)."""
        return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)

    def get_embedding(self, image: Image.Image) -> np.ndarray:
        img_bgr = self.preprocess(image)
        faces = self.app.get(img_bgr)

        if faces:
            emb = faces[0].embedding
        else:
            # Fallback: feed center-cropped 112×112 to the recognizer directly
            h, w = img_bgr.shape[:2]
            crop_size = min(h, w)
            y0 = (h - crop_size) // 2
            x0 = (w - crop_size) // 2
            crop = img_bgr[y0 : y0 + crop_size, x0 : x0 + crop_size]
            crop = cv2.resize(crop, (112, 112))
            emb = self._rec.get_feat([crop])[0]  # get_feat expects list of BGR HWC images

        norm = np.linalg.norm(emb)
        return emb / (norm + 1e-8)
