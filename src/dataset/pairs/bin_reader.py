import pickle
import io
import base64
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Tuple


class FacePairBinReader:
    """
    Classe agnóstica para leitura de arquivos .bin de pares de faces
    (gerados pelo FacePairGenerator). Prepara os dados brutos para
    diferentes interfaces de consumo (PIL, Base64, etc.).
    """

    def __init__(self, bin_path: str | Path):
        self.bin_path = Path(bin_path)
        if not self.bin_path.exists():
            raise FileNotFoundError(f"Arquivo binário não encontrado: {self.bin_path}")
        self.filename = self.bin_path.stem

    def get_filename(self) -> str:
        return self.filename

    def _bytes_to_pil(self, img_bytes: bytes) -> Image.Image:
        """Converte os bytes encodados pelo OpenCV para um objeto PIL Image."""
        return Image.open(io.BytesIO(img_bytes))

    def _bytes_to_base64(self, img_bytes: bytes) -> str:
        """Converte os bytes encodados para string Base64."""
        return base64.b64encode(img_bytes).decode('utf-8')

    def load_pil_pairs(self) -> List[Dict[str, Any]]:
        """
        Lê o arquivo .bin e retorna os pares com imagens no formato PIL.Image.
        Ideal para consumo direto em SDKs como google-generativeai.
        """
        with open(self.bin_path, 'rb') as f:
            bins, issame_list = pickle.load(f)

        dataset = []
        for i in range(len(issame_list)):
            img1_bytes = bins[2 * i]
            img2_bytes = bins[2 * i + 1]
            is_same = issame_list[i]

            dataset.append({
                "pair_id": i,
                "image1": self._bytes_to_pil(img1_bytes),
                "image2": self._bytes_to_pil(img2_bytes),
                "is_same": is_same
            })

        return dataset

    def load_base64_pairs(self) -> List[Dict[str, Any]]:
        """
        Lê o arquivo .bin e retorna os pares com imagens em Base64.
        Ideal para serialização em JSON e requisições HTTP (ex: APIs REST).
        """
        with open(self.bin_path, 'rb') as f:
            bins, issame_list = pickle.load(f)

        dataset = []
        for i in range(len(issame_list)):
            img1_bytes = bins[2 * i]
            img2_bytes = bins[2 * i + 1]
            is_same = issame_list[i]

            dataset.append({
                "pair_id": i,
                "image1_b64": self._bytes_to_base64(img1_bytes),
                "image2_b64": self._bytes_to_base64(img2_bytes),
                "is_same": is_same
            })

        return dataset


def load_pairs_from_bin(bin_path: str | Path) -> List[Tuple[Image.Image, Image.Image, int]]:
    """
    Carrega pares de faces a partir de um arquivo .bin gerado pelo FacePairGenerator.

    Retorna:
        lista de (img1, img2, is_same), onde:
          - img1, img2: PIL.Image.Image
          - is_same: int ∈ {0, 1}
    """
    reader = FacePairBinReader(bin_path)
    dataset = reader.load_pil_pairs()

    pairs: List[Tuple[Image.Image, Image.Image, int]] = []
    for item in dataset:
        img1 = item["image1"]
        img2 = item["image2"]
        label = int(item["is_same"])
        pairs.append((img1, img2, label))

    return pairs