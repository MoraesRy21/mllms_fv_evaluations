import matplotlib.pyplot as plt
import numpy as np
import random
import cv2

from dataset.pairs.bin_reader import FacePairBinReader
from utils import analytics_tools as at


class FacePairVisualizer:
    """
    Classe responsável por visualizar pares de imagens extraídos de arquivos .bin.
    Consome os dados do FacePairBinReader e garante a formatação estrita de cor
    para evitar anomalias de plotagem no Matplotlib.
    """
    def __init__(self, bin_reader: FacePairBinReader, directory, save_image: bool):
        self.dataset = bin_reader.load_pil_pairs()
        self.bin_filename = bin_reader.get_filename()
        self.directory = directory
        self.save_image = save_image

    def _prepare_for_plot(self, pil_img):
        """
        Força a conversão da imagem PIL para uma matriz Numpy RGB estrita (0-255).
        Isso previne que a engine de plotagem distorça as cores ou aplique mapas
        de cor indesejados.
        """
        return np.array(pil_img.convert('RGB'), dtype=np.uint8)

    def show_pair(self, img1, img2, is_same: bool, title: str = None, image_filename: str = None):
        """
        Plota um único par de imagens lado a lado.
        """

        fig, axes = plt.subplots(1, 2, figsize=(8, 4), facecolor='white')

        # Prepara as imagens com a configuração que funcionou no diagnóstico
        img1_ready = self._prepare_for_plot(img1)
        img2_ready = self._prepare_for_plot(img2)

        axes[0].imshow(img1_ready)
        axes[0].axis('off')

        axes[1].imshow(img2_ready)
        axes[1].axis('off')

        # color = 'green' if is_same else 'red'
        # match_text = "Same person" if is_same else "Different person"
        # main_title = title if title else f"Label: {match_text}"
        #
        # plt.suptitle(main_title, color=color, fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.show()

        if self.save_image:
            at.save_plot(fig, self.directory, image_filename)

    def show_by_id(self, pair_id: int, title:str = None):
        """
        Exibe um par específico com base no seu ID (índice da lista).
        """
        pair_data = self.dataset[pair_id]
        if title == " ":
            image_title = None
        elif title is not None:
            image_title = title
        else:
            image_title = f"Pair ID: {pair_id} | {'Same person' if pair_data['is_same'] else 'different'}"

        try:
            image_filename = self.bin_filename + "_pair_id-" + str(pair_id)
            self.show_pair(
                pair_data['image1'],
                pair_data['image2'],
                pair_data['is_same'],
                title=image_title,
                image_filename=image_filename
            )
        except IndexError:
            print(f"Erro: O pair_id {pair_id} não existe. O dataset tem {len(self.dataset)} pares.")
        except Exception as e:
            print(f"Erro ao visualizar o pair_id {pair_id}: {e}")

    def show_by_list(self, pair_ids: list[int]):
        """
        Itera sobre uma lista de IDs e exibe cada par sequencialmente.
        """
        print(f"Visualizando {len(pair_ids)} pares específicos...")
        for pair_id in pair_ids:
            self.show_by_id(pair_id)

    def show_random_pairs(self, num_pairs: int = 5):
        """
        Busca pares aleatórios e os exibe.
        """
        total_pairs = len(self.dataset)
        if total_pairs == 0:
            print("O dataset está vazio.")
            return

        indices = random.sample(range(total_pairs), min(num_pairs, total_pairs))
        self.show_by_list(indices)


class FacePairDiagnosticVisualizer:
    def __init__(self, bin_reader):
        self.dataset = bin_reader.load_pil_pairs()

    def test_image_decoding(self, pair_id: int = 0):
        if not self.dataset:
            print("Dataset vazio.")
            return

        # Força a conversão para RGB e garante que o tipo seja uint8 (0-255)
        pil_img = self.dataset[pair_id]['image1']
        img_arr = np.array(pil_img.convert('RGB'), dtype=np.uint8)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        # 1. Modo Normal (PIL limpo forçado para uint8)
        axes[0].imshow(img_arr)
        axes[0].set_title("1. PIL RGB (Padrão)")
        axes[0].axis('off')

        # 2. Inversão de Canais (Efeito Smurf)
        axes[1].imshow(img_arr[:, :, ::-1])
        axes[1].set_title("2. Inversão para BGR")
        axes[1].axis('off')

        # 3. Inversão Matemática (Raio-X)
        axes[2].imshow(255 - img_arr)
        axes[2].set_title("3. Invertida (Raio-X)")
        axes[2].axis('off')

        # 4. Decodificação agressiva do OpenCV
        img_cv2 = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        axes[3].imshow(img_cv2)
        axes[3].set_title("4. OpenCV BGR2RGB")
        axes[3].axis('off')

        plt.suptitle(f"Diagnóstico de Cor - Pair ID: {pair_id}", fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.show()