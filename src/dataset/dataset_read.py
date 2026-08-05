import os
from pathlib import Path
from typing import Iterable, Union, Optional, Callable, Dict

import cv2
import numpy as np
import pandas as pd
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from IPython.core.display import HTML
from IPython.core.display_functions import display

import utils.analytics_tools as at



def load_raw_dataset(dataset_path: str) -> dict:
    """Loads the dataset from the given path."""

    data: list = []
    for day_folder in sorted(os.listdir(dataset_path)):
        if day_folder.lower() == 'outliers':
            continue

        day_path = os.path.join(dataset_path, day_folder)
        if not os.path.isdir(day_path):
            continue

        for img_file in os.listdir(day_path):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                extracted_info = at.extract_info_image_instance(img_file)
                if extracted_info is not None:
                    data.append({
                        "filename": img_file,
                        "path": os.path.join(dataset_path, str(day_folder), img_file),
                        "day": day_folder,
                        "op": extracted_info[0],
                        "operator_code": extracted_info[1],
                        "vehicle_number": extracted_info[2],
                        "validator_number": extracted_info[3],
                        "buss_line": extracted_info[4],
                        "date_time": extracted_info[5],
                        "class_id": extracted_info[6],
                        "img_index": extracted_info[7]
                    })
    return data


def load_student_dataset(dataset_path: str):
    """
    Loads a "student register" dataset where filenames follow:
        0002522849.jpg
    In this dataset, the filename stem (without extension) is the class_id.

    Returns a list of dicts ready to be converted to a DataFrame.
    """
    data: list[dict] = []

    if not os.path.isdir(dataset_path):
        raise ValueError(f"Invalid dataset_path (not a directory): {dataset_path}")

    valid_exts = ('.jpg', '.jpeg', '.png')

    for entry in sorted(os.scandir(dataset_path), key=lambda e: e.name):
        if not entry.is_file():
            continue

        img_file = entry.name
        if not img_file.lower().endswith(valid_exts):
            continue

        name, ext = os.path.splitext(img_file)

        class_id = name.strip()

        if not class_id or not class_id.isdigit():
            continue

        data.append({
            "filename": img_file,
            "path": entry.path,
            "class_id": class_id # optional numeric view (useful for joins/sorts)
        })

    return data

def _make_date_parser(*, formats_by_col: Optional[Dict[str, str]] = None) -> Callable:
    """
    Cria um date_parser compatível com pd.read_csv que:
    - recebe um array de strings
    - devolve um array de datetimes
    - aplica formatos específicos por coluna, se informados
    """
    # read_csv passa um array de strings; não sabemos o nome da coluna aqui.
    # Então só conseguimos usar um ÚNICO formato ou algo genérico.
    # Para múltiplos formatos por coluna, o caminho é usar parse_dates e depois ajustar.
    #
    # Como workaround, vamos usar apenas um formato "principal" se vier só um,
    # ou fallback totalmente genérico se None.

    # Se só há um formato informado, pegamos ele; senão, deixamos None para ser genérico
    default_format = None
    if formats_by_col and len(set(formats_by_col.values())) == 1:
        default_format = next(iter(formats_by_col.values()))

    def parser(values):
        if default_format is not None:
            return pd.to_datetime(values, format=default_format, errors="coerce")
        return pd.to_datetime(values, errors="coerce")

    return parser


def load_dataset(path: Union[str, Path], *, sep: str = ";", encoding: str = "utf-8",
        date_cols: Iterable[str] = ("date_time", "day"), date_formats: Optional[Dict[str, str]] = None, **read_kwargs) -> pd.DataFrame:
    """
    Lê CSV ou Parquet.
    - Para CSV: usa parse_dates/date_parser para já retornar colunas de data como datetime.
    - Para Parquet: apenas read_parquet (datas normalmente já vêm corretas).

    Parâmetros
    ----------
    path:
        Caminho para o arquivo (.csv ou .parquet).
    sep, encoding:
        Usados apenas para CSV.
    date_cols:
        Quais colunas tratar como datas, se existirem.
    date_formats:
        Formato por coluna, ex:
            {
                "date_time": "%Y-%m-%d %H:%M:%S",
                "day": "%Y-%m-%d",
            }
        Em CSV, só é possível usar UM formato global no date_parser.
        Então, se você passar formatos diferentes por coluna, aqui ele vai
        cair no modo genérico (tenta inferir).
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".parquet", ".pq"}:
        # Normalmente Parquet já traz datetime certo.
        df = pd.read_parquet(path, **read_kwargs)
        return df

    if suffix not in {".csv"}:
        raise ValueError(f"Extensão não suportada: {suffix}")

    # CSV: usar parse_dates
    existing_date_cols = [c for c in date_cols if isinstance(c, str)]
    parse_cols = existing_date_cols if existing_date_cols else None

    # parser = None
    # if parse_cols:
    #     parser = _make_date_parser(formats_by_col=date_formats)

    df = pd.read_csv(
        path,
        sep=sep,
        encoding=encoding,
        parse_dates=parse_cols,
        #date_parser=parser,
        **read_kwargs,
    )

    # Se você passou formatos diferentes por coluna, aqui podemos
    # "refinar" coluna a coluna (sem custo de IO, já está em memória).
    if date_formats:
        for col, fmt in date_formats.items():
            if col in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = pd.to_datetime(df[col].astype(str), format=fmt, errors="coerce")

    return df

def dataset_info(data_frame: pd.DataFrame, show_sample: bool = False):
    """Prints general information about a DataFrame."""

    if show_sample:
        print("Sample of dataset:")
        display(data_frame)
        print()
    print("General Dataset Information:")
    print(f"Total of rows: {len(data_frame)}")
    print(f"Total of images: {len(data_frame)}")
    if 'day' in data_frame.columns :
        print(f"Number of days (folders): {data_frame['day'].nunique()}")
    print(f"Number of classes detected: {data_frame['class_id'].nunique()}\n")




class TransitFaceDataset(Dataset):
    """
    Reader universal para o dataset em CSV, baseado no Ground Truth da Fase 1.
    Alimenta o LinearProbingWrapper e o AdaFace com embeddings de 512d.
    """
    def __init__(self, dataframe, transform=None, subset='clean_train'):
        """
        Args:
            dataframe (pd.DataFrame): Dataframe com os dados do CSV.
            transform (callable, optional): Transformações do torchvision.
            subset (str): 'clean_train' (para treino) ou 'hard_negatives' (para avaliação).
        """
        self.transform = transform
        self.df = dataframe

        # 2. Filtra com base no Ground Truth estabelecido na Fase 1
        if subset == 'clean_train':
            # Pega apenas as imagens consistentes (ignorando as fraudes/outliers)
            # Assumindo que a coluna se chama 'semantic_outlier' ou similar
            self.df = self.df[self.df['is_semantic_outlier'] == False].copy()
        elif subset == 'hard_negatives':
            # Pega apenas as fraudes mapeadas para os testes de robustez
            self.df = self.df[self.df['is_semantic_outlier'] == True].copy()
        else:
            raise ValueError("Subset deve ser 'clean_train' ou 'hard_negatives'.")

        # 3. Mapeamento de Identidades para o AdaFace
        # O AdaFace exige que os labels vão de 0 a num_classes - 1.
        # Os 'class_id' originais do seu CSV podem ter saltos ou ser strings alfanuméricas.
        self.unique_classes = sorted(self.df['class_id'].unique())
        self.class_to_idx = {cls_id: idx for idx, cls_id in enumerate(self.unique_classes)}

        # Essa variável é vital para você instanciar a classe AdaFace corretamente!
        self.num_classes = len(self.unique_classes)
        print(f"Dataset carregado ({subset}): {len(self.df)} imagens | {self.num_classes} identidades.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(row['path'])

        # 1. Carrega a imagem original (480x640)
        image = Image.open(img_path).convert("RGB")

        # 2. Extrai as coordenadas do CSV
        x = int(row['x_bbox'])
        y = int(row['y_bbox'])
        w = int(row['w_bbox'])
        h = int(row['h_bbox'])

        # Opcional mas recomendado: Adicionar uma margem de segurança (padding)
        # para garantir que o queixo e o cabelo não sejam cortados
        margin = 10
        left = max(0, x - margin)
        top = max(0, y - margin)
        right = min(image.width, x + w + margin)
        bottom = min(image.height, y + h + margin)

        # 3. Recorta apenas o rosto
        face_crop = image.crop((left, top, right, bottom))

        # 4. Aplica as transformações (Resize para 112x112 + ToTensor + Normalize)
        if self.transform:
            face_crop = self.transform(face_crop)

        label = self.class_to_idx[row['class_id']]

        return face_crop, label



class SalvadorTransportDataset(Dataset):

    def __init__(self, df, task='recognition', target_size=(640, 640), transform=None, augmentations=None):
        """
        Ponto central único.
        Args:
            df: O dataframe filtrado.
            task: 'recognition' (ArcFace/AdaFace) ou 'detection' (YOLO).
            target_size: Tupla com o tamanho alvo, ex: (112, 112) ou (640, 640).
            transform: Transformações para o reconhecimento facial
            augmentations: Transformações leves adicionais (opcional).
        """
        self.df = df.reset_index(drop=True)
        self.task = task
        self.target_size = target_size
        self.transform = transform
        self.augmentations = augmentations

        # Mapeamento para AdaFace/ArcFace
        self.unique_classes = sorted(self.df['class_id'].unique())
        self.class_to_idx = {cls_id: i for i, cls_id in enumerate(self.unique_classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['path']

        if self.task == 'recognition':
            processed_data, target = self._process_for_recognition(img_path, row)

            # Aplica augmentation só na imagem, se houver
            if self.augmentations:
                processed_data = self.augmentations(processed_data)

            return processed_data, target

        elif self.task == 'detection':
            processed_data, pad_info = self._process_for_detection(img_path, row)

            return {
                'image': processed_data,
                'pad_info': pad_info,
                'original_index': row.name,
                'path': img_path
            }
        else:
            raise ValueError("A task deve ser 'recognition' ou 'detection'.")

    # ==========================================
    # LÓGICAS INTERNAS (Escondidas do Notebook)
    # ==========================================
    def _process_for_recognition_glt(self, img_path, row):
        """Lógica de Crop com o Ground Truth com LabAT"""

        # 1. Carrega a imagem original (480x640)
        image = Image.open(img_path).convert("RGB")

        # 2. Extrai as coordenadas do CSV
        x = int(row['glt_x_bbox'])
        y = int(row['glt_y_bbox'])
        w = int(row['glt_w_bbox'])
        h = int(row['glt_h_bbox'])

        # Opcional mas recomendado: Adicionar uma margem de segurança (padding)
        # para garantir que o queixo e o cabelo não sejam cortados
        margin = 10
        left = max(0, x - margin)
        top = max(0, y - margin)
        right = min(image.width, x + w + margin)
        bottom = min(image.height, y + h + margin)

        # 3. Recorta apenas o rosto
        face_crop = image.crop((left, top, right, bottom))

        # 4. Aplica as transformações (Resize para 112x112 + ToTensor + Normalize)
        if self.transform:
            face_crop = self.transform(face_crop)

        label = self.class_to_idx[row['class_id']]

        return face_crop, label

    def _process_for_recognition(self, img_path, row):
        """Lógica de Crop (Antiga Classe 1)"""

        # 1. Carrega a imagem original (480x640)
        image = Image.open(img_path).convert("RGB")

        # 2. Extrai as coordenadas do CSV
        x = int(row['x_bbox'])
        y = int(row['y_bbox'])
        w = int(row['w_bbox'])
        h = int(row['h_bbox'])

        # Opcional mas recomendado: Adicionar uma margem de segurança (padding)
        # para garantir que o queixo e o cabelo não sejam cortados
        margin = 10
        left = max(0, x - margin)
        top = max(0, y - margin)
        right = min(image.width, x + w + margin)
        bottom = min(image.height, y + h + margin)

        # 3. Recorta apenas o rosto
        face_crop = image.crop((left, top, right, bottom))

        # 4. Aplica as transformações (Resize para 112x112 + ToTensor + Normalize)
        if self.transform:
            face_crop = self.transform(face_crop)

        label = self.class_to_idx[row['class_id']]

        return face_crop, label

    def _process_for_detection(self, img_path, row):
        """Lógica de Letterbox (Antiga Classe 2)"""

        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Erro ao ler imagem: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]
        target_w, target_h = self.target_size

        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        image_resized = cv2.resize(image, (new_w, new_h))
        pad_w, pad_h = (target_w - new_w) // 2, (target_h - new_h) // 2

        # Fundo cinza padrão YOLO
        padded_image = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        padded_image[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = image_resized

        # Aqui, além da imagem, o YOLO precisa do pad_info para reverter caixas depois
        pad_info = {'scale': scale, 'pad_w': pad_w, 'pad_h': pad_h}

        return padded_image, pad_info