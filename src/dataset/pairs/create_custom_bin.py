import os
import pickle
import sys
from logging import Logger
from typing import Any, Callable

import cv2
import pandas as pd
from tqdm import tqdm
import random
from itertools import combinations

from utils.paths import PathBuilder


class FacePairGenerator:
    def __init__(self, logger: Logger, **generator_params):
        self.logger = logger

        self.max_pairs_per_class = generator_params.get('max_pairs_per_class', 5)
        self.margin = generator_params.get('image_strategy').get('margin', 10)
        self.resize_size = generator_params.get('image_strategy').get('resize_size', (122, 112))
        self.use_full_image = generator_params.get('image_strategy').get('use_full_image', False)
        self.resize_strategy = generator_params.get('image_strategy').get('resize_strategy', None)
        self.use_pivot_for_fraud = generator_params.get('use_pivot_for_fraud', True)
        self.fraud_clean_sample_size = generator_params.get('fraud_clean_sample_size', 8)

        random.seed(generator_params.get('seed', 42))

    def generate(self, df: pd.DataFrame, result_path: PathBuilder, gt_type: str, filtered: bool = False):
        """Gera o binário a partir de um DataFrame em memória."""
        self.logger.info(f"Iniciando geração de pares. Estratégia: {gt_type}")

        pares_para_avaliar = []

        if gt_type == "gt_semantic":
            self._mounting_semantic_pairs(df, pares_para_avaliar)
        elif gt_type == "gt_labat":
            self._mounting_labat_pairs(df, pares_para_avaliar)
        else:
            self.logger.error(f"Estratégia '{gt_type}' desconhecida.")
            raise ValueError(f"Estratégia {gt_type} inválida.")

        bins = []
        issame_list = []

        self.logger.info(f"Processando e encodando {len(pares_para_avaliar)} pares...")
        for img1_info, img2_info, is_same in tqdm(pares_para_avaliar, desc="Encoding Images"):
            try:
                bins.append(self._process_and_encode(img1_info))
                bins.append(self._process_and_encode(img2_info))
                issame_list.append(is_same)
            except Exception as e:
                self.logger.error(f"Erro ao processar par: {e}")
                continue

        pair_length = len(issame_list)

        os.makedirs(os.path.dirname(result_path), exist_ok=True)

        filename = self._get_filename(gt_type, filtered, pair_length)
        file_output = (result_path / filename).path
        with open(file_output, 'wb') as f:
            pickle.dump((bins, issame_list), f, protocol=pickle.HIGHEST_PROTOCOL)

        self.logger.info(f"Dataset binário salvo com sucesso em: {file_output}")
        self.logger.info(f"Estatísticas: {len(issame_list)} pares totais")

    def _get_filename(self, gt_type: str, filtered: bool, pair_length: int):
        filename = "face_eval_bin_pair-" + gt_type
        if self.use_full_image:
            filename += "-full_image"
        else:
            filename += "-cropped_image"

        if self.resize_strategy is not None:
            w, h = self.resize_size
            filename += f"-{self.resize_strategy}-{w}x{h}"
        else:
            filename += "-default_480x640"

        if self.use_pivot_for_fraud:
            filename += "-with_pivot"

        if filtered:
            filename += "-filtered"

        filename += f"-len{pair_length}"

        filename += ".bin"
        return filename


    def _process_and_encode(self, img_info):
        path, x, y, w, h = img_info

        # Trava de segurança para valores nulos
        if pd.isna(x) or pd.isna(y) or pd.isna(w) or pd.isna(h):
            raise ValueError(f"Bounding box corrompido (NaN) para a imagem: {path}")

        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Imagem não encontrada ou corrompida: {path}")

        target_w, target_h = self.resize_size

        if self.use_full_image:
            # None face cropping
            face_crop = img
        else:
            if pd.isna(x) or pd.isna(y) or pd.isna(w) or pd.isna(h):
                raise ValueError(f"Bounding box corrompido (NaN) para a imagem: {path}")

            l, t = max(0, int(x - self.margin)), max(0, int(y - self.margin))
            r, b = min(img.shape[1], int(x + w + self.margin)), min(img.shape[0], int(y + h + self.margin))

            # Face cropping
            face_crop = img[t:b, l:r]

        if face_crop.size == 0:
            raise ValueError(f"Crop resultou em uma imagem vazia para: {path}")

        if self.resize_strategy == 'warpe':
            final_img = cv2.resize(face_crop, self.resize_size)

        elif self.resize_strategy == 'pad_black':
            final_img = self._resize_with_padding(face_crop, target_w, target_h, pad_color=(0, 0, 0))

        elif self.resize_strategy == 'pad_mean':
            mean_color = cv2.mean(img)[:3]
            final_img = self._resize_with_padding(face_crop, target_w, target_h, pad_color=mean_color)

        elif self.resize_strategy == 'center_crop':
            final_img = self._resize_center_crop(face_crop, target_w, target_h)

        else:
            final_img = face_crop

        _, img_encoded = cv2.imencode('.jpg', final_img)
        return img_encoded.tobytes()

    def _resize_with_padding(self, img, target_w, target_h, pad_color):
        """Redimensiona mantendo aspecto e preenche o resto com pad_color."""
        h, w = img.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(img, (new_w, new_h))

        # Calcula pixels faltantes para dividir igualmente (top/bottom, left/right)
        delta_w = target_w - new_w
        delta_h = target_h - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)

    def _resize_center_crop(self, img, target_w, target_h):
        """Redimensiona pelo menor lado e corta o meio do maior lado."""
        h, w = img.shape[:2]
        scale = max(target_w / w, target_h / h) # Diferença crucial: usamos max() aqui
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(img, (new_w, new_h))

        # Encontra o centro
        y_center = new_h // 2
        x_center = new_w // 2

        # Corta a imagem alvo em volta do centro
        t = max(0, y_center - (target_h // 2))
        b = t + target_h
        l = max(0, x_center - (target_w // 2))
        r = l + target_w

        return resized[t:b, l:r]

    def _mounting_semantic_pairs(self, df, pares_para_avaliar: list[Any]):
        max_pares_por_classe = 5  # Para evitar que identidades com muitas fotos explodam o tamanho do arquivo

        # Exemplo: Separar o que é base limpa (para formar pares genuínos)
        # e o que é hard negative (para formar pares de fraude)
        # 1. Separar os dataframes com base no Ground Truth
        df_clean = df[df['is_semantic_outlier'] == False]
        df_fraud = df[df['is_semantic_outlier'] == True]

        # Função de extração com as colunas específicas do Labat
        extract_info = lambda row: (row['path'], row['x_bbox'], row['y_bbox'], row['w_bbox'], row['h_bbox'])

        self.logger.info("Gerando pares genuínos e fraudes...")
        grupos_clean = df_clean.groupby('class_id')
        self.logger.info(f"Base de dados possui {len(grupos_clean)} identidades 'clean'.")

        # Contadores para A, B e C
        total_pares_genuinos = 0
        total_pares_fraude = 0
        total_pares_impostores = 0

        progress_bar = tqdm(grupos_clean, desc='Clean Groups')
        for class_id, grupo in progress_bar:
            # --- A. PARES GENUÍNOS (Mesma pessoa, fotos diferentes) ---
            if len(grupo) >= 2:
                # Pega todas as combinações possíveis de fotos limpas dessa pessoa
                combos_genuinos = list(combinations(grupo.iterrows(), 2))
                # Limita a quantidade para manter o dataset balanceado
                random.shuffle(combos_genuinos)
                for (_, row1), (_, row2) in combos_genuinos[:max_pares_por_classe]:
                    pares_para_avaliar.append((extract_info(row1), extract_info(row2), True))
                    total_pares_genuinos += 1

            # --- B. PARES DE FRAUDE (Dono real vs. Fraudador usando o cartão) ---
            # Verifica se esse class_id tem fraudes registradas
            fraudes_dessa_classe = df_fraud[df_fraud['class_id'] == class_id]
            self.logger.info(f"Class ID {class_id} - Limpas: {len(grupo)}, Fraudes: {len(fraudes_dessa_classe)}")
            if not fraudes_dessa_classe.empty and not grupo.empty:
                if self.use_pivot_for_fraud:
                    # Pega a primeira foto limpa como "foto de cadastro/referência"
                    referencia = grupo.iloc[0]
                    for _, row_fraude in fraudes_dessa_classe.iterrows():
                        pares_para_avaliar.append((extract_info(referencia), extract_info(row_fraude), False))
                        total_pares_fraude += 1
                else:
                    # Sorteia N fotos limpas (limitado ao total de fotos disponíveis para evitar erro)
                    n_samples = min(len(grupo), self.fraud_clean_sample_size)
                    limpas_sorteadas = grupo.sample(n_samples)

                    # Cruza cada foto limpa sorteada com todas as fraudes daquela pessoa
                    for _, r_limpa in limpas_sorteadas.iterrows():
                        for _, r_fraude in fraudes_dessa_classe.iterrows():
                            pares_para_avaliar.append((extract_info(r_limpa), extract_info(r_fraude), False))
                            total_pares_fraude += 1

        self.logger.info(f"[A] Total de pares genuínos (intraclasse) gerados: {total_pares_genuinos}")
        self.logger.info(f"[B] Total de pares de fraude (intraclasse) gerados: {total_pares_fraude}")
        self.logger.info(f"Total de Pares Genuínos e Fraudes (intraclasse) gerados: {len(pares_para_avaliar)}")

        # --- C. PARES IMPOSTORES ALEATÓRIOS (Pessoas diferentes) ---
        self.logger.info("Gerando pares impostores aleatórios...")
        lista_class_ids = list(grupos_clean.groups.keys())
        num_impostores_desejados = len(pares_para_avaliar) // 2  # Para balancear com os genuínos/fraudes
        self.logger.info(f"Gerando {num_impostores_desejados} pares impostores aleatórios para balanceamento.")

        for _ in tqdm(range(num_impostores_desejados)):
            id_a, id_b = random.sample(lista_class_ids, 2)
            foto_a = df_clean[df_clean['class_id'] == id_a].sample(1).iloc[0]
            foto_b = df_clean[df_clean['class_id'] == id_b].sample(1).iloc[0]
            pares_para_avaliar.append((extract_info(foto_a), extract_info(foto_b), False))
            total_pares_impostores += 1

        self.logger.info(f"[C] Total de pares impostores (interclasses) gerados: {total_pares_impostores}")
        self.logger.info(f"Total de pares gerados: {len(pares_para_avaliar)}")
        # Agora 'pares_para_avaliar' está pronto para ser percorrido e empacotado no .bin

    def _mounting_labat_pairs(self, df, pares_para_avaliar):
        colunas_bbox = ['glt_x_bbox', 'glt_y_bbox', 'glt_w_bbox', 'glt_h_bbox']

        # 1. Filtra apenas o que passou pela anotação manual
        self.logger.info("Montando pares do labat...")
        self.logger.info(f"Quantidade de imagens {len(df)}")
        df_labat = df[(df['glt_done'] == True) & (df['glt_is_fraud'] is not None)]
        self.logger.info(f"Filtro de 'df['glt_done'] == True' aplicado, quantidade: {len(df_labat)}")
        df_labat = df_labat.dropna(subset=colunas_bbox)
        self.logger.info(f"Filtro de 'df_labat.dropna(subset=colunas_bbox)' aplicado, quantidade: {len(df_labat)}")

        # Função de extração com as colunas específicas do Labat
        extract_info = lambda row: (row['path'], row['glt_x_bbox'], row['glt_y_bbox'], row['glt_w_bbox'], row['glt_h_bbox'])

        ids_anotados = df_labat['class_id'].unique()
        self.logger.info(f"Base Labat filtrada: {len(ids_anotados)} identidades com anotação manual.")

        # Contadores para A, B e C
        total_pares_genuinos = 0
        total_pares_fraude = 0
        total_pares_impostores = 0

        self.logger.info(f"Gerando Pares Genuínus e Fraudes para {len(ids_anotados)} ids Intraclass")
        progress_bar = tqdm(ids_anotados, desc="Montando Pares Labat")
        for class_id in progress_bar:
            grupo = df_labat[df_labat['class_id'] == class_id]

            # Separa fotos legítimas (limpas) e fraudes daquela identidade
            limpas = grupo[grupo['glt_is_fraud'] == False]
            fraudes = grupo[grupo['glt_is_fraud'] == True]

            # --- A. PARES GENUÍNOS (Intraclasse) ---
            # Qualquer combinação de fotos limpas da mesma pessoa
            if len(limpas) >= 2:
                combos = list(combinations(limpas.iterrows(), 2))
                random.shuffle(combos)
                for (_, r1), (_, r2) in combos[:self.max_pairs_per_class]:
                    pares_para_avaliar.append((extract_info(r1), extract_info(r2), True))
                    total_pares_genuinos += 1

            # --- B. PARES DE FRAUDE (Intraclasse) ---
            # Pessoa vs Fraude com os dados dela
            if not fraudes.empty and not limpas.empty:
                if self.use_pivot_for_fraud:
                    # Como glt_is_pivot indica a foto mais nítida, damos preferência a ela como referência
                    pivos = limpas[limpas['glt_is_pivot'] == True]
                    ref = pivos.iloc[0] if not pivos.empty else limpas.iloc[0]

                    for _, r_fraude in fraudes.iterrows():
                        pares_para_avaliar.append((extract_info(ref), extract_info(r_fraude), False))
                        total_pares_fraude += 1
                else:
                    # Sorteia N fotos limpas (limitado ao total de fotos disponíveis para evitar erro)
                    n_samples = min(len(limpas), self.fraud_clean_sample_size)
                    limpas_sorteadas = limpas.sample(n_samples)

                    # Cruza cada foto limpa sorteada com todas as fraudes daquela pessoa
                    for _, r_limpa in limpas_sorteadas.iterrows():
                        for _, r_fraude in fraudes.iterrows():
                            pares_para_avaliar.append((extract_info(r_limpa), extract_info(r_fraude), False))
                            total_pares_fraude += 1

        # Logs pontuais das etapas A e B
        self.logger.info(f"[A] Total de pares genuínos (intraclasse) gerados no Labat: {total_pares_genuinos}")
        self.logger.info(f"[B] Total de pares de fraude (intraclasse) gerados no Labat: {total_pares_fraude}")
        self.logger.info(f"Total de pares gerados (genuínos + fraudes): {len(pares_para_avaliar)}")

        # --- C. PARES IMPOSTORES (Interclasses) ---
        # Cruzamento entre identidades diferentes para testar Falsos Positivos
        df_limpas_total = df_labat[df_labat['glt_is_fraud'] == False]
        ids_limpos = df_limpas_total['class_id'].unique()

        num_impostores = len(pares_para_avaliar) // 2
        self.logger.info(f"Gerando {num_impostores} pares impostores aleatórios (Interclasses).")

        for _ in range(num_impostores):
            # Escolhe duas identidades diferentes aleatoriamente
            id_a, id_b = random.sample(list(ids_limpos), 2)

            # Pega uma foto legítima de cada classe
            foto_a = df_limpas_total[df_limpas_total['class_id'] == id_a].sample(1).iloc[0]
            foto_b = df_limpas_total[df_limpas_total['class_id'] == id_b].sample(1).iloc[0]

            pares_para_avaliar.append((extract_info(foto_a), extract_info(foto_b), False))
            total_pares_impostores += 1

        self.logger.info(f"[C] Total de pares impostores (interclasses) gerados no Labat: {total_pares_impostores}")
        self.logger.info(f"Total de pares gerados: {len(pares_para_avaliar)}")