import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class PlantSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, mask_suffix="_mask"):
        """
        Dataset de segmentação de plantas.

        Args:
            images_dir (str): Caminho das imagens RGB.
            masks_dir (str): Caminho das máscaras correspondentes.
            transform (albumentations.Compose, opcional): Transformações de aumento de dados.
            mask_suffix (str): Sufixo usado nos arquivos de máscara (ex: "_mask").
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.mask_suffix = mask_suffix

        self.valid_exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

        # Lista apenas arquivos de imagem válidos
        self.images = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith(self.valid_exts)
        ])

        if not self.images:
            raise ValueError(f"❌ Nenhuma imagem válida encontrada em: {images_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)

        # Cria o nome da máscara com sufixo
        base, ext = os.path.splitext(img_name)
        mask_name = f"{base}{self.mask_suffix}.png"  # força extensão .png
        mask_path = os.path.join(self.masks_dir, mask_name)

        # --- Carrega imagem RGB ---
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"❌ Erro ao ler imagem: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0

        # --- Carrega máscara ---
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[⚠️ Aviso] Máscara não encontrada para {img_name}. Criando máscara vazia.")
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        mask = (mask > 127).astype(np.float32)  # binariza (0 ou 1)

        # --- Transforma (Albumentations) ---
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        # --- Formata para PyTorch ---
        if isinstance(image, np.ndarray):
            image = torch.tensor(image).permute(2, 0, 1)  # C, H, W
        if isinstance(mask, np.ndarray):
            mask = torch.tensor(mask).unsqueeze(0)        # 1, H, W

        return image, mask
