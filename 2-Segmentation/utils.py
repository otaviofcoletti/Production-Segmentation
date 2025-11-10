import cv2
import numpy as np
import os
from abc import ABC, abstractmethod

# Interfaces

class ImageProcessor(ABC):
    """Interface base para processadores de imagem."""
    @abstractmethod
    def process(self, image: np.ndarray) -> np.ndarray:
        pass


class MaskGenerator(ABC):
    """Interface base para geradores de máscara."""
    @abstractmethod
    def generate(self, image: np.ndarray) -> np.ndarray:
        pass


# Implementação do calculo de GLI e geração de máscara

class GLICalculator(ImageProcessor):
    """Calcula o índice GLI (Green Leaf Index) a partir de uma imagem RGB."""

    def process(self, image: np.ndarray) -> np.ndarray:
        if image is None or len(image.shape) < 3:
            raise ValueError("Imagem inválida — é necessário um array RGB.")

        R, G, B = image[:, :, 0].astype(float), image[:, :, 1].astype(float), image[:, :, 2].astype(float)
        GLI = (2 * G - R - B) / (2 * G + R + B + 1e-6)
        GLI_norm = cv2.normalize(GLI, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return GLI_norm


class ThresholdMaskGenerator(MaskGenerator):
    """Gera uma máscara binária a partir do GLI com base em um limiar."""

    def __init__(self, threshold_value: int = 150):
        self.threshold_value = threshold_value

    def generate(self, image: np.ndarray) -> np.ndarray:
        _, mask = cv2.threshold(image, self.threshold_value, 255, cv2.THRESH_BINARY)
        return mask


# Pipeline de execução

class GLIMaskPipeline:
    """Gerencia o fluxo completo de processamento GLI → Máscara."""

    def __init__(self, processor: ImageProcessor, mask_generator: MaskGenerator):
        self.processor = processor
        self.mask_generator = mask_generator

    def run(self, input_dir: str, output_dir: str = "output_gli") -> None:
        os.makedirs(output_dir, exist_ok=True)

        valid_ext = (".tif", ".tiff", ".jpg", ".jpeg", ".png")
        arquivos = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_ext)]

        if not arquivos:
            print("⚠️ Nenhuma imagem válida encontrada no diretório informado.")
            return

        print(f"🧩 {len(arquivos)} imagens encontradas. Iniciando processamento...")

        for nome in arquivos:
            caminho = os.path.join(input_dir, nome)
            img = cv2.imread(caminho)
            if img is None:
                print(f"⚠️ Erro ao ler {nome}, pulando...")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gli = self.processor.process(img_rgb)
            mask = self.mask_generator.generate(gli)

            nome_base = os.path.splitext(nome)[0]
            mask_path = os.path.join(output_dir, f"{nome_base}_mask.png")
            cv2.imwrite(mask_path, mask)

            print(f"✅ {nome} → máscara salva em {mask_path}")

        print(f"\n🎯 Concluído! Máscaras salvas em: {os.path.abspath(output_dir)}")
