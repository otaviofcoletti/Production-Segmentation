import rasterio
from rasterio.windows import Window
from rasterio.merge import merge
from rasterio.enums import Resampling
import os
import re
import numpy as np
import os
import re
import cv2
import numpy as np
import rasterio
from rasterio.transform import from_origin
from glob import glob


def crop_geotiff(input_tif, output_dir, tile_size=1024, overlap=0.2):
    os.makedirs(output_dir, exist_ok=True)
    with rasterio.open(input_tif) as src:
        w, h = src.width, src.height
        stride = int(tile_size * (1 - overlap))
        count = 0

        for y in range(0, h - tile_size + 1, stride):
            for x in range(0, w - tile_size + 1, stride):
                window = Window(x, y, tile_size, tile_size)
                transform = src.window_transform(window)
                tile = src.read(window=window)
                profile = src.profile
                profile.update({
                    'height': tile_size,
                    'width': tile_size,
                    'transform': transform
                })
                tile_path = os.path.join(output_dir, f"tile_{x}_{y}.tif")
                with rasterio.open(tile_path, 'w', **profile) as dst:
                    dst.write(tile)
                count += 1

        print(f"✅ {count} tiles gerados em {output_dir}")


import os
import re
import cv2
import numpy as np
import rasterio
from rasterio.transform import from_origin

def reconstruir_geotiff(tiles_dir, output_tif, tile_size=1024, overlap=0.2):
    """
    Reconstrói um único GeoTIFF a partir dos tiles segmentados,
    seguindo a mesma lógica de coordenadas usada na função crop_geotiff().

    Args:
        tiles_dir (str): Diretório contendo os tiles segmentados (nomes tipo tile_X_Y.tif)
        output_tif (str): Caminho de saída do mosaico final
        tile_size (int): Tamanho dos tiles originais (em pixels)
        overlap (float): Sobreposição usada no recorte original (ex: 0.2)
    """
    # Lista de tiles
    tile_files = [f for f in os.listdir(tiles_dir) if f.endswith((".tif", ".tiff"))]
    if not tile_files:
        raise ValueError("❌ Nenhum arquivo .tif encontrado no diretório informado.")

    # Extrai as coordenadas (x, y) do nome do arquivo
    coords = []
    for f in tile_files:
        match = re.search(r"tile_(\d+)_(\d+)", f)
        if match:
            x, y = map(int, match.groups())
            coords.append((x, y, os.path.join(tiles_dir, f)))
    if not coords:
        raise ValueError("❌ Nenhum nome de arquivo no formato tile_X_Y.tif foi encontrado.")

    # Determina a largura/altura total
    stride = int(tile_size * (1 - overlap))
    max_x = max(x for x, _, _ in coords)
    max_y = max(y for _, y, _ in coords)
    total_w = max_x + tile_size
    total_h = max_y + tile_size

    # Abre o primeiro tile para obter metadados
    with rasterio.open(coords[0][2]) as src:
        profile = src.profile.copy()
        dtype = src.dtypes[0]
        count = src.count
        transform0 = src.transform

    # Calcula o novo transform (assumindo origem no tile_0_0)
    xres = transform0.a
    yres = -transform0.e
    x0, y0 = transform0.c, transform0.f
    transform = from_origin(x0, y0, xres, yres)

    # Cria uma matriz vazia para o mosaico
    mosaic = np.zeros((count, total_h, total_w), dtype=dtype)

    print(f"🧩 Reconstruindo mosaico de {total_w}x{total_h}px a partir de {len(coords)} tiles...")

    # Preenche o mosaico
    for x, y, path in coords:
        with rasterio.open(path) as src:
            tile = src.read()
            h, w = tile.shape[1], tile.shape[2]
            mosaic[:, y:y+h, x:x+w] = tile

    # Atualiza o perfil
    profile.update({
        "height": total_h,
        "width": total_w,
        "transform": transform,
        "compress": "lzw"
    })

    # Salva o mosaico
    with rasterio.open(output_tif, "w", **profile) as dst:
        dst.write(mosaic)

    print(f"✅ Mosaico reconstruído salvo em: {output_tif}")


def dividir_treino_teste_geotiff(input_tif, output_train, output_test, eixo="vertical"):
    """
    Divide um GeoTIFF em duas metades (treino e teste),
    preservando os metadados espaciais (CRS, transform, etc.).

    Parâmetros:
    - input_tif: caminho do GeoTIFF de entrada
    - output_train: caminho de saída da metade de treino
    - output_test: caminho de saída da metade de teste
    - eixo: "vertical" (esquerda/direita) ou "horizontal" (superior/inferior)
    """

    with rasterio.open(input_tif) as src:
        width, height = src.width, src.height
        meta = src.meta.copy()

        if eixo == "vertical":
            # Divide em esquerda (train) e direita (test)
            mid_x = width // 2
            window_train = Window(0, 0, mid_x, height)
            window_test = Window(mid_x, 0, width - mid_x, height)
        elif eixo == "horizontal":
            # Divide em superior (train) e inferior (test)
            mid_y = height // 2
            window_train = Window(0, 0, width, mid_y)
            window_test = Window(0, mid_y, width, height - mid_y)
        else:
            raise ValueError("O parâmetro 'eixo' deve ser 'vertical' ou 'horizontal'.")

        # Extrai dados e metadados de cada metade
        for nome, window, output_path in [
            ("Treino", window_train, output_train),
            ("Teste", window_test, output_test),
        ]:
            transform = src.window_transform(window)
            meta.update({
                "height": int(window.height),
                "width": int(window.width),
                "transform": transform
            })

            data = src.read(window=window)
            with rasterio.open(output_path, "w", **meta) as dst:
                dst.write(data)
            print(f"✅ {nome} salvo em: {output_path}")

    print("🧩 Divisão concluída — arquivos prontos para gerar tiles.")

import numpy as np
import cv2
import rasterio
import os

import numpy as np
import cv2
import rasterio
import os

def sobrepor_mascara_verde(imagem_path, mascara_path, output_path, alpha=0.4):
    """
    Sobrepõe uma máscara binária sobre a imagem original, colorindo em verde as áreas segmentadas.
    Suporta imagens RGB e RGBA.
    """
    # Lê a imagem original
    with rasterio.open(imagem_path) as src:
        img = src.read()
        img = np.moveaxis(img, 0, -1)  # (C,H,W) -> (H,W,C)
        profile = src.profile

    # Lê a máscara e garante o mesmo tamanho
    with rasterio.open(mascara_path) as src:
        mask = src.read(1)

    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Cria máscara binária
    mask_bin = (mask > 127).astype(np.uint8)

    # Normaliza a imagem
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    # Se a imagem tiver 4 canais (RGBA), ignora o alfa para colorir
    if img.shape[2] == 4:
        rgb = img[:, :, :3].copy()
        alpha_channel = img[:, :, 3]
    else:
        rgb = img.copy()
        alpha_channel = None

    # Cria camada verde
    green_layer = np.zeros_like(rgb)
    green_layer[:, :, 1] = 255

    # Aplica a sobreposição verde
    blended = rgb.copy()
    blended[mask_bin == 1] = cv2.addWeighted(
        rgb[mask_bin == 1], 1 - alpha,
        green_layer[mask_bin == 1], alpha,
        0
    )

    # Reanexa o canal alfa se existir
    if alpha_channel is not None:
        blended = np.dstack((blended, alpha_channel))

    # Atualiza perfil e salva
    blended_raster = np.moveaxis(blended, -1, 0)  # (H,W,C) -> (C,H,W)
    profile.update({
        "count": blended_raster.shape[0],
        "dtype": 'uint8'
    })

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(blended_raster)

    print(f"✅ Máscara sobreposta salva em: {output_path}")


