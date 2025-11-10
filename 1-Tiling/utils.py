import os
import warnings
import rasterio
from rasterio.windows import Window
from rasterio.merge import merge
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.errors import NotGeoreferencedWarning



# Suprime avisos de arquivos sem georreferência — serão tratados manualmente
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

# ============================================================
# Utils
# ============================================================

def ensure_georeference(src):
    """
    Garante que um dataset Rasterio tenha CRS e transform válidos.
    Se não tiver, aplica valores fictícios.
    """
    crs = src.crs if src.crs is not None else "EPSG:4326"
    transform = src.transform

    # Caso o transform seja identidade ou None, cria um transform genérico
    if transform == rasterio.Affine.identity() or transform is None:
        transform = from_origin(0, 0, 1, 1)

    return crs, transform


def list_tif_files(directory):
    """
    Retorna uma lista de caminhos absolutos para todos os arquivos .tif/.tiff no diretório.
    """
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith((".tif", ".tiff"))
    ]


# ============================================================
# ✂️ 1. Crop de GeoTIFF em Tiles, params opcionais: 
# overlap e size_tile
# Comentários
# Foi feito um tratamento para arquivos sem georreferência serem processados também pois a biblioteca rasterio
# lança erros/warnings nesses casos. Assim, caso o arquivo não possua CRS/transform, são atribuídos valores fictícios.
        
# ============================================================

def crop_geotiff(input_tif, output_dir, tile_size=1024, overlap=0.2):
    """
    Divide um arquivo GeoTIFF em tiles menores, mantendo georreferência.
    Caso o arquivo não seja georreferenciado, aplica CRS/transform fictícios.

    Parâmetros:
    - input_tif: caminho do GeoTIFF de entrada
    - output_dir: pasta onde os tiles serão salvos
    - tile_size: tamanho (em pixels) de cada tile
    - overlap: porcentagem de sobreposição entre tiles (0.0–0.9)
    """
    os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(input_tif) as src:
        crs, transform = ensure_georeference(src)
        w, h = src.width, src.height

        stride = int(tile_size * (1 - overlap))
        count = 0

        for y in range(0, h - tile_size + 1, stride):
            for x in range(0, w - tile_size + 1, stride):
                window = Window(x, y, tile_size, tile_size)
                tile_data = src.read(window=window)

                profile = src.profile.copy()
                profile.update({
                    "height": tile_size,
                    "width": tile_size,
                    "transform": rasterio.windows.transform(window, transform),
                    "crs": crs
                })

                tile_path = os.path.join(output_dir, f"tile_{x}_{y}.tif")
                with rasterio.open(tile_path, "w", **profile) as dst:
                    dst.write(tile_data)

                count += 1

    print(f"✅ {count} tiles gerados em {output_dir}")


# ============================================================
# 🔄 2. FUNÇÃO: RECONSTRUIR MOSAICO A PARTIR DOS TILES
# ============================================================

def reconstruir_tif_segmentado(tiles_dir, output_path):
    """
    Reconstrói um mosaico GeoTIFF a partir de tiles segmentados.
    Se os tiles não tiverem CRS/transform válidos, aplica valores fictícios.
    """
    tile_files = list_tif_files(tiles_dir)
    if not tile_files:
        raise ValueError("Nenhum tile encontrado no diretório especificado!")

    print(f"🧩 Encontrados {len(tile_files)} tiles segmentados. Iniciando merge...")

    src_files_to_mosaic = []
    for fp in tile_files:
        src = rasterio.open(fp)
        crs, transform = ensure_georeference(src)

        profile = src.profile.copy()
        profile.update({"crs": crs, "transform": transform})
        src_files_to_mosaic.append(src)

    # Faz o merge considerando os metadados espaciais
    mosaic, out_transform = merge(src_files_to_mosaic, method="max")

    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform,
        "compress": "lzw"
    })

    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)

    for src in src_files_to_mosaic:
        src.close()

    print(f"✅ Mosaico reconstruído salvo em: {output_path}")


# ============================================================
# 🔀 3. FUNÇÃO: DIVIDIR GEOTIFF EM TREINO E TESTE
# ============================================================

def dividir_treino_teste_geotiff(input_tif, output_train, output_test, eixo="vertical"):
    """
    Divide um GeoTIFF em duas partes (treino e teste),
    preservando metadados espaciais (CRS, transform).

    Parâmetros:
    - input_tif: caminho do GeoTIFF original
    - output_train: saída da metade de treino
    - output_test: saída da metade de teste
    - eixo: 'vertical' (esquerda/direita) ou 'horizontal' (superior/inferior)
    """
    with rasterio.open(input_tif) as src:
        width, height = src.width, src.height
        meta = src.meta.copy()

        if eixo == "vertical":
            mid_x = width // 2
            windows = [
                ("Treino", Window(0, 0, mid_x, height), output_train),
                ("Teste", Window(mid_x, 0, width - mid_x, height), output_test)
            ]
        elif eixo == "horizontal":
            mid_y = height // 2
            windows = [
                ("Treino", Window(0, 0, width, mid_y), output_train),
                ("Teste", Window(0, mid_y, width, height - mid_y), output_test)
            ]
        else:
            raise ValueError("O parâmetro 'eixo' deve ser 'vertical' ou 'horizontal'.")

        for nome, window, output_path in windows:
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

