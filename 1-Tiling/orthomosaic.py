import argparse
import os
from utils import crop_geotiff, dividir_treino_teste_geotiff


def main():
    parser = argparse.ArgumentParser(
        description="🛰️ Utilitário para manipulação de imagens GeoTIFF — tiling e divisão em treino/teste."
    )

    # Argumentos obrigatórios
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Caminho para o arquivo GeoTIFF de entrada."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Diretório ou arquivo de saída, dependendo da operação."
    )

    # Argumentos opcionais
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1024,
        help="Tamanho (em pixels) de cada tile. Padrão = 1024."
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.2,
        help="Porcentagem de sobreposição entre tiles (0.0–0.9). Padrão = 0.2."
    )

    # Modo de treino/teste
    parser.add_argument(
        "--train",
        nargs="?",
        const="vertical",
        choices=["vertical", "horizontal"],
        help=(
            "Divide o GeoTIFF em treino/teste antes de gerar tiles. "
            "Use '--train vertical' (esquerda/direita) ou '--train horizontal' (superior/inferior). "
            "Se omitido, o script faz apenas o tiling da imagem inteira."
        )
    )

    args = parser.parse_args()


    input_path = args.input
    output_dir = args.output
    tile_size = args.tile_size
    overlap = args.overlap
    train_mode = args.train

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {input_path}")

    os.makedirs(output_dir, exist_ok=True)

    if train_mode:
        # Faz a divisão entre treino e teste
        print(f"📚 Dividindo {input_path} em treino e teste ({train_mode})...")

        train_path = os.path.join(output_dir, "train.tif")
        test_path = os.path.join(output_dir, "test.tif")

        dividir_treino_teste_geotiff(input_path, train_path, test_path, eixo=train_mode)

        # Gera tiles para ambos
        print("🧩 Gerando tiles da base de treino...")
        crop_geotiff(train_path, os.path.join(output_dir, "tiles_train"), tile_size, overlap)

        print("🧩 Gerando tiles da base de teste...")
        crop_geotiff(test_path, os.path.join(output_dir, "tiles_test"), tile_size, overlap)

        print("✅ Processo concluído: treino e teste gerados em", output_dir)

    else:
        # Apenas gera tiles da imagem inteira
        print(f"✂️ Gerando tiles do GeoTIFF completo: {input_path}")
        crop_geotiff(input_path, output_dir, tile_size, overlap)
        print(f"✅ Tiles salvos em: {output_dir}")



if __name__ == "__main__":
    main()
