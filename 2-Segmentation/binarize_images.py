import argparse
from utils import GLICalculator, ThresholdMaskGenerator, GLIMaskPipeline

def main():
    parser = argparse.ArgumentParser(
        description="Gera máscaras de vegetação baseadas no Green Leaf Index (GLI)."
    )

    parser.add_argument("--input", required=True, help="Diretório contendo as imagens de entrada")
    parser.add_argument("--output", default="output_gli", help="Diretório para salvar as máscaras")
    parser.add_argument("--limiar", type=int, default=150, help="Valor de threshold (0–255) para segmentação")

    args = parser.parse_args()

    processor = GLICalculator()
    mask_generator = ThresholdMaskGenerator(threshold_value=args.limiar)
    pipeline = GLIMaskPipeline(processor, mask_generator)

    pipeline.run(args.input, args.output)


if __name__ == "__main__":
    main()
