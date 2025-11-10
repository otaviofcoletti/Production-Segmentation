# Módulo de Rede Neural — Segmentação de Plantações (U-Net)

Este módulo contém o pipeline de treinamento e inferência de uma U-Net para segmentação de áreas verdes em imagens aéreas ou de drones.

O modelo aprende a prever máscaras binárias a partir de imagens RGB, utilizando dados gerados pelo módulo anterior (tiling e binarization).

## Estrutura do módulo
3-Neural_Network/
│
├── src/
│   ├── train_model.py        # Script de treinamento
│   ├── infer_model.py        # Script de inferência
│   ├── dataset.py            # Classe de dataset customizada
│   ├── model.py              # Implementação da U-Net
│   └── utils.py              # Funções auxiliares (opcional)
│
├── images/                   # (Exemplo) imagens para inferência
├── masks/                    # (Exemplo) máscaras geradas
└── runs/                     # Modelos treinados e checkpoints

## Treinamento do modelo

O script train_model.py permite treinar uma U-Net a partir de um diretório de imagens RGB e suas máscaras binárias correspondentes.

## Exemplo de uso
python train_model.py \
  --rgb "path\to\tiles_train" \
  --groundtruth "path\to\masks_train" \
  --modelpath "runs" \
  --epochs 20 \
  --batch-size 4 \
  --lr 1e-4

## Parâmetros
Parâmetro	Descrição	Padrão
--rgb	Caminho para as imagens de entrada	(obrigatório)
--groundtruth	Caminho para as máscaras correspondentes	(obrigatório)
--modelpath	Diretório onde o modelo será salvo	runs
--epochs	Número de épocas de treino	20
--batch-size	Tamanho do batch	4
--lr	Taxa de aprendizado	1e-4

## Inferência

O script infer_model.py aplica o modelo treinado em novas imagens, gerando máscaras preditas.

## Exemplo de uso
python infer_model.py \
  --model "runs/unet_best.pth" \
  --input "path\to\tiles_test" \
  --output "path\to\predictions"

## Parâmetros
Parâmetro	Descrição	Padrão
--model	Caminho para o modelo .pth treinado	(obrigatório)
--input	Pasta contendo imagens a segmentar	(obrigatório)
--output	Pasta de saída para salvar as máscaras geradas	output_predictions
--threshold	Limiar para binarização das máscaras	0.5