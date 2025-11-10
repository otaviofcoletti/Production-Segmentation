# OrthoMosaic Tools — Tiling, Reconstrução e Divisão Treino/Teste

Este módulo fornece utilitários em Python para manipulação de **imagens GeoTIFF** de grandes dimensões, permitindo:
- Gerar *tiles* (recortes) para processamento por modelos de deep learning;
- Dividir uma imagem ortomosaico em conjuntos de **treino e teste**;
- Reconstruir o mosaico completo a partir de tiles segmentados.

---

## Estrutura do projeto

Production-Segmentation
└── 1-Tiling/
├── data/ Onde estão as imagens para segmentar
├── orthomosaic.py executar o tiling e divisão
├── utils.py  Funções auxiliares (tiling, reconstrução, split)
├── output_dataset/  Onde serão salvos os resultados

Parâmetros opcionais:

--tile-size: tamanho dos tiles (padrão: 1024)

--overlap: sobreposição percentual entre tiles (padrão: 0.2)

### 1. Exemplo para transformar uma imagem completa em um único dataset:

python main.py --input dados/mapa.tif --output output_tiles --tile-size 512 --overlap 0.0

### 2. Exemplo para uma imagem grande virar um conjunto de treino e teste

Se você quiser criar subconjuntos de treinamento e teste, use a flag --train e como quer dividir a imagem, com um corte vertical ou horizontal.

Vertical (divide esquerda/direita)
python main.py --input dados/mapa.tif --output output_dataset --train vertical

Horizontal (divide superior/inferior)
python main.py --input dados/mapa.tif --output output_dataset --train horizontal