# Módulo de Segmentação – Green Leaf Index (GLI)

Este módulo realiza a geração automática de máscaras de vegetação a partir de imagens RGB, utilizando o Green Leaf Index (GLI) — um índice espectral que destaca regiões verdes (folhas, plantas, grama etc.) com base nas intensidades dos canais de cor.

Ele serve como etapa de pré-processamento para sistemas de visão computacional ou redes neurais de segmentação, permitindo criar máscaras binárias precisas sem necessidade de rotulação manual.

## Estrutura do Projeto
binarize_images.py     # Script CLI principal
utils.py               # Classes e interfaces (baseadas em SOLID)
README.md              # Este arquivo


### Gerar máscaras com limiar padrão (150)

python binarize_images.py --input caminho_das_imagens --output masks_train

### Gerar máscaras com limiar customizado
python binarize_images.py --input caminho_das_imagens --output masks_train --limiar 150


#### O limiar é qual o valor do pixel gerado pelo GLI que será considerado como vegetação, em nossos testes 150 teve um bom resultado