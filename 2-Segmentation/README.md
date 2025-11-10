## 📁 Estrutura do Projeto
binarize_images.py     # Script CLI principal
utils.py               # Classes e interfaces (baseadas em SOLID)
README.md              # Este arquivo



### Gerar máscaras com limiar padrão (150)

python binarize_images.py --input caminho_das_imagens --output masks_train

### Gerar máscaras com limiar customizado
python binarize_images.py --input caminho_das_imagens --output masks_train --limiar 150


#### O limiar é qual o valor do pixel que será considerado como vegetação, em nossos testes 150 teve um bom resultado