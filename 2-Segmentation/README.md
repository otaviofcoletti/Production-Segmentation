📁 Estrutura do Projeto
binarize_images.py     # Script CLI principal
utils.py               # Classes e interfaces (baseadas em SOLID)
README.md              # Este arquivo


# Gerar máscaras com limiar padrão (150)
python binarize_images.py --input "tiles_treino" --output "masks_treino"

# Gerar máscaras com limiar customizado
python binarize_images.py --input "tiles_teste" --output "masks_teste" --limiar 120
