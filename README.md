# Production-Segmentation
Pipeline de segmentação de plantações


Neste repositório temos a seguinte organização:

No diretório 1-Tiling temos a quebra das imagens grandes em blocos para facilitar um posterior processamento por redes neurais, onde há o cumprimento da tarefa 1.

No diretório 2-Segmentation, temos o módulo de segmentação das imagens e criação do groundtruth de mascaras binarizadas, nesse projeto utilizamos o GLI por conta de alguns testes que fizemos foi o que teve melhor desempenho analisando visualmente, nele está o cumprimento da tarefa 2.

No diretório 3-Neural_Network há os arquivos relacionados ao treino e inferência do modelo de segmentação, onde há o cumprimento das tarefas 3 e 4.

Na pasta 4-Evaluation está a avaliação do modelo e um readme com os resultados que tivemos ao realizar a segmentação de metade da imagem que não foi usada no treino e também de uma imagem diferente.

## Comandos rápidos para utilização do repositório
### Criar venv e instalar dependencias
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Pipeline de treinamento

#### Gerar tiles

```
cd 1-Tiling
python orthomosaic.py --input data\plantacao0.tif --output tiles_output --train vertical
```

#### Gerar máscaras GLI
```
cd 2-Segmentation
python binarize_images.py --input 1-Tiling\tiles_output\tiles_train --output masks_train
```

#### Treinar a rede
```
python train_model.py --rgb 1-Tiling\tiles_output\tiles_train --groundtruth masks_train
```

#### Executar inferência
```
python infer_model.py --model runs/unet_best.pth --input tiles_test --output masks_pred
```

### Pipeline de inferência

```
cd 1-Tiling
python orthomosaic.py --input data\plantacao0.tif --output tiles_output 
```

#### Gerar máscaras GLI
```
cd 2-Segmentation
python binarize_images.py --input 1-Tiling\tiles_output --output masks
```

#### Executar inferência
```
python infer_model.py --model runs/unet_best.pth --input tiles_output --output masks_pred
```