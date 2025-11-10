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


### Resultados

#### Imagem de input

<img width="2315" height="1492" alt="plantacao2" src="https://github.com/user-attachments/assets/7fa6c5ca-3bf4-4055-ad77-2b5a88d0ecb0" />

#### Máscara gerada pela rede neural

<img width="2304" height="1280" alt="reconstrucao_masks_plantacao1" src="https://github.com/user-attachments/assets/c8d1588e-efb7-4d77-9e92-fb7238a9e534" />

#### Sobreposição da máscara na imagem original

<img width="2315" height="1492" alt="sobreposicao0" src="https://github.com/user-attachments/assets/6eaa5018-c08b-4264-be32-793312f6894b" />





