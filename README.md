# 🌤️ Detecção de Nuvens em Imagens do Satélite Sentinel-2 com Redes Neurais Totalmente Convolucionais.

Este repositório contém o código e os modelos treinados utilizados na minha dissertação de mestrado, que teve como objetivo a **detecção automática de nuvens em imagens do satélite Sentinel-2** utilizando **redes neurais totalmente convolucionais (UNet)** com diferentes encoders e o dataset **CloudSen12**.

## 📜 Sobre a Dissertação

O trabalho explora o uso da arquitetura **UNet** com dois encoders distintos:

- [EfficientNet-B1](https://arxiv.org/abs/1905.11946)
- [MobileNet-V2](https://arxiv.org/abs/1801.04381)

Foram treinados **8 modelos** com variações nos seguintes aspectos:

- **Tipo de imagem**: 
  - **L1C** (nível 1C) — imagens sem correção atmosférica.
  - **L2A** (nível 2A) — imagens com correção atmosférica.
- **Bandas utilizadas**:
  - **RGB + NIR** (B04, B03, B02, B08).
  - **Todas as bandas disponíveis**.

---

## ✅ Modelos Treinados

| Modelo ID | Encoder         | Tipo de Imagem | Bandas Utilizadas    |
|-----------|-----------------|----------------|----------------------|
| 1         | EfficientNet-B1 | L1C            | RGB + NIR (4 bandas) |
| 2         | EfficientNet-B1 | L1C            | Todas as bandas     |
| 3         | EfficientNet-B1 | L2A            | RGB + NIR (4 bandas) |
| 4         | EfficientNet-B1 | L2A            | Todas as bandas     |
| 5         | MobileNet-V2    | L1C            | RGB + NIR (4 bandas) |
| 6         | MobileNet-V2    | L1C            | Todas as bandas     |
| 7         | MobileNet-V2    | L2A            | RGB + NIR (4 bandas) |
| 8         | MobileNet-V2    | L2A            | Todas as bandas     |

---

## 🚀 Tecnologias Utilizadas

- **Python**
- **Pytorch**
- **Pytorch-lightning**
- **Segmentations-models-pytorch**
- **UNet com encoders EfficientNet-B1 e MobileNet-V2**
- Manipulação de imagens: **NumPy**, **Rasterio**
- Visualização de dados: **Matplotlib**

---

## 📁 Estrutura do Repositório
├── data/ # Scripts para preparação dos dados

├── models/ # Modelos treinados (ou links para download) 

├── notebooks/ # Notebooks de treinamento e avaliação 

├── utils/ # Funções auxiliares 

├── main.py # Script principal para treino/teste 

└── README.md # Este arquivo

## 📊 Resultados

## 📝 Referências



