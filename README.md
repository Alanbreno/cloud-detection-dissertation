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

O Dataset utilizado foi o [CloudSen12](https://www.nature.com/articles/s41597-022-01878-2).
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
- **[Pytorch](https://pytorch.org/)**
- **[Pytorch-lightning](https://lightning.ai/docs/pytorch/stable/)**
- **[Segmentations-models-pytorch(SMP)](https://github.com/qubvel-org/segmentation_models.pytorch)**
- **UNet com encoders EfficientNet-B1 e MobileNet-V2 através do SMP**
- Manipulação de imagens: **NumPy**, **Rasterio**
- Visualização de dados: **Matplotlib**

---

## 📁 Estrutura do Repositório
├── scripts_colab_**l1c/l2a**_**4/all**_bands/ # Scripts e Notebooks para treinamento e teste dos modelos. L1C ou L2A é referente ao nível de processamento dos imagens. All ou 4 é referente ao número de bandas usadas na entrada dos modelos.

├── Unet_**all/4**_bands _**l1c/l2a**/ # Logs de treinamento validação e teste, ckeckpoints e modelos salvos. 

└── README.md # Este arquivo

## 📊 Resultados
### Comparação de métricas entre os modelos treinados.

| Modelo                          | L1C Acurácia | L1C IoU | L1C F1-Score | L2A Acurácia | L2A IoU | L2A F1-Score |
|--------------------------------|--------------|---------|--------------|--------------|---------|--------------|
| Todas as bandas - EfficientNet-B1 | **95.28**     | **82.74** | **90.56**     | 95.04       | 81.95   | 90.08        |
| Todas as bandas - MobileNet-V2    | 94.93         | 81.60     | 89.87         | 94.59       | 80.46   | 89.17        |
| RGB + NIR - EfficientNet-B1       | 94.86         | 81.36     | 89.72         | **94.87**   | **81.38** | **89.73**     |
| RGB + NIR - MobileNet-V2          | 94.44         | 79.99     | 88.88         | 94.61       | 80.53   | 89.21        |

### Comparação de Modelos com Classes Semelhantes

| Modelos                          | Acurácia   | IoU      | F1-Score  | Fonte                     |
|----------------------------------|------------|----------|-----------|----------------------------|
| L1C 13 Bandas - EfficientNet-B1 | **95,28%** | **82,75%** | **90,56%** | O Autor                   |
| L2A 4 Bandas - EfficientNet-B1  | 94,87%     | 81,38%   | 89,73%    | O Autor                   |
| UnetMob-V1                      | 94,63%     | 80,62%   | 89,27%    | Aybar (2022)              |
| UnetMob-V2                      | 94,83%     | 81,27%   | 89,67%    | Aybar (2024)              |
| Sensei-V2                       | 94,40%     | 79,84%   | 88,79%    | Francis Sensor (2024)     |

### Classes e Superclasses de pixels para avaliação dos modelos com classes diferentes, segunda a abordagem proposta em Aybar (2022).

| Código | Classe           | Superclasse 1 | Superclasse 2 | Prioridade |
|--------|------------------|----------------|----------------|------------|
| 0      | Claro            | Não-nuvem      | Válido         | 4          |
| 1      | Nuvem Espessa    | Nuvem          | Inválido       | 1          |
| 2      | Nuvem Fina       | Nuvem          | Inválido       | 3          |
| 3      | Sombra de Nuvem  | Não-nuvem      | Inválido       | 2          |


### Resultados da comparação entre modelos para a Superclasse 1, Nuvem e Não Nuvem

| Modelo                        | Acurácia (%) | IoU (%) | F1-Score (%) |
|------------------------------|--------------|---------|---------------|
| L1C 13 Bandas EfficientNet-B1 | **94.17**    | **85.74** | **92.33**     |
| L2A 4 Bandas EfficientNet-B1  | 93.56        | 84.30    | 91.48         |
| Sen2Cor                       | 80.64        | 52.41    | 68.78         |
| S2Cloudless                   | 86.33        | 68.31    | 81.17         |
| Fmask                         | 87.26        | 71.66    | 83.49         |
| KappaMask_L1C                | 85.69        | 69.38    | 81.92         |
| KappaMask_L2A                | 80.00        | 63.08    | 77.36         |
| CD_FCNN_RGBI                 | 80.52        | 51.62    | 68.09         |
| CD_FCNN_RGBISWIR             | 81.48        | 53.27    | 69.52         |


### Resultados da comparação entre modelos para a Superclasse 2, Válido e Inválido

| Modelo                        | Acurácia (%) | IoU (%) | F1 (%)     |
|------------------------------|--------------|---------|------------|
| L1C 13 Bandas EfficientNet-B1 | **93.98**    | **88.03** | **93.64**  |
| L2A 4 Bandas EfficientNet-B1  | 93.32        | 86.75    | 92.90      |
| Sen2Cor                       | 73.36        | 46.85    | 63.81      |
| S2Cloudless                   | 80.26        | 61.28    | 75.99      |
| Fmask                         | 86.21        | 74.66    | 85.49      |
| KappaMask_L1C                | 85.23        | 73.32    | 84.60      |
| KappaMask_L2A                | 80.23        | 67.96    | 80.92      |
| CD_FCNN_RGBI                 | 72.17        | 43.50    | 60.63      |
| CD_FCNN_RGBISWIR             | 72.68        | 44.07    | 61.18      |

## 📝 Referências
Esta Dissertação: _Modelos de Detecção de Nuvens em Imagens Mulespectrais do Sentinel-2_, Alan Breno Soares Corrêa, Programa de Pós Graduação em Engenharia Elétrica (PPGEE), Universidade Federal do Pará (UFPA), 2025.

## 🤝 Agradecimentos
A Deus, por me conceder força, sabedoria e perseverança para seguir adiante, mesmo diante dos desafios. Sem a Sua presença em minha vida, nada disso seria possível.

À minha esposa, Paula Cristina, e ao meu filho, Alan Miguel, que são a razão maior da minha luta e superação. O amor, o apoio e a paciência de vocês foram fundamentais em todos os momentos dessa caminhada.

Aos meus pais, Antonio Carlos e Adilce Meire, por todo amor, apoio e pelos valores que me ensinaram desde a infância. Foram exemplos de dedicação e coragem, e tudo o que conquistei até aqui carrega um pedaço do esforço de vocês.

Aos meus irmãos, Antonio, Alex, Eleticia e Lelma, por serem parte essencial da minha vida, mesmo quando distantes, sei que posso contar com vocês.

À minha tia Ana, que me acolheu no início da minha jornada acadêmica, permitindo que eu morasse com ela e pudesse dar os primeiros passos na universidade. Seu gesto de generosidade e carinho foi crucial para que eu pudesse chegar até aqui.

Aos amigos que fiz na UFPA - Lucian (meu amigo e coorientador de TCC), Mozart, Clara e Caio -, pelos momentos de parceria, aprendizado e amizade. A convivência com vocês tornou esta jornada acadêmica mais leve e enriquecedora.

Ao professor Dr. Fabrício Barros, por me acolher no Laboratório de Computação e Telecomunicações (LCT), me orientar desde o Trabalho de Conclusão de Curso (TCC) e, principalmente, por acreditar no meu potencial, incentivando-me a seguir no caminho da pesquisa e do mestrado. Sua orientação e apoio foram fundamentais para a realização deste trabalho.

Ao professor Dr. Marcos Seruffo, cuja orientação, apoio e incentivo foram essenciais para a continuidade deste sonho. Sua confiança e palavras de encorajamento me deram forças para seguir em frente.

À empresa \textit{Carbonext - Nature and Future}, pela parceria no Programa de Mestrado e Doutorado Acadêmico para Inovação (MAI/DAI), que me possibilitou manter o foco e a dedicação durante o curso. A oportunidade de aliar pesquisa e prática profissional foi essencial para o desenvolvimento deste trabalho.

Aos novos colegas Jean, Leo, Pedro, Williane e Albert, cuja parceria e amizade ao longo dos últimos anos foram extremamente enriquecedoras, tanto para o crescimento pessoal quanto acadêmico.

A todos os meus professores, que, ao longo da minha formação, contribuíram com conhecimento, dedicação e inspiração, deixando marcas importantes na minha trajetória.

Por fim, a todos que, de alguma forma, fizeram parte dessa caminhada, o meu mais sincero e profundo agradecimento.



