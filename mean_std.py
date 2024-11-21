import read_paths_image as rpi
import config
import tqdm
import numpy as np
import rasterio

data_high_509_train = rpi.get_image_paths(config.DIR_BASE)
data_high_509_train = data_high_509_train[data_high_509_train["split"] == "train"]


# Inicializar variáveis para somas e contagens para cada banda
num_bandas = 13  # Sentinel-2 tem 13 bandas
somas = np.zeros(num_bandas)
somas_quadrados = np.zeros(num_bandas)
contagem_total = np.zeros(num_bandas)

# Primeira passada: calcular as médias para cada banda
print("Calculando as médias...")
for i in tqdm(range(len(data_high_509_train)), desc="Passo 1/2"):
    img_path = data_high_509_train.iloc[i]["file_path"]

    # Lê todas as bandas da imagem
    imagem = rasterio.open(img_path).read()

    # Transforma em array numpy
    imagem = np.array(imagem).astype(np.float32)

    for banda in range(num_bandas):
        banda_atual = imagem[banda, :, :]  # Selecionar a banda específica
        validos = banda_atual[banda_atual > 0]  # Ignorar valores nulos ou inválidos (e.g., 0)
        somas[banda] += np.sum(validos)
        contagem_total[banda] += len(validos)

# Calcular a média para cada banda
medias = somas / contagem_total
print(f"Soma total de cada banda: {somas}")
print(f"Contagem total de cada banda: {contagem_total}")

# Segunda passada: calcular o desvio padrão para cada banda
print("Calculando os desvios padrão...")
for i in tqdm(range(len(data_high_509_train)), desc="Passo 2/2"):
    img_path = data_high_509_train.iloc[i]["file_path"]

    # Lê todas as bandas da imagem
    imagem = rasterio.open(img_path).read()

    # Transforma em array numpy
    imagem = np.array(imagem).astype(np.float32)


    for banda in range(num_bandas):
        banda_atual = imagem[banda, :, :]
        validos = banda_atual[banda_atual > 0]
        somas_quadrados[banda] += np.sum((validos - medias[banda]) ** 2)

# Calcular o desvio padrão para cada banda
desvios_padroes = np.sqrt(somas_quadrados / contagem_total)

# Salvar estatísticas para uso posterior
np.savez("/home/mseruffo/CloudSen12+/estatisticas_dataset_512_high_train.npz", medias=medias, desvios_padroes=desvios_padroes)

print("Processamento concluído!")
print("Médias por banda:", medias)
print("Desvios padrão por banda:", desvios_padroes)
