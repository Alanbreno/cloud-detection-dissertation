import read_paths_image as rpi
import config
import tqdm
import numpy as np
import rasterio

# Carregar os caminhos das imagens
data_high_509_train = rpi.get_image_paths(config.DIR_BASE)
data_high_509_train = data_high_509_train[data_high_509_train["split"] == "train"]

# Número de bandas do Sentinel-2
num_bandas = 13  

# Inicializar arrays para máximos e mínimos de cada banda
maximos = np.full(num_bandas, -np.inf)  # Valores iniciais muito baixos
minimos = np.full(num_bandas, np.inf)   # Valores iniciais muito altos

# Calcular máximos e mínimos em uma única passada
print("Calculando máximos e mínimos...")
for i in tqdm(range(len(data_high_509_train)), desc="Processando imagens"):
    img_path = data_high_509_train.iloc[i]["file_path"]

    # Ler todas as bandas da imagem
    imagem = rasterio.open(img_path).read()

    # Transforma em array numpy
    imagem = np.array(imagem).astype(np.float32)

    for banda in range(num_bandas):
        banda_atual = imagem[banda, :, :]
        validos = banda_atual#[banda_atual > 0]  # Ignorar valores nulos ou inválidos (e.g., 0)
        
        # Atualizar máximos e mínimos
        maximos[banda] = max(maximos[banda], np.max(validos))
        minimos[banda] = min(minimos[banda], np.min(validos))

# Salvar máximos e mínimos para uso posterior
np.savez("/home/mseruffo/CloudSen12+/max_min_dataset_512_high_train.npz", maximos=maximos, minimos=minimos)

print("Processamento concluído!")
print("Máximos por banda:", maximos)
print("Mínimos por banda:", minimos)
