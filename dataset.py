import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import config
import torch
import rasterio

class CoreDataset(Dataset):
    def __init__(self, subset: pd.DataFrame, index_mask, augmentations=None):
        subset.reset_index(drop=True, inplace=True)
        self.subset = subset
        self.index_mask = index_mask
        self.augmentations = augmentations
        estatisticas = np.load(config.DIR_BASE + "CloudSen12+/max_min_dataset_512_high_train.npz")
        self.max = estatisticas["maximos"]
        self.min = estatisticas["minimos"]
        
    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index: int):
        img_path = self.subset.iloc[index]["file_path"]

        # Lê todas as bandas da imagem
        bandas = rasterio.open(img_path).read()

        # Transforma em array numpy
        bandas = np.array(bandas)
        
        # Assumindo que as bandas estão nos primeiros canais
        X = bandas[0:13, :, :].astype(np.float32)
        
        # Normalizando as bandas
        for i in range(13):
            X[i] = (X[i] - self.min[i]) / (self.max[i] - self.min[i])
            
        # Assumindo que o alvo está no canal 14 (index 13)
        y = bandas[self.index_mask, :, :].astype(np.int64)
        
        if self.augmentations:
            # Transpor a imagem de (bands, height, width) para (height, width, bands) para trabalhar com Albumentations
            augmented = self.augmentations(image=X.transpose(1, 2, 0), mask=y)
            X = augmented["image"].float()  # Convertendo para tensor e float32
            y = augmented[
                "mask"
            ].long()  # Convertendo a máscara para tensor long (para classificação)
        else:
            X = torch.from_numpy(X).float()
            y = torch.from_numpy(y).long()
            
        
        assert np.isfinite(X).all(), f"Entrada contém valores não finitos: {X}"
        assert np.isfinite(y).all(), f"Máscara contém valores não finitos: {y}"
        assert y.min() >= 0 and y.max() < 4, f"Máscara fora do intervalo esperado: min={y.min()}, max={y.max()}"
        assert y.dtype == torch.long, f"Tipo de dado incorreto em y: {y.dtype}"

        return X, y
