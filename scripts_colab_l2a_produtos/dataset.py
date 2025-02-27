import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import config
import torch
import rasterio as rio
import tacoreader
import warnings



def calculate_ndvi_ndwi_bndvi(nir, red, green, blue):
    """
    Calcula o NDVI e o NDWI corretamente, substituindo valores inválidos por zeros
    e suprimindo warnings durante os cálculos.

    Parâmetros:
    - nir: array ou matriz do infravermelho próximo (Near Infrared).
    - red: array ou matriz da banda vermelha (Red).
    - green: array ou matriz da banda verde (Green).

    Retorna:
    - ndvi: Índice de Vegetação por Diferença Normalizada.
    - ndwi: Índice de Água por Diferença Normalizada.
    """
    # Suprimir warnings temporariamente
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # Calcular NDVI: (NIR - Red) / (NIR + Red)
        ndvi = (nir - red) / (nir + red)
        # Calcular NDWI: (Green - NIR) / (Green + NIR)
        ndwi = (green - nir) / (green + nir)
        # Calcular BNDVI (NIR - Blue) / (NIR + Blue)
        bndvi = (nir - blue) / (nir + blue)

    # Substituir divisões por zero ou valores inválidos por 0
    ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=0.0, neginf=0.0)
    ndwi = np.nan_to_num(ndwi, nan=0.0, posinf=0.0, neginf=0.0)
    bndvi = np.nan_to_num(bndvi, nan=0.0, posinf=0.0, neginf=0.0)

    return ndvi, ndwi, bndvi

class CoreDataset(Dataset):
    def __init__(self, subset, augmentations=None):
        self.subset = subset
        self.augmentations = augmentations
        self.cache = {}

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx: int):
        if idx not in self.cache:
            sample = self.subset.read(idx)
            s2l2a: str = sample.read(0)
            target: str = sample.read(1)
            self.cache[idx] = (s2l2a, target)
        else:
            s2l2a, target = self.cache[idx]

        # Open the files and load data
        with rio.open(s2l2a) as src, rio.open(target) as dst:
            s2l2a_data: np.ndarray = src.read([2,3,4,8]).astype(np.float32)
            target_data: np.ndarray = dst.read().astype(np.int64)
            
        target_data = target_data.squeeze()  # Removendo a dimensão extra
        
        NDVI, NDWI, BNDVI = calculate_ndvi_ndwi_bndvi(s2l2a_data[3], s2l2a_data[2], s2l2a_data[1], s2l2a_data[0])
        # Concatenar os índices de vegetação em uma nova imagem
        s2l2a_data_prod = np.concatenate([NDVI[np.newaxis], NDWI[np.newaxis], BNDVI[np.newaxis]], axis=0)
        
        
        if self.augmentations:
            # Transpor a imagem de (bands, height, width) para (height, width, bands) para trabalhar com Albumentations
            augmented = self.augmentations(image=s2l2a_data_prod.transpose(1, 2, 0), mask=target_data)
            s2l2a_data_prod = augmented["image"].float()  # Convertendo para tensor e float32
            target_data = augmented["mask"].long()  # Convertendo a máscara para tensor long (para classificação)
        else:
            s2l2a_data_prod = torch.from_numpy(s2l2a_data_prod).float()
            target_data = torch.from_numpy(target_data).long()
            
        
        assert np.isfinite(s2l2a_data_prod).all(), f"Entrada contém valores não finitos: {s2l2a_data_prod}"
        assert np.isfinite(target_data).all(), f"Máscara contém valores não finitos: {target_data}"
        assert target_data.min() >= 0 and target_data.max() < 4, f"Máscara fora do intervalo esperado: min={target_data.min()}, max={target_data.max()}"
        assert target_data.dtype == torch.long, f"Tipo de dado incorreto em target_data: {target_data.dtype}"

        return s2l2a_data_prod, target_data
