import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import config
import torch
import rasterio as rio
import tacoreader

class CoreDataset(Dataset):
    def __init__(self, subset, subset_extra, augmentations=None):
        self.subset = subset
        self.subset_extra = subset_extra[subset_extra["tortilla:id"].isin(subset["tortilla:id"])]
        self.augmentations = augmentations
        self.cache = {}

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx: int):
        if idx not in self.cache:
            sample = self.subset.read(idx)
            extra_sample = self.subset_extra.read(idx)
            s2l2a: str = sample.read(0)
            target: str = sample.read(1)
            scl: str = extra_sample.read(5)
            self.cache[idx] = (s2l2a, target, scl)
        else:
            s2l2a, target, scl = self.cache[idx]

        # Open the files and load data
        with rio.open(s2l2a) as src, rio.open(target) as dst, rio.open(scl) as scl:
            s2l2a_data: np.ndarray = src.read([2,3,4,8]).astype(np.float32)/10000
            target_data: np.ndarray = dst.read(1).astype(np.int64)
            scl_data: np.ndarray = scl.read(1).astype(np.float32)
            
        scl_data[(scl_data != 3) & (scl_data != 9) & (scl_data != 8)] = 0 # Definindo como 0 as classes que não são de interesse
        scl_data[scl_data == 3] = 0.33333 # Definindo como 0.33 a classe de sombra de nuvens
        scl_data[scl_data == 9] = 1       # Definindo como 1 a classe alta probabilidade de nuvens
        scl_data[scl_data == 8] = 0.66666 # Definindo como 0.66 a classe baixa probabilidade de nuvens
        
        # Concatenares as bandas de SCL com as bandas de S2L2A
        s2l2a_data = np.concatenate([s2l2a_data, scl_data[None, ...]], axis=0)
        
        
        if self.augmentations:
            # Transpor a imagem de (bands, height, width) para (height, width, bands) para trabalhar com Albumentations
            augmented = self.augmentations(image=s2l2a_data.transpose(1, 2, 0), mask=target_data)
            s2l2a_data = augmented["image"].float()  # Convertendo para tensor e float32
            target_data = augmented["mask"].long()  # Convertendo a máscara para tensor long (para classificação)
        else:
            s2l2a_data = torch.from_numpy(s2l2a_data).float()
            target_data = torch.from_numpy(target_data).long()
            
        
        assert np.isfinite(s2l2a_data).all(), f"Entrada contém valores não finitos: {s2l2a_data}"
        assert np.isfinite(target_data).all(), f"Máscara contém valores não finitos: {target_data}"
        assert target_data.min() >= 0 and target_data.max() < 4, f"Máscara fora do intervalo esperado: min={target_data.min()}, max={target_data.max()}"
        assert target_data.dtype == torch.long, f"Tipo de dado incorreto em target_data: {target_data.dtype}"

        return s2l2a_data, target_data
