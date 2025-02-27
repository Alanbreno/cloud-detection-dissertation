import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
import rasterio as rio
import tacoreader

class CoreDataset(Dataset):
    def __init__(self, subset, augmentations=None, bandas=[1,2,3,4,5,6,7,8,9,10,11,12,13]):
        self.subset = subset
        self.augmentations = augmentations
        self.cache = {}
        self.bandas = bandas

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
            s2l2a_data: np.ndarray = src.read(self.bandas).astype(np.float32)/10000
            target_data: np.ndarray = dst.read(1).astype(np.int64)
        
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
