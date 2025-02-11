import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import config
import torch
import rasterio as rio
import tacoreader

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
            s2l1c: str = sample.read(0)
            target: str = sample.read(1)
            self.cache[idx] = (s2l1c, target)
        else:
            s2l1c, target = self.cache[idx]

        # Open the files and load data
        with rio.open(s2l1c) as src, rio.open(target) as dst:
            s2l1c_data: np.ndarray = src.read().astype(np.float32)/10000
            target_data: np.ndarray = dst.read().astype(np.int64)
            
        target_data = target_data.squeeze()  # Removendo a dimensão extra
        
        if self.augmentations:
            # Transpor a imagem de (bands, height, width) para (height, width, bands) para trabalhar com Albumentations
            augmented = self.augmentations(image=s2l1c_data.transpose(1, 2, 0), mask=target_data)
            s2l1c_data = augmented["image"].float()  # Convertendo para tensor e float32
            target_data = augmented["mask"].long()  # Convertendo a máscara para tensor long (para classificação)
        else:
            s2l1c_data = torch.from_numpy(s2l1c_data).float()
            target_data = torch.from_numpy(target_data).long()
            
        
        assert np.isfinite(s2l1c_data).all(), f"Entrada contém valores não finitos: {s2l1c_data}"
        assert np.isfinite(target_data).all(), f"Máscara contém valores não finitos: {target_data}"
        assert target_data.min() >= 0 and target_data.max() < 4, f"Máscara fora do intervalo esperado: min={target_data.min()}, max={target_data.max()}"
        assert target_data.dtype == torch.long, f"Tipo de dado incorreto em target_data: {target_data.dtype}"

        return s2l1c_data, target_data
