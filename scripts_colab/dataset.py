import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import config
import mlstac
import torch

class CoreDataset(Dataset):
    def __init__(self, subset: pd.DataFrame, index_mask, augmentations=None):
        subset.reset_index(drop=True, inplace=True)
        self.subset = subset
        self.index_mask = index_mask
        self.augmentations = augmentations
        estatisticas = np.load(config.DIR_BASE + "CloudSen12+/estatisticas_dataset_512_high_train.npz")
        self.medias = estatisticas["medias"]
        self.desvios_padroes = estatisticas["desvios_padroes"]
        
    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index: int):
        # Obter o caminho do arquivo a partir do DataFrame
        bandas = mlstac.get_data(dataset=self.subset.iloc[index], quiet=True).squeeze()
        
        # Assumindo que as bandas estão nos primeiros canais
        X = bandas[0:13, :, :].astype(np.float32)
        imagem_normalizada = np.zeros_like(X)
        for banda in range(13):
            imagem_normalizada[banda, :, :] = (X[banda, :, :] - self.medias[banda]) / self.desvios_padroes[banda]
            
        X = imagem_normalizada

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