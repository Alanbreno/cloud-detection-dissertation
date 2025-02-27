import torch
import pandas as pd
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import CoreDataset

# Pipeline de augmentations com Shift e Rotação
augmentation_pipeline = A.Compose(
    [
        A.HorizontalFlip(p=0.5),  # Flip horizontal com 50% de chance
        A.VerticalFlip(p=0.5),  # Flip vertical com 50% de chance
        A.RandomRotate90(p=0.5),  # Rotação em múltiplos de 90 graus (90, 180, 270)
        ToTensorV2(),  # Converte para tensores PyTorch
    ]
)

class CoreDataModule(pl.LightningDataModule):
    def __init__(self, dataframe: pd.DataFrame, batch_size: int = 4, bandas=[1,2,3,4,5,6,7,8,9,10,11,12,13]):
        super().__init__()

        # Separar o DataFrame em datasets de treino, validação e teste
        self.train_dataset = dataframe[dataframe["tortilla:data_split"] == "train"]
        self.validation_dataset = dataframe[dataframe["tortilla:data_split"] == "validation"]
        self.test_dataset = dataframe[dataframe["tortilla:data_split"] == "test"]

        # Definir o batch_size
        self.batch_size = batch_size
        self.bandas = bandas

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=CoreDataset(
                self.train_dataset,
                augmentations=augmentation_pipeline,
                bandas=self.bandas
            ),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=CoreDataset(
                self.validation_dataset,
                augmentations=None,
                bandas=self.bandas),
            batch_size=self.batch_size,
            num_workers=8,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=CoreDataset(self.test_dataset,
                                augmentations=None,
                                bandas=self.bandas),
            batch_size=self.batch_size,
            num_workers=8,
        )
