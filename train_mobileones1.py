import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from datamodule import CoreDataModule
from model import Unet_CD_Sentinel2
import config
import read_paths_image as rpi
import metrics

# Para usar modelos diferentes, basta adicionar as variáveis correspondentes no arquivo config.py

# Primeiro Treinamento com as imagens 512x512
tb_logger = TensorBoardLogger(config.DIR_LOG, name=config.NAME_MOBILEONE)

# Gera o dataframe com as imagens 512x512
df_512 = rpi.get_image_paths(config.DIR_BASE)

# Define the datamodule
datamodule = CoreDataModule(
    dataframe=df_512,
    batch_size=config.BATCH_SIZE
)

# Define the model
model = Unet_CD_Sentinel2(
    encoder_name=config.ENCODER_NAME_MOBILEONE,
    classes=config.CLASSES,
    in_channels=config.IN_CHANNELS,
    learning_rate=config.LEARNING_RATE,
)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=config.DIR_ROOT_MOBILEONE,
    filename="{epoch}-{train_loss:.2f}-{val_loss:.2f}-trainHigh512",
    monitor="val_loss",
    mode="min",
    save_top_k=1,
)

earlystopping_callback = pl.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, mode="min"
)

callbacks = [checkpoint_callback, earlystopping_callback]

# Define the trainer
trainer = pl.Trainer(
    max_epochs=config.EPOCHS,
    log_every_n_steps=1,
    callbacks=callbacks,
    accelerator=config.ACCELERATOR,
    precision="16-mixed",
    logger=tb_logger,
    default_root_dir=config.DIR_ROOT_MOBILEONE,
)

# Start the training
trainer.fit(model=model, datamodule=datamodule)
# Carregar o melhor modelo diretamente
model = Unet_CD_Sentinel2.load_from_checkpoint(
    checkpoint_callback.best_model_path,
    encoder_name=config.ENCODER_NAME_MOBILEONE,
    classes=config.CLASSES,
    in_channels=config.IN_CHANNELS,
    learning_rate=config.LEARNING_RATE,
)

# run val dataset
val_metrics = trainer.validate(model, datamodule=datamodule, verbose=True)
print(val_metrics)

# run test dataset
test_metrics = trainer.test(model, datamodule=datamodule, verbose=True)
print(test_metrics)

acuracia, acuracia_balanceada, iou, f1_score, f2_score, recall = metrics.calculate_metrics(datamodule.test_dataloader(), model.model)

# Salva o modelo treinado
smp_model = model.model
smp_model.save_pretrained(config.DIR_ROOT_MOBILEONE + "/" + config.NAME_MOBILEONE)
