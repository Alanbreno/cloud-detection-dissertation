# Encoders disponíveis no Segmentation Models Pytorch
ENCODER_NAME_MOBILENET = "mobilenet_v2"
ENCODER_NAME_RESNET18 = "resnet18"
ENCODER_NAME_EFFICIENTNETB1 = "efficientnet-b1"
ENCODER_NAME_MOBILEONES1 = "mobileone_s1"


# Nomes das pastas dos modelos
NAME_MOBILENET = "Unet_mobilenet_v2"
NAME_RESNET18 = "Unet_resnet18"
NAME_EFFICIENTNETB1 = "Unet_efficientnet-b1"
NAME_MOBILEONES1 = "Unet_mobileone_s1"


# Hiperparâmetros do modelo
LEARNING_RATE = 1e-3
EPOCHS = 100
BATCH_SIZE = 32
CLASSES = 4
IN_CHANNELS = 4 # Blue, Green, Red, NIR
ACCELERATOR = "auto"

# Diretórios
DIR_BASE = "/content/drive/MyDrive/taco_CloudSen12/"
DIR_LOG = "/content/drive/MyDrive/Unet_4_bands_l1c/lightning_logs/"

# Diretórios raiz dos modelos
DIR_ROOT_MOBILENET = "/content/drive/MyDrive/Unet_4_bands_l1c/lightning_logs/Unet_mobilenet_v2"
DIR_ROOT_RESNET18 = "/content/drive/MyDrive/Unet_4_bands_l1c/lightning_logs/Unet_resnet18"
DIR_ROOT_EFFICIENTNETB1 = "/content/drive/MyDrive/Unet_4_bands_l1c/lightning_logs/Unet_efficientnet-b1"
DIR_ROOT_MOBILEONES1 = "/content/drive/MyDrive/Unet_4_bands_l1c/lightning_logs/Unet_mobileone_s1"

