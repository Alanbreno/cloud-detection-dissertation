# Encoders disponíveis no Segmentation Models Pytorch
ENCODER_NAME_EFFICIENTNETB0 = "timm-efficientnet-b0"
ENCODER_NAME_EFFICIENTNETB2 = "timm-efficientnet-b1"
ENCODER_NAME_EFFICIENTNETB3 = "timm-efficientnet-b2"

# Nomes das pastas dos modelos
NAME_EFFICIENTNETB0 = "Unet_timm-efficientnet-b0"
NAME_EFFICIENTNETB2 = "Unet_timm-efficientnet-b2"
NAME_EFFICIENTNETB3 = "Unet_timm-efficientnet-b3"

# Hiperparâmetros do modelo
LEARNING_RATE = 1e-3
EPOCHS = 100
BATCH_SIZE = 16
CLASSES = 4
IN_CHANNELS = 13
ACCELERATOR = "auto"

# Diretórios
DIR_BASE = "/home/mseruffo/"
DIR_LOG = "/home/mseruffo/Unet/lightning_logs/"

# Diretórios raiz dos modelos
DIR_ROOT_EFFICIENTNETB0 = "/home/mseruffo/Unet/lightning_logs/Unet_timm-efficientnet-b0"
DIR_ROOT_EFFICIENTNETB2 = "/home/mseruffo/Unet/lightning_logs/Unet_timm-efficientnet-b2"
DIR_ROOT_EFFICIENTNETB3 = "/home/mseruffo/Unet/lightning_logs/Unet_timm-efficientnet-b3"
