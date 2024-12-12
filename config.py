# Encoders disponíveis no Segmentation Models Pytorch
ENCODER_NAME_EFFICIENTNETB0 = "timm-efficientnet-b0"
ENCODER_NAME_EFFICIENTNETB1 = "timm-efficientnet-b1"
ENCODER_NAME_MOBILENETV2 = "mobilenet_v2"
ENCODER_NAME_MOBILEONE = "mobileone_s0"

# Nomes das pastas dos modelos
NAME_EFFICIENTNETB0 = "Unet_timm-efficientnet-b0"
NAME_EFFICIENTNETB1 = "Unet_timm-efficientnet-b1"
NAME_MOBILENETV2 = "Unet_mobilenet_v2"
NAME_MOBILEONE = "Unet_mobileone_s0"

# Hiperparâmetros do modelo
LEARNING_RATE = 1e-3
EPOCHS = 100
BATCH_SIZE = 32
CLASSES = 4
IN_CHANNELS = 4
ACCELERATOR = "auto"

# Diretórios
DIR_BASE = "/home/mseruffo/"
DIR_LOG = "/home/mseruffo/Unet_4bands/lightning_logs/"

# Diretórios raiz dos modelos
DIR_ROOT_EFFICIENTNETB0 = "/home/mseruffo/Unet_4bands/lightning_logs/Unet_timm-efficientnet-b0"
DIR_ROOT_EFFICIENTNETB1 = "/home/mseruffo/Unet_4bands/lightning_logs/Unet_timm-efficientnet-b1"
DIR_ROOT_MOBILENETV2 = "/home/mseruffo/Unet_4bands/lightning_logs/Unet_mobilenet_v2"
DIR_ROOT_MOBILEONE = "/home/mseruffo/Unet_4bands/lightning_logs/Unet_mobileone_s0"
