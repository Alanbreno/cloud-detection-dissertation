import pandas as pd
import glob


def get_image_paths(diretorio_base):
    # Listar arquivos de treino, validação e teste para as imagens de 512x512
    train_files = glob.glob(
        diretorio_base + "CloudSen12+/train/*/*/*.tif"
    )
    val_files = glob.glob(
        diretorio_base + "CloudSen12+/val/*/*/*.tif"
    )
    test_files = glob.glob(
        diretorio_base + "CloudSen12+/test/*/*/*.tif"
    )
    
    # Criar DataFrames e adicionar a coluna 'set_type'
    train_df = pd.DataFrame(train_files, columns=["file_path"])
    train_df["set_type"] = "train"

    val_df = pd.DataFrame(val_files, columns=["file_path"])
    val_df["set_type"] = "val"

    test_df = pd.DataFrame(test_files, columns=["file_path"])
    test_df["set_type"] = "test"
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    print(f"Total de imagens: {len(df)}")
    return df
