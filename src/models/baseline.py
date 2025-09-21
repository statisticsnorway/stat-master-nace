# imported libraries
import fasttext
from src.config import SAVE_PATH
import os


def run_fasttext_model():
    if os.path.exists(f"{SAVE_PATH}/model_nace.bin"):
        model = fasttext.load_model(f"{SAVE_PATH}/model_nace.bin")
        
    else:
        # Skipgram model, finetuned:
        model = fasttext.train_supervised(input=f"{SAVE_PATH}/train_fasttext.txt", autotuneValidationFile=f"{SAVE_PATH}/val_fasttext.txt") # Hyperparameter tuning by using "autotuneValidationFile" parameter
        #Saving the model
        model.save_model(f"{SAVE_PATH}/model_nace.bin")
    return model


# hierarchical model

