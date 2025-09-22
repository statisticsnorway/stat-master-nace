# imported libraries
import fasttext
from src.config import SAVE_PATH
import os


def run_fasttext_model(model_file='model_nace', train_file='train_fasttext', val_file='val_fasttext'):
    if os.path.exists(f"{SAVE_PATH}/{model_file}.bin"):
        model = fasttext.load_model(f"{SAVE_PATH}/model_nace.bin")
        
    else:
        # Skipgram model, finetuned:
        model = fasttext.train_supervised(input=f"{SAVE_PATH}/{train_file}.txt", autotuneValidationFile=f"{SAVE_PATH}/{val_file}.txt") # Hyperparameter tuning by using "autotuneValidationFile" parameter
        #Saving the model
        model.save_model(f"{SAVE_PATH}/{model_file}.bin")
        model = fasttext.load_model(f"{SAVE_PATH}/{model_file}.bin")
    return model


# hierarchical model

def hier_fasttext():
    model_div = run_fasttext_model(train_div, val_div)
    
    
def fasttext_each_hier():
    ...
