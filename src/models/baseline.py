# imported libraries
import fasttext
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.model_selection import train_test_split
import sklearn.metrics as m

# Skipgram model, finetuned:
model = fasttext.train_supervised(input=f"{data}/train_fasttext.txt", autotuneValidationFile=f"{data}/val_fasttext.txt") # Hyperparameter tuning by using "autotuneValidationFile" parameter

#Saving the model
model.save_model(f"{data}/model_nace.bin")

# using saved model
model = fasttext.load_model(f"{data}/model_nace.bin")


# hierarchical model

