# imported libraries
import os
import numpy as np
import pandas as pd
import fasttext

from sklearn.model_selection import train_test_split
from src.config import MODELS_FASTXT, DATA_FX_TR_VAL_TE, DATA_FX_TR_TE


def fasttext_dataprep(df: pd.DataFrame, columns:list[str], df_file:str)-> pd.DataFrame:
    df = df.copy()
    df[columns] = df[columns].astype(str).replace(r'\b(nan|none|na)\b', '', regex=True)
    # Create fasttext format
    df["fasttext_format"] = "__label__" + df[columns].agg(' '.join, axis=1)
    # Remove extra spaces
    df["fasttext_format"] = df["fasttext_format"].str.replace(r'\s+', ' ', regex=True).str.strip()

    df["fasttext_format"].to_csv(f"{df_file}.txt", index=False, header=False)
    #pd.read_csv(f"{DATA_FX_TR_VAL_TE}test.csv")
    
    return df 


def pred_prep(df:pd.DataFrame, input_cols:list[str], output_cols:str)->list[str]:
    labels = df[output_cols].astype(str).to_numpy()#[:, 0]
    if len(input_cols) >1:
        input_txt = df[input_cols].astype(str).agg(' '.join, axis=1).tolist()
        #df = df.copy()
        #df["tekst og navn"] = input_txt
    else: 
        if isinstance(input_cols, list):
            input_cols=input_cols[0]
        input_txt = df[input_cols].astype(str).tolist()
    return input_txt, labels


    

### Fasttext result preparation
def output_prep(labels:list[str], pred_probs=None):
    labels = np.char.replace(np.ravel(np.array(labels)), "__label__", "")

    #labels = [l[0].replace('__label__', '') for l in labels]
    #labels = np.array(labels)
    if pred_probs is not None:
        pred_probs=np.ravel(np.array(pred_probs))
    return labels

def hyper_params(model_file):

    model = fasttext.load_model(f"{MODELS_FASTXT}{model_file}.bin")
    args = model.f.getArgs()

    print("lr:", args.lr)
    print("dim:", args.dim)
    print("epoch:", args.epoch)
    print("wordNgrams:", args.wordNgrams)
    print("loss:", args.loss)


