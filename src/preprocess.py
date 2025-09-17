# imported libraries
import fasttext
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.model_selection import train_test_split
import sklearn.metrics as m
import json
import os


def column_subset(df):
    """Choosing a subset of columns """
    return df[["orgnr", "tekst", "navn", "SN2007"]]


def cleaning_df(df:pd.DataFrame) -> pd.DataFrame:
    """Cleaning dataframe by handling missing values """
    df = df.replace(
        to_replace="* Har ingen korrespondanse i SN2007", 
        value=np.nan)
    return df


def mapping(df, key_col, value_col, filename):
    return dict(zip(df[key_col], df[value_col]))
    

def save_mapping(mapping_dict, filename, folder="mappings"):
    os.makedirs(folder, exist_ok=True)  # creating folder if it doesn't exist
    filepath = os.path.join(folder, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(mapping_dict, f, ensure_ascii=False, indent=4)

    print(f"Mapping saved to {filepath}")


def load_mapping(filename, folder="mappings"):
    filepath = os.path.join(folder, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
