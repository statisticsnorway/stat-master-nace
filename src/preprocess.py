# imported libraries
import fasttext
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.model_selection import train_test_split
import sklearn.metrics as m



def column_subset(df):
    """Choosing a subset of columns """
    return df[["orgnr", "tekst", "navn", "SN2007"]]


def cleaning_df(df:pd.DataFrame) -> pd.DataFrame:
    """Cleaning dataframe by handling missing values """
    df = df.replace(
        to_replace="* Har ingen korrespondanse i SN2007", 
        value=np.nan)
    return df

