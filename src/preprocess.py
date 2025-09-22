# imported libraries
import numpy as np
import pandas as pd
import json
import os


def column_subset(df):
    """Choosing a subset of columns """
    #return df[["orgnr", "tekst", "navn", "SN2007"]]
    return df[["tekst", "navn", "sn2025_1"]]

def cleaning_df(df:pd.DataFrame) -> pd.DataFrame:
    """Cleaning dataframe by handling missing values """
    if (df == "* Har ingen korrespondanse i SN2007").any().any():
        df = df.replace(
            to_replace="* Har ingen korrespondanse i SN2007", 
            value=np.nan)
    if set(["tekst", "navn", "sn2025_1"]).issubset(set(df.columns)):
        df = df[df.groupby("sn2025_1")["navn"].transform("count") >10]
        df = df[df['sn2025_1']!='00.000']
    return df


def df_hier_levels(df: pd.DataFrame, column:str)-> dict[str, pd.DataFrame]:
    """Split a DataFrame into multiple DataFrames based on values in a column.
    Returns them as a dictionary."""
    
    new_dfs = {}
    levels = df[column].unique()
    for i in levels:
        new_dfs[i] = df[df[column] == i]
    return new_dfs

def derive_hier(df: pd.DataFrame, subclass_col:str, section_map):
    """  ["section", "division", "group", "class"] """
    df = df.copy()
    codes = df[subclass_col].astype(str).values
    df["division"] = [c[:2] for c in codes]
    df["group"]    = [c[:4] for c in codes]
    df["class"]    = [c[:5] for c in codes]
    df["section"] = df["division"].map(section_map)
    return df



"""
def load_mapping(filename, folder="mappings"):
    filepath = os.path.join(folder, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
    print("Mapping loaded")


def save_mapping(mapping_dict, filename, folder="mappings"):
    os.makedirs(folder, exist_ok=True)  # creating folder if it doesn't exist
    filepath = os.path.join(folder, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(mapping_dict, f, ensure_ascii=False, indent=4)

    print(f"Mapping saved to {filepath}")
    
def mapping(df, key_col, value_col, filename, folder="mappings"):
    filepath = os.path.join(folder, filename)
    if os.path.exists(filepath):
        mp = load_mapping(filename)
    else:
        save_mapping(dict(zip(df[key_col], df[value_col])), filename)
        mp = load_mapping(filename)
    return mp
"""



