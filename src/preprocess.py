# imported libraries
import numpy as np
import pandas as pd
import json
import os
import string
import re
from bs4 import BeautifulSoup
from io import StringIO
import requests

from config import DATA_BR_TEST, DATA_BR_TRAIN, TRANSITION_DATA_PATH, HIERARCHY_DATA, HIERARCHY_DATA_PRUNED, DATA



first_names_menn_df = pd.read_excel(f"{DATA}manneNavn.xlsx")
first_names_kvinner_df = pd.read_excel(f"{DATA}kvinneNavn.xlsx")
first_names_df = pd.concat([first_names_menn_df, first_names_kvinner_df], ignore_index=True)

last_names_df = pd.read_excel(f"{DATA}etternavn.xlsx")

# Converting to sets
first_names = set(first_names_df.iloc[:, 0].dropna().str.strip())
last_names = set(last_names_df.iloc[:, 0].dropna().str.strip())

# Combining into one set
all_names = first_names.union(last_names)

# Compiling regex 
pattern = r"\b(" + "|".join(map(re.escape, all_names)) + r")s?\b"
name_regex = re.compile(pattern, flags=re.IGNORECASE)

all_names_lower = {n.lower() for n in all_names}


def general_preprocess(series):
    # Convert to string
    s = series.astype(str)
    
    # Removing HTML tags efficiently
    s = s.apply(lambda x: BeautifulSoup(x, "html.parser").get_text())
    
    # Lowercase
    s = s.str.lower()
    
    # Removing numbers
    s = s.str.replace(r"\d+", "", regex=True)
    
    # Removing special characters
    s = s.str.replace(r"[^\w\sæøåÆØÅ]", " ", regex=True)
    
    # Remove names from all_names_lower
    s = s.apply(lambda x: " ".join([w for w in x.split() if w not in all_names_lower]))
    
    # Removing any form of 'nan'
    s.str.replace(r'\b[Nn][Aa][Nn]?\b', '', regex=True)
    
    # Removing extra spaces
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()

    # Removing punctuation
    s = s.apply(lambda x: x.translate(str.maketrans(string.punctuation, " " * len(string.punctuation))))
    
    return s

def column_subset(df):
    """Choosing a subset of columns """
    return df[["company_activity", "company_name", "nace_21_code", "nace_21_description_nb"]]

def cleaning_df(df:pd.DataFrame, fasttext_data) -> pd.DataFrame:
    df.copy()
    
    if (df == "* Har ingen korrespondanse i SN2007").any().any():
        df = df.replace(
            to_replace="* Har ingen korrespondanse i SN2007", 
            value=np.nan)
           
    if set(["company_activity", "company_name", "nace_21_code"]).issubset(set(df.columns)):
        df = df[df.groupby("nace_21_code")["company_name"].transform("count") >20]
        df = df[df['nace_21_code']!='00.000']
    
    # general preprocess
    if fasttext_data==True:
        df['company_activity'] = general_preprocess(df['company_activity'])
        df['company_name'] = general_preprocess(df['company_name'])
    return df

"""
def df_hier_levels(df: pd.DataFrame, column:str)-> dict[str, pd.DataFrame]:
    "Split a DataFrame into multiple DataFrames based on values in a column.
    Returns them as a dictionary."
    
    new_dfs = {}
    levels = df[column].unique()
    for i in levels:
        new_dfs[i] = df[df[column] == i]
    return new_dfs
"""
def derive_hier(df: pd.DataFrame, subclass_col:str, section_map):
    """  ["section", "division", "group", "class"] """
    df = df.copy()
    df["division"] = df[subclass_col].str[:2]
    df["group"]    = df[subclass_col].str[:4]
    df["class"]    = df[subclass_col].str[:5]
    df["section"] = df["division"].map(section_map)
    return df


def prune_tree(df, cols=["section", "division", "group", "class", "nace_21_code"]):
    df = df.copy()

    if cols==None:
        # pruning for level 5
        mask_lvl5 = (df["level"] == 5)
        mask_remove_lvl5 = df.loc[mask_lvl5, "code"].str.contains(r"\.\d*0+$", na=False)
        df.loc[mask_lvl5 & mask_remove_lvl5, "code"] = None

        # pruning for deeper levels
        levels = sorted(df["level"].unique(), reverse=True)[:-1]
        for l in levels[1:]:
            mask_next = (df["level"] == l + 1) & (df["code"].isna())
            parent_codes = df.loc[mask_next, "parentCode"].unique()
            if len(parent_codes) > 0:
                pattern = "|".join([re.escape(p) for p in parent_codes])
                mask_to_remove = (
                    (df["level"] == l)
                    & df["code"].notna()
                    & df["code"].str.contains(pattern)
                    & df["code"].str.contains(r"\.\d*0+$", na=False)
                )
                df.loc[mask_to_remove, "code"] = None                

    else:
        for i, col in enumerate(cols[:-1]):
            next_col = cols[i + 1]

            mask_ends_with_zeros = df[col].astype(str).str.contains(r"\.\d*0+$", na=False)

            # Getting all codes that have at least one child
            valid_children_prefixes = (
                df[next_col]
                .dropna()
                .astype(str)
                .apply(lambda x: x.split(".")[0])  # generalize prefix
            )

            # Keeping rows that have no child with that prefix
            has_child = df[col].astype(str).isin(valid_children_prefixes)

            # Removing only if ends with zeros and has no child
            mask_to_remove = mask_ends_with_zeros & ~has_child
            df.loc[mask_to_remove, col] = None

        # last level
        last_col = cols[-1]
        df[last_col] = df[last_col].mask(df[last_col].astype(str).str.contains(r"\.\d*0+$", na=False))

    return df

    
def run_preprocess(save_file, df, fasttext_data=True):
    df = df.copy()
    # NACE 2007 Hierarchi
    df_hier = pd.read_csv(StringIO(requests.get(HIERARCHY_DATA).text), delimiter=',')
    print('reading df')
    mask = df['company_name'].str.contains('helse', case=False, na=False) & \
        df['company_name'].str.contains('nongkhai', case=False, na=False)

    print(df.loc[mask, 'company_name'])


    #df = df.fillna('')
    
    # Getting data for sn-codes, org-nr and text and filtering to only include groups with 10 > datapoints
    df = cleaning_df(df=df, fasttext_data=fasttext_data)

    df_sn25 = column_subset(df)

    # turning datasett into hierarkies
    map_sec = dict(zip(df_hier[df_hier["level"]==2]["code"], df_hier[df_hier["level"]==2]["parentCode"]))
    df_sn25_hier = derive_hier(df=df_sn25, subclass_col='nace_21_code', section_map=map_sec)
    df = df.replace(['None'], '', regex=False)

    df_sn25_hier.to_csv(f"{save_file}.csv", index=False)
    
    df_hier = prune_tree(df=df_hier, cols=None)
    df_hier.to_csv(HIERARCHY_DATA_PRUNED, index=False)
    return df_sn25_hier


if __name__=='__main__':
    #80 percent dataset
    train = pd.read_parquet(DATA_BR_TRAIN)
    #80 percent dataset
    test = pd.read_parquet(DATA_BR_TEST)

    df = pd.concat([train, test], ignore_index=True)

    df_fx = run_preprocess(save_file=f"{DATA}data_preprocessed", df=df)

    df_lm = run_preprocess(save_file=f"{DATA}data_prep_lm_new", df=df, fasttext_data=False)

    df_lm.head()


