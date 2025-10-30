# imported libraries
import numpy as np
import pandas as pd
import json
import os
import string
import re
from bs4 import BeautifulSoup

from src.config import DATA_PATH, OLD_DATA, TRANSITION_DATA_PATH, HIERARCHY_DATA, RANDOM_STATE, SAVE_PATH, HIERARCHY_DATA_PRUNED


first_names_menn_df = pd.read_excel("/home/stud-msh/stat-master-nace/data/manneNavn.xlsx")
first_names_kvinner_df = pd.read_excel("/home/stud-msh/stat-master-nace/data/kvinneNavn.xlsx")
first_names_df = pd.concat([first_names_menn_df, first_names_kvinner_df], ignore_index=True)

last_names_df = pd.read_excel("/home/stud-msh/stat-master-nace/data/etternavn.xlsx")

# Converting to sets
first_names = set(first_names_df.iloc[:, 0].dropna().str.strip())
last_names = set(last_names_df.iloc[:, 0].dropna().str.strip())

# Combining into one set
all_names = first_names.union(last_names)

# Compiling regex 
pattern = r"\b(" + "|".join(map(re.escape, all_names)) + r")s?\b"
name_regex = re.compile(pattern, flags=re.IGNORECASE)

all_names_lower = {n.lower() for n in all_names}

def remove_names(text):
    # Remove punctuation (replace with spaces)
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    # Remove names
    words = [w for w in text.split() if w.lower() not in all_names_lower]
    return " ".join(words)

def general_preprocess(text):
    #Further preprocessing
    text = text.lower()  # Lowercase
    text = re.sub(r'\d+', '', text)  # Removing numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Removing punctuation
    text = re.sub(r"[^a-zA-Z0-9æøåÆØÅ]", " ", text)  # Removing special characters
    text = BeautifulSoup(text, "html.parser").get_text()  # Removing HTML tags
    return text

def column_subset(df):
    """Choosing a subset of columns """
    #return df[["orgnr", "tekst", "navn", "SN2007"]]
    return df[["tekst", "navn", "sn2025_1"]]

def cleaning_df(df:pd.DataFrame) -> pd.DataFrame:
    df.copy()
    
    if (df == "* Har ingen korrespondanse i SN2007").any().any():
        df = df.replace(
            to_replace="* Har ingen korrespondanse i SN2007", 
            value=np.nan)
           
    if set(["tekst", "navn", "sn2025_1"]).issubset(set(df.columns)):
        df = df[df.groupby("sn2025_1")["navn"].transform("count") >20]
        df = df[df['sn2025_1']!='00.000']
    
    # general preprocess
    df['tekst'] = df['tekst'].apply(general_preprocess)
    df['navn'] = df['navn'].apply(general_preprocess)
    
    # Filtering out names
    df['navn'] = df['navn'].apply(remove_names)
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
    df["division"] = df[subclass_col].str[:2]
    df["group"]    = df[subclass_col].str[:4]
    df["class"]    = df[subclass_col].str[:5]
    df["section"] = df["division"].map(section_map)
    return df


def prune_tree(df, cols=["section", "division", "group", "class", "sn2025_1"]):
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

    
def run_preprocess():
    # NACE 2007 Hierarchi
    df_hier = pd.read_csv(HIERARCHY_DATA,sep=";",encoding="latin-1")
    # Treningsdata
    df = pd.read_parquet(DATA_PATH)

    # Getting data for sn-codes, org-nr and text and filtering to only include groups with 10 > datapoints
    df = cleaning_df(df)
    df_sn25 = column_subset(df)

    # splitting the df_hier into multiple DataFrames based on level
    df_hiers = df_hier_levels(df=df_hier, column='level')
    df_hier_div = df_hiers[2]

    # turning datasett into hierarkies
    map_sec = dict(zip(df_hier["code"], df_hier["parentCode"]))
    df_sn25_hier = derive_hier(df=df_sn25, subclass_col='sn2025_1', section_map=map_sec)
    df_sn25_hier.to_csv(f"{SAVE_PATH}/data_fasttext/data_preprocessed.csv", index=False)
    print(df_sn25_hier)
    
    df_hier = prune_tree(df_hier, cols=None)
    df_hier.to_csv(HIERARCHY_DATA_PRUNED, index=False)
    print(df_hier)



if __name__=='__main__':
    run_preprocess()


