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
from difflib import SequenceMatcher
import spacy
from nltk.stem.snowball import SnowballStemmer

from config import DATA_BR_TEST, DATA_BR_TRAIN, HIERARCHY_DATA, HIERARCHY_DATA_PRUNED, DATA, DATA_LM_TR_VAL_TE,DATA_FX_TR_VAL_TE

nlp = spacy.load("nb_core_news_sm")  # Norwegian Bokmål
stemmer = SnowballStemmer("norwegian")

import os
import pandas as pd

first_names_menn_df = pd.read_excel(os.path.join(DATA, "manneNavn.xlsx"))
first_names_kvinner_df = pd.read_excel(os.path.join(DATA, "kvinneNavn.xlsx"))
first_names_df = pd.concat([first_names_menn_df, first_names_kvinner_df],ignore_index=True)
last_names_df = pd.read_excel(os.path.join(DATA, "etternavn.xlsx"))

# Converting to sets
first_names = set(first_names_df.iloc[:, 0].dropna().str.strip())
last_names = set(last_names_df.iloc[:, 0].dropna().str.strip())

# Combining into one set
all_names = first_names.union(last_names)

# Compiling regex 
pattern = r"\b(" + "|".join(map(re.escape, all_names)) + r")s?\b"
name_regex = re.compile(pattern, flags=re.IGNORECASE)

all_names_lower = {n.lower() for n in all_names}

def basic_general_preprocess(series):
    # Convert to string
    s = series.fillna("").astype(str).str.lower()

    # Remove HTML tags
    s = s.apply(lambda x: BeautifulSoup(x, "html.parser").get_text())

    # Remove names (vectorized)
    pattern_names = r'\b(' + '|'.join(all_names_lower) + r')\b'
    s = s.str.replace(pattern_names, '', regex=True)

    # Remove any 'nan'
    s = s.str.replace(r'\b[nN][aA][nN]?\b', '', regex=True)

    # Remove decorative clutter (***, ~~~, ===, etc.)
    s = s.str.replace(r'[\*\~\=\^\_\-]', '', regex=True)

    # Remove extra spaces
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()

    return s

def basic_cleaning_df(df:pd.DataFrame) -> pd.DataFrame:
    df.copy()
    
    
    df['company_activity'] = basic_general_preprocess(df['company_activity'].fillna(' ').astype(str))
    df['company_name'] = basic_general_preprocess(df['company_name'].fillna(' ').astype(str))
    df['company_purpose'] = basic_general_preprocess(df['company_purpose'].fillna(' ').astype(str))
        
    return df

# Helper function to safely convert to string and strip
def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def remove_column_duplicates(df, print_similar=True, fuzzy_threshold=0.85):
    df = df.copy()

    for idx, row in df.iterrows():
        name = safe_str(row['company_name'])
        activity = safe_str(row['company_activity'])
        purpose = safe_str(row['company_purpose'])

        # Check activity similarity
        if activity and similar(name, activity) >= fuzzy_threshold:
            if print_similar:
                print(f"[Activity] Row {idx}: Name='{row['company_name']}', Activity='{row['company_activity']}', Similarity={similar(name, activity):.2f}")

        # Check purpose similarity
        max_purpose_sim = max(similar(name, purpose), similar(activity, purpose))
        if purpose and max_purpose_sim >= fuzzy_threshold:
            if print_similar:
                print(f"[Purpose] Row {idx}: Name='{row['company_name']}', Activity='{row['company_activity']}', Purpose='{row['company_purpose']}', Max Similarity={max_purpose_sim:.2f}")

    # Remove exact duplicates (as before)
    df['company_activity'] = df.apply(
        lambda row: '' if safe_str(row['company_activity']) == safe_str(row['company_name'])
                    else row['company_activity'],
        axis=1
    )

    df['company_purpose'] = df.apply(
        lambda row: '' if safe_str(row['company_purpose']) in [safe_str(row['company_name']),
                                                               safe_str(row['company_activity'])]
                    else row['company_purpose'],
        axis=1
    )

    return df


def general_preprocess(series, stop_word=False, lemmatization=False, stemming=False, batch_size=500):
   
    # Convert to string and lowercase
    s = series.fillna("").astype(str).str.lower()

    # Remove HTML tags
    s = s.apply(lambda x: BeautifulSoup(x, "html.parser").get_text())

    # Remove numbers and special characters (keep æøå)
    s = s.str.replace(r"\d+", "", regex=True)
    s = s.str.replace(r"[^\w\sæøå]", " ", regex=True)

    # Remove names (vectorized)
    pattern_names = r'\b(' + '|'.join(all_names_lower) + r')\b'
    s = s.str.replace(pattern_names, '', regex=True)

    # Remove any 'nan'
    s = s.str.replace(r'\b[nN][aA][nN]?\b', '', regex=True)

    # Remove extra spaces
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()

    # ---- NLP processing ----
    if stop_word or lemmatization:
        texts = s.tolist()
        processed_texts = []

        for doc in nlp.pipe(texts, batch_size=batch_size, n_process=1):
            tokens = []
            for token in doc:
                if stop_word and token.is_stop:
                    continue
                if token.is_punct:
                    continue
                tokens.append(token.lemma_ if lemmatization else token.text)
            processed_texts.append(" ".join(tokens))

        s = pd.Series(processed_texts, index=series.index)

    # ---- Stemming ----
    if stemming:
        s = s.apply(lambda text: " ".join([stemmer.stem(w) for w in text.split()]))

    return s


def column_subset(df):
    """Choosing a subset of columns """
    return df[["orgnr", "company_activity", "company_name", "company_purpose", "nace_21_code", "nace_21_description_nb", "nace_21_description_en"]]

def cleaning_df(df:pd.DataFrame, fasttext_data, stop_word=False, lemmatization=False, stemming=False) -> pd.DataFrame:
    df.copy()
    
    #if set(["orgnr","company_activity", "company_name", "nace_21_code"]).issubset(set(df.columns)):
    #    df = df[df.groupby("nace_21_code")["orgnr"].transform("count") >20]
    #    df = df[df['nace_21_code']!='00.000']

    # Ensure company_activity, company_name, company_purpose are strings
    df['company_activity'] = df['company_activity'].fillna("").astype(str)
    df['company_name'] = df['company_name'].fillna("").astype(str)
    df['company_purpose'] = df['company_purpose'].fillna("").astype(str)
    
    # general preprocess
    if fasttext_data==True:
        df['company_activity'] = general_preprocess(df['company_activity'].fillna(' '), stop_word=stop_word, lemmatization=lemmatization, stemming=stemming)
        df['company_name'] = general_preprocess(df['company_name'].fillna(' '), stop_word=stop_word, lemmatization=lemmatization, stemming=stemming)
        df['company_purpose'] = general_preprocess(df['company_purpose'].fillna(' '), stop_word=stop_word, lemmatization=lemmatization, stemming=stemming)
        
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
    #df_hier = pd.read_csv(HIERARCHY_DATA, dtype={'code':str}, sep=";", encoding="latin1")

    #df = df.fillna('')
    
    # Getting data for sn-codes, org-nr and text and filtering to only include groups with 10 > datapoints
    df = cleaning_df(df=df, fasttext_data=fasttext_data)
    df = remove_column_duplicates(df) 
    #df_sn25 = column_subset(df)

    # turning datasett into hierarkies
    #map_sec = dict(zip(df_hier[df_hier["level"]==2]["code"], df_hier[df_hier["level"]==2]["parentCode"]))
    #df_sn25_hier = derive_hier(df=df_sn25, subclass_col='nace_21_code', section_map=map_sec)
    #df = df.replace(['None'], '', regex=False)

    #df_sn25_hier.to_csv(save_file, index=False)
    
    df.to_csv(save_file, index=False)
    
    #df_hier = prune_tree(df=df_hier, cols=None)
    #df_hier.to_csv(HIERARCHY_DATA_PRUNED, index=False)
    return df


if __name__=='__main__':
    #80 percent dataset
    """
    train = pd.read_csv(DATA_BR_TRAIN, 
                             dtype={
                                 'nace_21_code':str,
                                 }, 
                             keep_default_na=False, 
                             na_values=[])
    #80 percent dataset
    test = pd.read_csv(DATA_BR_TEST, 
                             dtype={
                                 'nace_21_code':str,
                                 }, 
                             keep_default_na=False, 
                             na_values=[])

    df = pd.concat([train, test], ignore_index=True)

    #df_fx = run_preprocess(save_file=f"{DATA}data_preprocessed", df=df)

    df_lm = run_preprocess(save_file=f"{DATA}data_prep_lm_new", df=df, fasttext_data=False)

    df_lm.head()
   
    train_lm = pd.read_csv(os.path.join(DATA_LM_TR_VAL_TE, "train_dup.csv"), dtype={"nace_21_code": str})
    val_lm = pd.read_csv(os.path.join(DATA_LM_TR_VAL_TE, "val_dup.csv"), dtype={"nace_21_code": str})
    test_lm = pd.read_csv(os.path.join(DATA_LM_TR_VAL_TE, "test_dup.csv"), dtype={"nace_21_code": str})

    train_lm_dup=remove_column_duplicates(train_lm) 
    val_lm_dup=remove_column_duplicates(val_lm) 
    test_lm_dup=remove_column_duplicates(test_lm) 

    train_lm_dup.to_csv(os.path.join(DATA_LM_TR_VAL_TE, "train.csv"), index=False)
    val_lm_dup.to_csv(os.path.join(DATA_LM_TR_VAL_TE, "val.csv"), index=False)
    test_lm_dup.to_csv(os.path.join(DATA_LM_TR_VAL_TE, "test.csv"), index=False)
    """
    train_lm_dup = pd.read_csv(os.path.join(DATA_LM_TR_VAL_TE, "train.csv"), dtype={"nace_21_code": str})
    val_lm_dup = pd.read_csv(os.path.join(DATA_LM_TR_VAL_TE, "val.csv"), dtype={"nace_21_code": str})
    test_lm_dup = pd.read_csv(os.path.join(DATA_LM_TR_VAL_TE, "test.csv"), dtype={"nace_21_code": str})
    

    train_lm_cl=basic_cleaning_df(train_lm_dup) 
    val_lm_cl=basic_cleaning_df(val_lm_dup) 
    test_lm_cl=basic_cleaning_df(test_lm_dup) 

    train_lm_cl.to_csv(os.path.join(DATA_LM_TR_VAL_TE, "train_cl.csv"), index=False)
    val_lm_cl.to_csv(os.path.join(DATA_LM_TR_VAL_TE, "val_cl.csv"), index=False)
    test_lm_cl.to_csv(os.path.join(DATA_LM_TR_VAL_TE, "test_cl.csv"), index=False)
