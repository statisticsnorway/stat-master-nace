import pandas as pd
from src.config import DATA_PATH, OLD_DATA, TRANSITION_DATA_PATH, HIERARCHY_DATA, RANDOM_STATE, SAVE_PATH
from src.exploration import explore_data_transition, count_per_category

# Treningsdata
df = pd.read_parquet(DATA_PATH)

# Overgangssett
df_transition = pd.read_csv(TRANSITION_DATA_PATH, dtype={'SN2025':str}, sep=";")


# NACE 2007 Hierarchi
df_hier = pd.read_csv(HIERARCHY_DATA,sep=";",encoding="latin-1")


# Analysing the transition from sn07 to sn25
df_trans_clean = cleaning_df(df_transition)
explore_data_transition(df_trans_clean)

# Number of companies under each SN subclass
groups_sn25 = count_per_category(df[["navn", "sn2025_1"]],"sn2025_1") 
print(f"number of companies under each NACE category {groups_sn25}")
