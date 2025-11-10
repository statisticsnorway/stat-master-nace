import pandas as pd
from io import StringIO
import requests

from src.config import TRANSITION_DATA_PATH, HIERARCHY_DATA, DATA
from src.exploration import explore_data_transition, count_per_category, freq_table
from src.preprocess import prune_tree#,cleaning_df

input_cols = ['tekst', 'navn']

# Treningsdata
df = pd.read_csv(f"{DATA}data_preprocessed.csv", dtype={'division':str, 'group':str, 'class':str, 'sn2025_1':str})


# Overgangssett
df_transition = pd.read_csv(TRANSITION_DATA_PATH, dtype={'SN2025':str}, sep=";")


# NACE 2025 Hierarchi
df_hier = pd.read_csv(StringIO(requests.get(HIERARCHY_DATA).text), delimiter=',')
map_hier = dict(zip(df_hier['code'], df_hier['name']))

# Analysing the transition from sn07 to sn25
#df_trans_clean = cleaning_df(df_transition)
explore_data_transition(df_transition)

df = prune_tree(df)


hierarchies=['section','division', 'group', 'class', 'sn2025_1']

# Number of companies under each SN class on each level
for hier in hierarchies:
    groups_hier = count_per_category(df[["tekst","navn", hier]],hier)  ################# feil tror jeg
    print(f"number of companies under each NACE category in {hier} is \n {groups_hier}")
    ################## print name of the companies?


# Tf-idf for each hierarchy level:
for hier in hierarchies:
    tfidf_means_df, max_tfidf=freq_table(df=df, mapping=map_hier, input_cols=input_cols, hier=hier)
    tfidf_means_df.to_csv(f"results/data_explore/tf_idf/tfidf_{hier}.csv")
    max_tfidf.to_csv(f"results/data_explore/max_tf_idf/max_tfidf_{hier}.csv")

  
            
