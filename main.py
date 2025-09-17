import pandas as pd
from preprocessing import column_subset
from exploration import explore_data_transition


"""
Purpose: run the pipeline (like a recipe).

Should not contain raw cleaning logic.

Instead, it should just call functions you’ve already defined elsewhere.
"""


# Treningsdata
df = pd.read_parquet(DATA_PATH)

# Overgangssett
df_transition = pd.read_csv(TRANSITION_DATA_PATH, dtype{"SN2025":str}, sep=";")


# NACE 2007 Hierarchi
df_hier = pd.read_csv(HIERARCHY_DATA,sep=";",encoding="latin-1")

# Hent ut data for gamle nace-koder, org-nr og fritekst.
df_sn07 = column_subset(df)

explore_data_transition(df_transition)

val_input_txt, val_labels, test_input_txt, test_labels = fasttext_input(
    df=df_sn07, columns=["SN2007","tekst","navn"], statify_column="SN2007", 
    input_cols_val=["tekst","navn"], output_cols_val=["SN2007"], 
    input_cols_test=["tekst","navn"], output_cols_test=["SN2007"])


map_sn07 = mapping(df_overgang, "SN2007", "SN2007 Tittel")
save_mapping(map_sn07, "sn2007_mapping.json")



df_res = output_prep(pred_labels=pred_labels, true_labels=test_labels, input_text=test_input)