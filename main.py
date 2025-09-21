import pandas as pd
from src.config import DATA_PATH, OLD_DATA, TRANSITION_DATA_PATH, HIERARCHY_DATA, RANDOM_STATE, SAVE_PATH
from src.preprocess import column_subset, cleaning_df, mapping, save_mapping, load_mapping
from src.exploration import explore_data_transition, count_per_category
from src.utils.baseline_utils import fasttext_input, wrong_preds_df
from src.models.baseline import run_fasttext_model
from src.metrics import metrics


"""
to activate environment: source /home/stud-msh/stat-master-nace/.venv/bin/activate
Purpose: run the pipeline (like a recipe).

Should not contain raw cleaning logic.

Instead, it should just call functions you’ve already defined elsewhere.
"""


# Treningsdata
df = pd.read_parquet(DATA_PATH)

# Overgangssett
df_transition = pd.read_csv(TRANSITION_DATA_PATH, dtype={'SN2025':str}, sep=";")


# NACE 2007 Hierarchi
df_hier = pd.read_csv(HIERARCHY_DATA,sep=";",encoding="latin-1")

# Getting data for old sn-codes, org-nr and text and filtering to only include groups with 10> datapoints
df = cleaning_df(df)
df_sn25 = column_subset(df)

# Analysing the transition from sn07 to sn25
df_trans_clean = cleaning_df(df_transition)
explore_data_transition(df_trans_clean)

# Number of companies under each SN subclass
groups_sn25 = count_per_category(df[["navn", "sn2025_1"]],"sn2025_1") 
print(f"number of companies under each NACE category {groups_sn25}")

    
# Running fasttext
val_input, val_labels, test_input, test_labels = fasttext_input(
    df=df_sn25, columns=["sn2025_1","tekst","navn"], statify_column="sn2025_1", 
    input_cols_val=["tekst","navn"], output_cols_val=["sn2025_1"], 
    input_cols_test=["tekst","navn"], output_cols_test=["sn2025_1"])


map_sn25 = mapping(df_transition, "SN2025", "SN2025 Tittel", "sn2025_mapping.json")
save_mapping(map_sn25, "sn2025_mapping.json")


# saving finetuned fasttext model and predicting using that
model = run_fasttext_model()
pred_labels, probs = model.predict(test_input)

# Analysing output of fasttext
df_res = wrong_preds_df(pred_labels=pred_labels, true_labels=test_labels, input_text=test_input, mapping_file="sn2025_mapping.json")
df_results = metrics(test_labels, pred_labels)



# Hierarkies of SN code
sn_hiers = ["Division", "Group", "Class", "Subclass"]

# Visualising the distribution of each hierarchical level and number of distinct divisions
"""
Hvilke grupper er det som blir mest oppsplittet? 46, men denne gruppen har ikke størst mengde datapunkter. Gruppe 62 har størst mengde datapunkter, men kun 1 oppsplitting. 85 og 26 har 3 og 2 oppsplittinger i nevnt rekkefølge, men er de gruppene med færrest datapunkter. 

Finn måter å visualisere hierarkisk informasjon på en oversiktlig måte. 
Hvilke områder er det som ser vanskelige ut når det kommer til å gi prediksjoner? De som har færrest datapunkter men mer enn 1 oppsplittinger.
"""


# visualising the hierarchy of the SN codes and saving them in the image folder
