import pandas as pd
from src.config import DATA_PATH, OLD_DATA, TRANSITION_DATA_PATH, HIERARCHY_DATA, RANDOM_STATE, SAVE_PATH
from src.preprocess import column_subset, cleaning_df
from src.utils.baseline_utils import fasttext_input, wrong_preds_df, output_prep
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


# Getting data for old sn-codes, org-nr and text and filtering to only include groups with 10> datapoints
df = cleaning_df(df)
df_sn25 = column_subset(df)

    
# Running fasttext
val_input, val_labels, test_input, test_labels = fasttext_input(
    df=df_sn25, columns=["sn2025_1","tekst","navn"], statify_column="sn2025_1", 
    input_cols_val=["tekst","navn"], output_cols_val=["sn2025_1"], 
    input_cols_test=["tekst","navn"], output_cols_test=["sn2025_1"])


map_sn25 = dict(zip(df_transition["SN2025"], df_transition["SN2025 Tittel"]))



# saving finetuned fasttext model and predicting using that
model = run_fasttext_model()
pred_labels, probs = model.predict(test_input)

# preparing output
pred_labels, test_labels = output_prep(pred_labels, test_labels)

df_results = metrics(test_labels, pred_labels)
print(df_results)

# Analysing output of fasttext
df_res = wrong_preds_df(pred_labels=pred_labels, true_labels=test_labels, input_text=test_input, map_file=map_sn25)

print(df_res)
df_res.to_csv("wrong_preds_df.csv", index=False)



# Visualising the distribution of each hierarchical level and number of distinct divisions
"""
Hvilke grupper er det som blir mest oppsplittet? 46, men denne gruppen har ikke størst mengde datapunkter. Gruppe 62 har størst mengde datapunkter, men kun 1 oppsplitting. 85 og 26 har 3 og 2 oppsplittinger i nevnt rekkefølge, men er de gruppene med færrest datapunkter. 

Finn måter å visualisere hierarkisk informasjon på en oversiktlig måte. 
Hvilke områder er det som ser vanskelige ut når det kommer til å gi prediksjoner? De som har færrest datapunkter men mer enn 1 oppsplittinger.
"""


# visualising the hierarchy of the SN codes and saving them in the image folder
