import pandas as pd
from src.config import DATA_PATH, OLD_DATA, TRANSITION_DATA_PATH, HIERARCHY_DATA, RANDOM_STATE, SAVE_PATH
from src.preprocess import column_subset, cleaning_df, df_hier_levels, derive_hier
from src.utils.baseline_utils import fasttext_input, wrong_preds_df, output_prep
from src.models.baseline import run_fasttext_model
from src.metrics import metrics

# NACE 2007 Hierarchi
df_hier = pd.read_csv(HIERARCHY_DATA,sep=";",encoding="latin-1")
# Treningsdata
df = pd.read_parquet(DATA_PATH)

# Getting data for old sn-codes, org-nr and text and filtering to only include groups with 10> datapoints
df = cleaning_df(df)
df_sn25 = column_subset(df)

# splitting the df_hier into multiple DataFrames based on level
df_hiers=df_hier_levels(df=df_hier, column='level')
#df_hier_1, df_hier_2, df_hier_3, df_hier_4, df_hier_5 = df_hiers[1],df_hiers[2],df_hiers[3],df_hiers[4],df_hiers[5] # TODO: må kanskje slette noen hvis jeg ikke får bruk for de.
df_hier_div = df_hiers[2]
    
# turning datasett into 
map_sec = dict(zip(df_hier["code"], df_hier["parentCode"]))
df_sn25_hier = derive_hier(df=df_sn25, subclass_col='sn2025_1', section_map=map_sec)
print(df_sn25_hier)


hierarchies = ["section", "division", "group", "class"]
levels = [1, 2, 3, 4]
hier_metrics = {}
df_wrong_res_hier={}

for hier, hier_level in zip(hierarchies, levels):
    print(f"\n=== Training on hierarchy: {hier} ===")
    
    # train and pred for each hierarchy
    val_input, val_labels, test_input, test_labels = fasttext_input(
        df=df_sn25_hier, columns=[hier,"tekst","navn"], statify_column=hier, 
    input_cols_val=["tekst","navn"], output_cols_val=[hier], 
    input_cols_test=["tekst","navn"], output_cols_test=[hier],
    train_file=f"train_fasttext_{hier}", val_file=f"val_fasttext_{hier}", test_file=f"test_fasttext_{hier}")
    model = run_fasttext_model(model_file=f"model_nace_{hier}", train_file=f"train_fasttext_{hier}", 
                               val_file=f"val_fasttext_{hier}")
    
    map_hier = dict(zip(df_hier[df_hier['level'] == hier_level]['code'], 
                    df_hier[df_hier['level'] == hier_level]['name']))

    # predicting
    pred_labels, probs = model.predict(test_input)

    # preparing output
    pred_labels, test_labels = output_prep(pred_labels, test_labels)

    df_results = metrics(test_labels, pred_labels)
    
    hier_metrics[hier]=df_results
    
    # Analysing output of fasttext
    df_res = wrong_preds_df(pred_labels=pred_labels, true_labels=test_labels, input_text=test_input, map_file=map_hier)
    df_wrong_res_hier[hier]=df_res
    print(df_wrong_res_hier)
    print(hier_metrics)
    df_res.to_csv(f"{hier}_df_wrong_res.csv", index=False)
    df_results.to_csv(f"{hier}_metrics.csv", index=False)

    """
    If we want to finetune new fasttext models, we must delete the saved models in SAVE_PATH and then rerun the run_fastext_model code.
    """