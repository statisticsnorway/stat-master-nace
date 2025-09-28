import pandas as pd
from src.config import DATA_PATH, OLD_DATA, TRANSITION_DATA_PATH, HIERARCHY_DATA, RANDOM_STATE, SAVE_PATH
from src.preprocess import column_subset, cleaning_df, df_hier_levels, derive_hier
from src.utils.baseline_utils import fasttext_input, wrong_preds_df, output_prep, pred_prep
from src.models.baseline import objective_cv, tune_fasttext_cv, fasttext_train_fn
from src.metrics import metrics
import numpy as np

# NACE 2007 Hierarchi
df_hier = pd.read_csv(HIERARCHY_DATA,sep=";",encoding="latin-1")
# Treningsdata
df = pd.read_parquet(DATA_PATH)

# Getting data for old sn-codes, org-nr and text and filtering to only include groups with 10> datapoints
df = cleaning_df(df)
df_sn25 = column_subset(df)

# splitting the df_hier into multiple DataFrames based on level
df_hiers=df_hier_levels(df=df_hier, column='level')
df_hier_div = df_hiers[2]
    
# turning datasett into 
map_sec = dict(zip(df_hier["code"], df_hier["parentCode"]))
df_sn25_hier = derive_hier(df=df_sn25, subclass_col='sn2025_1', section_map=map_sec)
print(df_sn25_hier)


hierarchies = ["section", "division", "group", "class", "sn2025_1"]
levels = [1, 2, 3, 4, 5]
hier_metrics_train = {}
df_wrong_res_hier_train={}
hier_metrics_test = {}
df_wrong_res_hier_test={}




for hier, hier_level in zip(hierarchies, levels):
    print(f"\n=== Training on hierarchy: {hier} ===")
    
    # train and pred for each hierarchy
    train, test = fasttext_input(
        df=df_sn25_hier, columns=[hier,"tekst","navn"], statify_column=hier,
    train_file=f"data_fasttext/train_fasttext_{hier}", test_file=f"data_fasttext/test_fasttext_{hier}")
    
    train_input_txt, train_labels, train=pred_prep(train, input_cols=["tekst","navn"], output_cols=[hier])
    test_input_txt, test_labels, test=pred_prep(test, input_cols=["tekst","navn"], output_cols=[hier])
    
    
    # hyperparameter tuning with k-fold cv
    best_params = tune_fasttext_cv(train, input_cols=["tekst","navn"], output_cols=[hier], n_trials=3)
    
    model = fasttext_train_fn(train_file=f"{SAVE_PATH}/data_fasttext/train_fasttext_{hier}.txt", best_params=best_params, model_file=f"{SAVE_PATH}/models_fasttext/model_nace_{hier}.txt") 
                        
    
    map_hier = dict(zip(df_hier[df_hier['level'] == hier_level]['code'], 
                    df_hier[df_hier['level'] == hier_level]['name']))
    
    # Predictions on train and test sets
    pred_labels_train, probs_train = model.predict(train_input_txt)
    pred_labels_test, probs_test = model.predict(test_input_txt)
    
    
    # preparing output
    pred_labels_train = output_prep(pred_labels_train)
    train_labels_arr = np.array(train_labels)
    pred_labels_test = output_prep(pred_labels_test)
    test_labels_arr = np.array(test_labels)
    
    #metrics
    df_results_train = metrics(train_labels_arr, pred_labels_train)
    df_results_test = metrics(test_labels_arr, pred_labels_test)
    hier_metrics_train[hier] = df_results_train
    hier_metrics_test[hier] = df_results_test

    
    
     # analyzing wrong predictions for train and test
    df_res_train = wrong_preds_df(pred_labels_train, train_labels_arr, train_input_txt, map_hier)
    df_res_test = wrong_preds_df(pred_labels_test, test_labels_arr, test_input_txt, map_hier)
    df_wrong_res_hier_train[hier] = df_res_train
    df_wrong_res_hier_test[hier] = df_res_test

    # saving the results
    df_res_train.to_csv(f"results/fasttext_each_hier_level/{hier}_df_wrong_res_train.csv", index=False)
    df_results_train.to_csv(f"results/fasttext_each_hier_level/{hier}_metrics_train.csv")
    df_res_test.to_csv(f"results/fasttext_each_hier_level/{hier}_df_wrong_res_test.csv", index=False)
    df_results_test.to_csv(f"results/fasttext_each_hier_level/{hier}_metrics_test.csv")

    # printing them
    
    print('df_results_train')
    print(df_results_train)
    print('df_results_test')
    print(df_results_test)
    print("df_res_train")
    print(df_res_train)
    print("df_res_test")
    print(df_res_test)
    
    """
    If we want to finetune new fasttext models, we must delete the saved models in SAVE_PATH and then rerun the run_fastext_model code.
    """
    

    
