import json
import argparse
import pandas as pd
import numpy as np
import os
import fasttext
from src.metrics import metrics
from src.config import HIERARCHY_DATA, RANDOM_STATE, THREAD, DATASETS, JSON_FILES, RES_FASTXT_FLAT,DATA_FX_TR_VAL_TE
from src.utils.baseline_utils import output_prep, pred_prep,fasttext_dataprep
from src.models.baseline import tune_fasttext_cv
from src.utils.utils import seed_everything


# --- argument parser ---
parser = argparse.ArgumentParser(description="Run FastText model with different tuning strategies")
parser.add_argument("--hierarchies", type=str, nargs="+", default=['section', 'division', 'group', 'class'],
                    help="Specify by str which hierarchy level to train o.")
parser.add_argument("--levels", type=int, nargs="+", default=[1,2,3,4],
                    help="Specify by int which hierarchy level to train on.")
parser.add_argument("--input_colm", type=str, nargs="+", default=['company_activity','company_name'],
                    help="Specify which input column to train on.")
args = parser.parse_args()


# parameters
seed_value=RANDOM_STATE
thread=THREAD
seed_everything(seed_value)


# ===============================
# Load data
# ===============================
dtype_map = {
    'company_activity': str,
    'company_name': str,
    'company_purpose': str,
    'nace_21_code': str,
    'division':str, 
    'group':str, 
    'class':str,
    'nace_21_description_nb': str
}

train = pd.read_csv(
    os.path.join(DATA_FX_TR_VAL_TE, f"train.csv"),
    dtype=dtype_map,
    keep_default_na=False, na_values=[]
).fillna("").set_index('orgnr')

"""
val = pd.read_csv(
    os.path.join(DATASETS, f"val.csv"),
    dtype=dtype_map,
    keep_default_na=False, na_values=[]
).fillna("").set_index('orgnr')
"""

test = pd.read_csv(
    os.path.join(DATA_FX_TR_VAL_TE, f"test.csv"),
    dtype=dtype_map,
    keep_default_na=False, na_values=[]
    ).fillna("").set_index('orgnr')

df_hier = pd.read_csv(HIERARCHY_DATA, dtype={'code':str}, sep=";", encoding="latin1")

# Fasttext classifier on each hierarchy level
for hier, hier_level in zip(args.hierarchies, args.levels):
    print(f"\n=== Training on hierarchy: {hier} ===")
    
    train_file = os.path.join(DATA_FX_TR_VAL_TE, f'train_{hier}')

    train_=fasttext_dataprep(df=train, 
                      columns=[hier] + args.input_colm, 
                      df_file=train_file)
    
    # train and pred for each hierarchy    
    test_input_txt, test_labels=pred_prep(test, input_cols=args.input_colm, output_cols=hier)

    # hyperparameter tuning with k-fold cv
    best_params = tune_fasttext_cv(df_train=train, input_cols=args.input_colm, output_cols=hier, seed=seed_value,thread=thread, n_trials=20)
    with open(os.path.join(JSON_FILES,f'best_params_{hier}_clean.json'), 'w') as f:
        json.dump(best_params, f, indent=4)


    model = fasttext.train_supervised(
        input=train_file+'.txt',
        lr=best_params["lr"],
        epoch=best_params["epoch"],
        wordNgrams=best_params["wordNgrams"],
        seed=seed_value, 
        thread=thread
    )
                    
    # Predictions on train and test sets
    pred_labels_test, probs_test = model.predict(test_input_txt)
    
    # preparing output
    pred_labels_test = output_prep(pred_labels_test)
    test_labels_arr = np.array(test_labels)
    
    #metrics
    df_results_test = metrics(test_labels_arr, pred_labels_test)
    df_results_test.to_csv(os.path.join(RES_FASTXT_FLAT,f"{hier}_metrics_test_clean.csv"))



    
