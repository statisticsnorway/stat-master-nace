import json
import argparse
import pandas as pd
import numpy as np
from io import StringIO
import requests
import os
from src.metrics import metrics, df_to_table, metrics_levels
from src.analyse_preds import wrong_preds_df, all_preds_df
from matplotlib.backends.backend_pdf import PdfPages

from src.config import HIERARCHY_DATA, RANDOM_STATE, THREAD, DATASETS, MODELS_FASTXT, JSON_FILES, RES_FASTXT_FLAT
from src.utils.baseline_utils import output_prep, pred_prep
from src.models.baseline import tune_fasttext_cv, fasttext_train_fn
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
    os.path.join(DATASETS, f"train.csv"),
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
    os.path.join(DATASETS, f"test.csv"),
    dtype=dtype_map,
    keep_default_na=False, na_values=[]
    ).fillna("").set_index('orgnr')



df_hier = pd.read_csv(StringIO(requests.get(HIERARCHY_DATA).text), delimiter=',')

# model running and evaluation
hier_metrics_train = {}
df_wrong_res_hier_train={}
df_wrong_res_hier_test={}



# Fasttext classifier on each hierarchy level
for hier, hier_level in zip(args.hierarchies, args.levels):
    print(f"\n=== Training on hierarchy: {hier} ===")
    map_hier = dict(zip(df_hier[df_hier['level'] == hier_level]['code'], 
                    df_hier[df_hier['level'] == hier_level]['name']))
    
    # train and pred for each hierarchy    
    train_input_txt, train_labels, train=pred_prep(train, input_cols=args.input_colm, output_cols=[hier])
    test_input_txt, test_labels, test=pred_prep(test, input_cols=args.input_colm, output_cols=[hier])

    
    # hyperparameter tuning with k-fold cv
    best_params = tune_fasttext_cv(df_train=train, input_cols=args.input_colm, output_cols=[hier], seed=seed_value,thread=thread, n_trials=20)
    with open(f'{JSON_FILES}best_params{hier}.json', 'w') as f:
        json.dump(best_params, f, indent=4)
    

    model = fasttext_train_fn(train_file=f"train_fasttext_hyptune_{hier}.txt", seed=seed_value, thread=thread,
                            best_params=best_params, model_file=f"{MODELS_FASTXT}model_nace_{hier}.bin") 
                    
    # Predictions on train and test sets
    pred_labels_test, probs_test = model.predict(test_input_txt)
    
    # preparing output
    pred_labels_test = output_prep(pred_labels_test)
    test_labels_arr = np.array(test_labels)
    
    #metrics
    df_results_test = metrics(test_labels_arr, pred_labels_test)

     # analyzing wrong predictions for train and test
    df_res_test = wrong_preds_df(pred_labels=pred_labels_test, true_labels=test_labels_arr, input_text=test_input_txt, mapping=map_hier)[1]

    # saving the results
    df_res_test.to_csv(f"{hier}_df_wrong_res_test.csv", index=False)
    df_results_test.to_csv(os.path.join(RES_FASTXT_FLAT,f"{hier}_metrics_test.csv"))



    
