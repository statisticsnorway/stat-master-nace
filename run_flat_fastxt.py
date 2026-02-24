import json
import argparse
import pandas as pd
import numpy as np
import os
import fasttext
from src.metrics import metrics, df_to_table, metrics_levels, mean_std
from matplotlib.backends.backend_pdf import PdfPages

from src.config import RANDOM_STATE, THREAD, DATASETS, JSON_FILES, RES_FASTXT_FLAT,DATA_FX_TR_VAL_TE
from src.utils.baseline_utils import output_prep, pred_prep
from src.utils.utils import seed_everything


# --- argument parser ---
parser = argparse.ArgumentParser(description="Run FastText model with different tuning strategies")
parser.add_argument("--hierarchies", type=str, nargs="+", default=['section', 'division', 'group', 'class', 'nace_21_code'],
                    help="Specify by str which hierarchy level to train on.")
parser.add_argument("--levels", type=int, nargs="+", default=[1,2,3,4,5],
                    help="Specify by int which hierarchy level to train on.")
parser.add_argument("--input_colm", type=str, nargs="+", default=['company_activity','company_name'],
                    help="Specify which input column to train on.")
args = parser.parse_args()


# parameters
seed_list=[1,2,3,4,RANDOM_STATE]
thread=THREAD
hier='nace_21_code'
hier_level=5
exp_name='hyptune_cv_company_activity_company_name'

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

train_file = os.path.join(DATA_FX_TR_VAL_TE, f"train_{exp_name}.txt")

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

# loading best parameters
with open(os.path.join(JSON_FILES, f"best_params_{exp_name}.json"), 'r', encoding='utf-8') as file:
    best_params=json.load(file)

# model running and evaluation
macro_f1_list = []
weighted_f1_list = []
brier_list = []
Hf1_list = []

all_preds=[]


# Fasttext classifier-on each seed and return CI
for seed_value in seed_list:
    seed_everything(seed_value)
    print(f"\n=== Training on seed: {seed_value}===")

    # train and pred for each hierarchy    
    test_input_txt, test_labels=pred_prep(test, input_cols=args.input_colm, output_cols=hier)
    

    model = fasttext.train_supervised(
        input=train_file,
        seed=seed_value, 
        thread=thread,
        **best_params
    )
                    
    # Predictions on train and test sets
    pred_labels_test, probs_test = model.predict(test_input_txt)
    probs_test=np.array(probs_test)[:,0]

    # preparing output
    pred_labels_test = output_prep(pred_labels_test)
    
    df_res = pd.DataFrame({
        'preds':pred_labels_test,
        'pred_probs':probs_test,
        'seed_value': seed_value,
        },
        index=test.index)

    all_preds.append(df_res)    


    # -------------  metrics --------------------------
    df_results_test = metrics(test_labels, pred_labels_test)
    print('macro_f1 \n', df_results_test)
    macro_f1 = df_results_test.loc['macro', 'f1']
    weighted_f1 = df_results_test.loc['weighted', 'f1']
    brier = df_results_test.loc['score', 'brier_score']
    Hf1 = df_results_test.loc['score', 'HF1']

    macro_f1_list.append(macro_f1)
    weighted_f1_list.append(weighted_f1) 
    brier_list.append(brier)
    Hf1_list.append(Hf1)

    if seed_value == RANDOM_STATE:
        #--------------------
        #last seed analysis
        #--------------------
        res_sub_test, res_cl_test, res_gro_test, res_div_test, res_sec_test = metrics_levels(target=test_labels, pred=pred_labels_test)
        with PdfPages(os.path.join(RES_FASTXT_FLAT,f"test_results_sub.pdf")) as pdf:
            pdf.savefig(df_to_table(res_sub_test, "Subclass Results"))
            pdf.savefig(df_to_table(res_cl_test, "Class Results"))
            pdf.savefig(df_to_table(res_gro_test, "Group Results"))
            pdf.savefig(df_to_table(res_div_test, "Division Results"))
            pdf.savefig(df_to_table(res_sec_test, "Section Results"))

        df_results_test.to_csv(os.path.join(RES_FASTXT_FLAT,f"{hier}_metrics_test.csv"))


# all preds saved to a file
df_all_preds=pd.concat(all_preds)
df_all_preds.to_parquet(os.path.join(RES_FASTXT_FLAT,"preds_flat_fasttext.parquet"))



#-----------------
# standard dev.
#-----------------
summary = pd.DataFrame({
    "macro_f1": mean_std(macro_f1_list),
    "weighted_f1": mean_std(weighted_f1_list),
    "brier": mean_std(brier_list),
    "HF1": mean_std(Hf1_list)
}, index=["mean", "standard deviation"]).T

summary.to_csv(os.path.join(RES_FASTXT_FLAT,f"mean_std.csv"))



            
