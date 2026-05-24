import json
import argparse
import pandas as pd
import numpy as np
import os
from src.metrics import metrics, df_to_table, metrics_levels
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.dummy import DummyClassifier

from src.config import RANDOM_STATE, DATASETS, RES_DUMMY
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
seed_value=RANDOM_STATE
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


# --- Import data ---
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

# ===============================
# Train model
# ===============================

# Fasttext classifier on each hierarchy level
for hier, hier_level in zip(args.hierarchies, args.levels):
    print(f"\n=== Training on hierarchy: {hier} ===")
       
    # Dummy classifier
    dummy_clf = DummyClassifier(strategy="most_frequent", random_state=seed_value)
    dummy_clf.fit((train[args.input_colm].astype(str).agg(' '.join, axis=1)).tolist(), train[hier].to_list())
    preds_clf_test = dummy_clf.predict(test[args.input_colm].astype(str).agg(' '.join, axis=1).tolist())   
    

    # metrics on test dataset
    test_labels_arr = test[hier].tolist()

    print('preds_clf_test\n', preds_clf_test)
    if hier == 'nace_21_code':
        res_sub_test_dum, res_cl_test_dum, res_gro_test_dum, res_div_test_dum, res_sec_test_dum = metrics_levels(target=test_labels_arr, pred=preds_clf_test)    
        with PdfPages(os.path.join(RES_DUMMY, "dummy_test_results_sub.pdf")) as pdf:
            pdf.savefig(df_to_table(res_sub_test_dum, "Subclass Results"))
            pdf.savefig(df_to_table(res_cl_test_dum, "Class Results"))
            pdf.savefig(df_to_table(res_gro_test_dum, "Group Results"))
            pdf.savefig(df_to_table(res_div_test_dum, "Division Results"))
            pdf.savefig(df_to_table(res_sec_test_dum, "Section Results"))

        

    

    
    

    
