import pandas as pd
import json
import os
from src.metrics import df_to_table, metrics_levels, mean_std
from matplotlib.backends.backend_pdf import PdfPages

from src.config import RANDOM_STATE,THREAD, DATA_FX_TR_VAL_TE, JSON_FILES, DATASETS, RES_HIER_FASTXT
from src.utils.baseline_utils import pred_prep
from src.models.baseline_hier import train_hier_fasttext, predict_hier_fasttext, load_hier_fasttext_models
from src.utils.utils import seed_everything
from src.metrics import metrics, metrics_levels


seed_list=[1,2,3,4,RANDOM_STATE]
thread=THREAD
exp_name='hyptune_cv_company_activity_company_name'
hierarchies = ["root", "section", "division", "group", "class", "nace_21_code"]
input_col = ["company_activity", "company_name"]


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

# ===============================
# initialising lists
# ===============================
macro_f1_list = []
weighted_f1_list = []
brier_list = []
Hf1_list = []
all_preds=[]

test_input_txt, test_labels=pred_prep(test, input_cols=input_col, output_cols="nace_21_code")
print('test labels', test_labels)
if os.path.exists(os.path.join(JSON_FILES,'best_paramssection.json')):
    with open(os.path.join(JSON_FILES,'best_paramssection.json'), "r") as f:
        best_params = json.load(f)
else:
    print("Load parameters on run_hier_explore.py script")

# ===============================
# running model
# ===============================

for seed_value in seed_list:
    print(f"\n=== Training on seed: {seed_value}===")
    seed_everything(seed_value)
    

    # loading parameters
    
    """if os.path.exists(os.path.join(JSON_FILES,"fasttext_hier_model_paths.json")):
        with open(os.path.join(JSON_FILES,"fasttext_hier_model_paths.json"), "r") as f:
            models_paths = json.load(f)
            models=load_hier_fasttext_models(models_paths)
    else:"""
    models_paths = train_hier_fasttext(train, input_col, label_hiers=hierarchies, seed=seed_value, best_params=best_params, thread=thread)
    models=load_hier_fasttext_models(models_paths)


    # Predictions on train and test sets
    pred_labels_test,probs_test = predict_hier_fasttext(models, test_input_txt)
    print('preds ', pred_labels_test)
    df_res = pd.DataFrame({
        'preds':pred_labels_test,
        'pred_probs':probs_test,
        'seed_value': seed_value,
        },
        index=test.index)

    all_preds.append(df_res)    

    # ===============================
    # metrics
    # ===============================
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
        res_sub_test, res_cl_test, res_gro_test, res_div_test, res_sec_test = metrics_levels(target=test_labels, pred=pred_labels_test)
        with PdfPages(os.path.join(RES_HIER_FASTXT,f"test_hier_results.pdf")) as pdf:
            pdf.savefig(df_to_table(res_sub_test, "Subclass Results"))
            pdf.savefig(df_to_table(res_cl_test, "Class Results"))
            pdf.savefig(df_to_table(res_gro_test, "Group Results"))
            pdf.savefig(df_to_table(res_div_test, "Division Results"))
            pdf.savefig(df_to_table(res_sec_test, "Section Results"))

# ===============================
# all seeds results
# ===============================
# all preds saved to a file
df_all_preds=pd.concat(all_preds)

# Ensure directory exists
os.makedirs(RES_HIER_FASTXT, exist_ok=True)

try:
    # Try saving as Parquet
    df_all_preds.to_parquet(os.path.join(RES_HIER_FASTXT, "preds_hier_fasttext.parquet"))
    print("Saved predictions as Parquet")
except Exception as e:
    print(f"Parquet save failed: {e}")
    # Fall back to CSV
    df_all_preds.to_csv(os.path.join(RES_HIER_FASTXT, "preds_hier_fasttext.csv"), index=True)
    print("Saved predictions as CSV instead")


#-----------------
# standard dev.
#-----------------
summary = pd.DataFrame({
    "macro_f1": mean_std(macro_f1_list),
    "weighted_f1": mean_std(weighted_f1_list),
    "brier": mean_std(brier_list),
    "HF1": mean_std(Hf1_list)
}, index=["mean", "standard deviation"]).T

summary.to_csv(os.path.join(RES_HIER_FASTXT,f"mean_std.csv"))


    
    

    
