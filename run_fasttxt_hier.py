import pandas as pd
import numpy as np
import json
import os
from io import StringIO
import requests
from src.metrics import df_to_table, metrics_levels, wrong_preds_df
from matplotlib.backends.backend_pdf import PdfPages

from src.config import HIERARCHY_DATA, RANDOM_STATE,THREAD, DATA, DATA_FX_TR_TE, DATA_FX_TR_VAL_TE, RES_HIER_M, JSON_FILES
from src.utils.baseline_utils import fasttext_input, output_prep, pred_prep
from src.models.baseline_hier import train_hier_fasttext, predict_hier_fasttext, load_hier_fasttext_models
from src.utils.utils import seed_everything

seed_value=RANDOM_STATE
thread=THREAD
seed_everything(seed_value)

df = pd.read_csv(f"{DATA}data_preprocessed.csv", dtype={'company_activity':str,'company_name':str,'division':str, 'group':str, 'class':str, 'nace_21_code':str,'nace_21_description_nb':str}, keep_default_na=False, na_values=[]).fillna("")
df_hier = pd.read_csv(StringIO(requests.get(HIERARCHY_DATA).text), delimiter=',')

hierarchies = ["section", "division", "group", "class", "nace_21_code"]
levels = [1, 2, 3, 4, 5]
hier_metrics_train = {}
df_wrong_res_hier_train={}
hier_metrics_test = {}
df_wrong_res_hier_test={}



input_col = ["company_activity", "company_name"]

# train and pred for each hierarchy
train_, test_, val_ = fasttext_input(
    df=df, columns=["nace_21_code"]+input_col, statify_column="nace_21_code", seed=seed_value,
train_file=f"train_hier_fasttext", test_file=f"test_hier_fasttext", val_file=f"val_hier_fasttext")


train_input_txt, train_labels, train=pred_prep(train_, input_cols=input_col, output_cols=["nace_21_code"])
test_input_txt, test_labels, test=pred_prep(test_, input_cols=input_col, output_cols=["nace_21_code"])
val_input_txt, val_labels, val=pred_prep(val_, input_cols=input_col, output_cols=["nace_21_code"])


# hyperparameter tuning with k-fold cv

if os.path.exists(f"{JSON_FILES}best_paramsdivision.json"):
    with open(f"{JSON_FILES}best_paramsdivision.json", "r") as f:
        best_params = json.load(f)
else:
    print("Load parameters on run_hier_explore.py script")

if os.path.exists(f"{JSON_FILES}fasttext_hier_model_paths.json"):
    with open(f"{JSON_FILES}fasttext_hier_model_paths.json", "r") as f:
        models_paths = json.load(f)
        model=load_hier_fasttext_models(models_paths)
else:
    model = train_hier_fasttext(df, input_col, label_hiers=hierarchies, seed=seed_value, best_params=best_params, thread=thread)


map_hier = dict(zip(df_hier[df_hier['level'] == 5]['code'], 
                df_hier[df_hier['level'] == 5]['name']))

# Predictions on train and test sets
pred_labels_train = predict_hier_fasttext(model, train_input_txt)
pred_labels_test = predict_hier_fasttext(model, test_input_txt)

# preparing output
pred_labels_train = output_prep(pred_labels_train)
train_labels_arr = np.array(train_labels)
pred_labels_test = output_prep(pred_labels_test)
test_labels_arr = np.array(test_labels)

# hier metrics for subclass
res_sub_tr, res_cl_tr, res_gro_tr, res_div_tr = metrics_levels(target=train_labels_arr, pred=pred_labels_train)
res_sub_te, res_cl_te, res_gro_te, res_div_te = metrics_levels(target=test_labels_arr, pred=pred_labels_test)


with PdfPages(f"{RES_HIER_M}train_hier_results_sub.pdf") as pdf:
    pdf.savefig(df_to_table(res_sub_tr, "Subclass Results"))
    pdf.savefig(df_to_table(res_cl_tr, "Class Results"))
    pdf.savefig(df_to_table(res_gro_tr, "Group Results"))
    pdf.savefig(df_to_table(res_div_tr, "Division Results"))

with PdfPages(f"{RES_HIER_M}test_hier_results_sub.pdf") as pdf:
    pdf.savefig(df_to_table(res_sub_te, "Subclass Results"))
    pdf.savefig(df_to_table(res_cl_te, "Class Results"))
    pdf.savefig(df_to_table(res_gro_te, "Group Results"))
    pdf.savefig(df_to_table(res_div_te, "Division Results"))

# analyzing wrong predictions for train and test
df_res_train = wrong_preds_df(pred_labels_train, train_labels_arr, train_input_txt, map_hier)
df_res_test = wrong_preds_df(pred_labels_test, test_labels_arr, test_input_txt, map_hier)

# saving the results
df_res_train.to_csv(f"{RES_HIER_M}hier_df_wrong_res_train.csv", index=False)
res_cl_tr.to_csv(f"{RES_HIER_M}hier_metrics_train.csv")
df_res_test.to_csv(f"{RES_HIER_M}hier_df_wrong_res_test.csv", index=False)
res_cl_te.to_csv(f"{RES_HIER_M}hier_metrics_test.csv")

# printing them

print('df_results_train')
print(res_cl_tr)
print('df_results_test')
print(res_cl_te)
print("df_res_train")
print(df_res_train)
print("df_res_test")
print(df_res_test)




    
    

    
