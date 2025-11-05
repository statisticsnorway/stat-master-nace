import pandas as pd
import numpy as np
import json
import os
from io import StringIO
import requests
from src.metrics import metrics, df_to_table, metrics_levels
from matplotlib.backends.backend_pdf import PdfPages

from src.config import DATA_PATH, OLD_DATA, TRANSITION_DATA_PATH, HIERARCHY_DATA, RANDOM_STATE, SAVE_PATH
from src.utils.baseline_utils import fasttext_input, wrong_preds_df, output_prep, pred_prep
from src.models.baseline import objective_cv, tune_fasttext_cv, train_hier_fasttext, train_hier_fasttext, predict_hier_fasttext, load_hier_fasttext_models, predict_hier_fasttext_lazy
from src.utils.utils import seed_everything

seed_value=42
thread=8
seed_everything(seed_value)

data_folder='data/'

df = pd.read_csv(f"{SAVE_PATH}/data_preprocessed.csv", dtype={'division':str, 'group':str, 'class':str, 'sn2025_1':str})
df_hier = pd.read_csv(StringIO(requests.get(HIERARCHY_DATA).text), delimiter=',')

hierarchies = ["section", "division", "group", "class", "sn2025_1"]
levels = [1, 2, 3, 4, 5]
hier_metrics_train = {}
df_wrong_res_hier_train={}
hier_metrics_test = {}
df_wrong_res_hier_test={}



input_col = ["tekst", "navn"]

# train and pred for each hierarchy
train_, test_, val_ = fasttext_input(
    df=df, columns=["sn2025_1"]+input_col, statify_column="sn2025_1", seed=seed_value,
train_file=f"{data_folder}train_hier_fasttext", test_file=f"{data_folder}test_hier_fasttext",
val_file=f"{data_folder}val_hier_fasttext")


train_input_txt, train_labels, train=pred_prep(train_, input_cols=input_col, output_cols=["sn2025_1"])
test_input_txt, test_labels, test=pred_prep(test_, input_cols=input_col, output_cols=["sn2025_1"])
val_input_txt, val_labels, val=pred_prep(val_, input_cols=input_col, output_cols=["sn2025_1"])


# hyperparameter tuning with k-fold cv

#best_params = tune_fasttext_cv(train, input_cols=input_col, seed=seed_value,thread=thread, output_cols=["sn2025_1"], n_trials=3)
with open("best_paramssn2025_1.json", "r") as f:
    best_params = json.load(f)

if os.path.exists("fasttext_hier_model_paths.json"):
    with open("fasttext_hier_model_paths.json", "r") as f:
        models_paths = json.load(f)
        #model=load_hier_fasttext_models(models_paths)
        pred_labels_train=[]
        for text_tr in train_input_txt:
            pred_labels_train.append(predict_hier_fasttext_lazy(models_paths, text_tr))
            
        pred_labels_test = []
        for text_te in test_input_txt:
            pred_labels_test.append(predict_hier_fasttext_lazy(models_paths, text_te))
        
else:
    model = train_hier_fasttext(df, input_col, label_hiers=hierarchies, seed=seed_value, best_params=best_params, thread=thread)



map_hier = dict(zip(df_hier[df_hier['level'] == 5]['code'], 
                df_hier[df_hier['level'] == 5]['name']))

# Predictions on train and test sets
#pred_labels_train = predict_hier_fasttext(model, train_input_txt)
#pred_labels_test = predict_hier_fasttext(model, test_input_txt)


# preparing output
pred_labels_train = output_prep(pred_labels_train)
train_labels_arr = np.array(train_labels)
pred_labels_test = output_prep(pred_labels_test)
test_labels_arr = np.array(test_labels)

#metrics
#df_results_train = metrics(train_labels_arr, pred_labels_train)
#df_results_test = metrics(test_labels_arr, pred_labels_test)


# hier metrics for subclass
res_cl_tr, res_gro_tr, res_div_tr = metrics_levels(target=train_labels_arr, pred=pred_labels_train)
res_cl_te, res_gro_te, res_div_te = metrics_levels(target=test_labels_arr, pred=pred_labels_test)

with PdfPages("results/hier_fsttxt_model/train_hier_results_sub.pdf") as pdf:
    pdf.savefig(df_to_table(res_cl_tr, "Class Results"))
    pdf.savefig(df_to_table(res_gro_tr, "Group Results"))
    pdf.savefig(df_to_table(res_div_tr, "Division Results"))

with PdfPages("results/hier_fsttxt_model/test_hier_results_sub.pdf") as pdf:
    pdf.savefig(df_to_table(res_cl_te, "Class Results"))
    pdf.savefig(df_to_table(res_gro_te, "Group Results"))
    pdf.savefig(df_to_table(res_div_te, "Division Results"))

# analyzing wrong predictions for train and test
df_res_train = wrong_preds_df(pred_labels_train, train_labels_arr, train_input_txt, map_hier)
df_res_test = wrong_preds_df(pred_labels_test, test_labels_arr, test_input_txt, map_hier)

# saving the results
df_res_train.to_csv(f"results/hier_fsttxt_model/hier_df_wrong_res_train.csv", index=False)
res_cl_tr.to_csv(f"results/hier_fsttxt_model/hier_metrics_train.csv")
df_res_test.to_csv(f"results/hier_fsttxt_model/hier_df_wrong_res_test.csv", index=False)
res_cl_te.to_csv(f"results/hier_fsttxt_model/hier_metrics_test.csv")

# printing them

print('df_results_train')
print(res_cl_tr)
print('df_results_test')
print(res_cl_te)
print("df_res_train")
print(df_res_train)
print("df_res_test")
print(df_res_test)


    
    

    
