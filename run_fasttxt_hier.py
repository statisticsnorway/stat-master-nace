import pandas as pd
import numpy as np
from src.metrics import metrics, df_to_table, metrics_levels
from matplotlib.backends.backend_pdf import PdfPages

from src.config import DATA_PATH, OLD_DATA, TRANSITION_DATA_PATH, HIERARCHY_DATA, RANDOM_STATE, SAVE_PATH
from src.utils.baseline_utils import fasttext_input, wrong_preds_df, output_prep, pred_prep
from src.models.baseline import objective_cv, tune_fasttext_cv, train_hier_fasttext, predict_hier_fasttext, train_hier_fasttext, predict_hier_fasttext
from src.utils.utils import seed_everything

seed_value=42
thread=8
seed_everything(seed_value)

df_sn25_hier = pd.read_csv(f"{SAVE_PATH}/data_fasttext/data_preprocessed.csv")
df_hier = pd.read_csv(HIERARCHY_DATA,sep=";",encoding="latin-1")

hierarchies = ["section", "division", "group", "class", "sn2025_1"]
levels = [1, 2, 3, 4, 5]
hier_metrics_train = {}
df_wrong_res_hier_train={}
hier_metrics_test = {}
df_wrong_res_hier_test={}



input_colm = ["tekst"]

# train and pred for each hierarchy
train_, test_, val_ = fasttext_input(
    df=df_sn25_hier, columns=[hierarchies,"tekst", "navn"], statify_column="sn2025_1", seed=seed_value,
train_file=f"data_fasttext/train_hier_fasttext", test_file=f"data_fasttext/test_hier_fasttext",
val_file=f"data_fasttext/val_hier_fasttext")

train_input_txt, train_labels, train=pred_prep(train_, input_cols=input_colm, output_cols=["sn2025_1"])
test_input_txt, test_labels, test=pred_prep(test_, input_cols=input_colm, output_cols=["sn2025_1"])
val_input_txt, val_labels, val=pred_prep(val_, input_cols=input_colm, output_cols=["sn2025_1"])

# hyperparameter tuning with fasttext
"""
model = run_fasttext_model(f"models_fasttext/model_fasttext_{hier}", 
f"data_fasttext/train_fasttext_hyptune_{hier}", seed= seed_value, thread=thread,
f"data_fasttext/val_fasttext_hyptune_{hier}")
"""
# hyperparameter tuning with k-fold cv

best_params = tune_fasttext_cv(train, input_cols=input_colm, seed=seed_value,thread=thread, output_cols=["sn2025_1"], n_trials=3)

model = train_hier_fasttext(df, input_col, label_hier, seed, best_params=, thread=None)

map_hier = dict(zip(df_hier[df_hier['level'] == hier_level]['code'], 
                df_hier[df_hier['level'] == hier_level]['name']))

# Predictions on train and test sets
pred_labels_train = predict_hier_fasttext(models, train_input_txt)
pred_labels_test = predict_hier_fasttext(models, test_input_txt)


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

with PdfPages("train_hier_results_sub.pdf") as pdf:
    pdf.savefig(df_to_table(res_cl_tr, "Class Results"))
    pdf.savefig(df_to_table(res_gro_tr, "Group Results"))
    pdf.savefig(df_to_table(res_div_tr, "Division Results"))

with PdfPages("test_hier_results_sub.pdf") as pdf:
    pdf.savefig(df_to_table(res_cl_te, "Class Results"))
    pdf.savefig(df_to_table(res_gro_te, "Group Results"))
    pdf.savefig(df_to_table(res_div_te, "Division Results"))

# analyzing wrong predictions for train and test
df_res_train = wrong_preds_df(pred_labels_train, train_labels_arr, train_input_txt, map_hier)
df_res_test = wrong_preds_df(pred_labels_test, test_labels_arr, test_input_txt, map_hier)

# saving the results
df_res_train.to_csv(f"results/fasttext_hier_text/hier_df_wrong_res_train.csv", index=False)
df_results_train.to_csv(f"results/fasttext_hier_text/hier_metrics_train.csv")
df_res_test.to_csv(f"results/fasttext_hier_text/hier_df_wrong_res_test.csv", index=False)
df_results_test.to_csv(f"results/fasttext_hier_text/hier_metrics_test.csv")

# printing them

print('df_results_train')
print(df_results_train)
print('df_results_test')
print(df_results_test)
print("df_res_train")
print(df_res_train)
print("df_res_test")
print(df_res_test)


    
    

    
