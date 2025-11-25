import json
import argparse
import pandas as pd
import numpy as np
from io import StringIO
import requests
from src.metrics import metrics, df_to_table, metrics_levels
from src.analyse_preds import wrong_preds_df
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.dummy import DummyClassifier

from src.config import HIERARCHY_DATA, RANDOM_STATE, THREAD, DATA, DATA_FX_TR_VAL_TE, MODELS_FASTXT, JSON_FILES,RES_CV_TEXT,RES_CV_TEXT_NAME, RES_AUTO_TEXT_NAME,RES_AUTO_TEXT
from src.utils.baseline_utils import fasttext_input, output_prep, pred_prep
from src.models.baseline import run_fasttext_model, tune_fasttext_cv, fasttext_train_fn
from src.utils.utils import seed_everything


# --- argument parser ---
parser = argparse.ArgumentParser(description="Run FastText model with different tuning strategies")
parser.add_argument("--mode", type=str, choices=["autotune", "cv"], required=True,
                    help="Choose tuning method: 'autotune' for FastText's built-in auto-tune, or 'cv' for Optuna cross-validation tuning.")
parser.add_argument("--hierarchies", type=str, nargs="+", default=['section', 'division', 'group', 'class', 'nace_21_code'],
                    help="Specify by str which hierarchy level to train on, e.g. 'level1', 'level2', etc.")
parser.add_argument("--levels", type=int, nargs="+", default=[1,2,3,4,5],
                    help="Specify by int which hierarchy level to train on, e.g. 'level1', 'level2', etc.")
parser.add_argument("--input_colm", type=str, nargs="+", default=['company_activity','company_name'],
                    help="Specify which input column to train on, e.g. 'company_activity', 'company_name', etc.")
args = parser.parse_args()


# parameters
seed_value=RANDOM_STATE
thread=THREAD
seed_everything(seed_value)


# import of data
df = pd.read_csv(f"{DATA}data_preprocessed.csv",  dtype={'company_activity':str,'company_name':str,'division':str, 'group':str, 'class':str, 'nace_21_code':str,'nace_21_description_nb':str}, keep_default_na=False, na_values=[]).fillna("")
df_hier = pd.read_csv(StringIO(requests.get(HIERARCHY_DATA).text), delimiter=',')

# model running and evaluation
hier_metrics_train = {}
df_wrong_res_hier_train={}
df_wrong_res_hier_test={}

# results folder
if args.mode == "autotune" and args.input_colm==['company_activity']:
    res = RES_AUTO_TEXT
elif args.mode == "autotune" and args.input_colm==['company_activity', 'company_name']:
    res = RES_AUTO_TEXT_NAME
elif args.mode == "cv" and args.input_colm==['company_activity']:
    res = RES_CV_TEXT
elif args.mode == "cv" and args.input_colm==['company_activity','company_name']: 
    res = RES_CV_TEXT_NAME

# Fasttext classifier on each hierarchy level
for hier, hier_level in zip(args.hierarchies, args.levels):
    print(f"\n=== Training on hierarchy: {hier} ===")
    
    # train and pred for each hierarchy
    train_, test_, val_ = fasttext_input(
        df=df, columns=[hier] + args.input_colm, statify_column=hier, seed=seed_value,
    train_file=f"{DATA_FX_TR_VAL_TE}train_fasttext_hyptune_{hier}", test_file=f"{DATA_FX_TR_VAL_TE}test_fasttext_hyptune_{hier}",
    val_file=f"{DATA_FX_TR_VAL_TE}val_fasttext_hyptune_{hier}")
    
    train_input_txt, train_labels, train=pred_prep(train_, input_cols=args.input_colm, output_cols=[hier])
    test_input_txt, test_labels, test=pred_prep(test_, input_cols=args.input_colm, output_cols=[hier])
    val_input_txt, val_labels, val=pred_prep(val_, input_cols=args.input_colm, output_cols=[hier])
    
    # hyperparameter tuning with fasttext
    if args.mode == "autotune":
        model = run_fasttext_model(f"{MODELS_FASTXT}model_fasttext_auto_{hier}", 
                                f"train_fasttext_hyptune_{hier}", 
                                f"val_fasttext_hyptune_{hier}",
                                seed=seed_value, thread=thread)
        
    elif args.mode == "cv":
        # hyperparameter tuning with k-fold cv
        best_params = tune_fasttext_cv(df_train=train, input_cols=args.input_colm, output_cols=[hier], seed=seed_value,thread=thread, n_trials=20)

        with open(f'{JSON_FILES}best_params{hier}.json', 'w') as f:
            json.dump(best_params, f, indent=4)
        
        model = fasttext_train_fn(train_file=f"{DATA_FX_TR_VAL_TE}train_fasttext_hyptune_{hier}.txt", seed=seed_value, thread=thread,
                                best_params=best_params, model_file=f"{MODELS_FASTXT}model_nace_{hier}.bin") 
                        
    

    map_hier = dict(zip(df_hier[df_hier['level'] == hier_level]['code'], 
                    df_hier[df_hier['level'] == hier_level]['name']))
    
    
    # Dummy classifier
    dummy_clf = DummyClassifier(strategy="most_frequent", random_state=seed_value)
    dummy_clf.fit((train[args.input_colm].astype(str).agg(' '.join, axis=1)).tolist(), train[hier])
    preds_clf_train = dummy_clf.predict(train_input_txt)
    preds_clf_test = dummy_clf.predict(test_input_txt)
    
    
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
    # dummy
    df_dummy_train= metrics(train_labels_arr, preds_clf_train)
    df_dummy_test= metrics(test_labels_arr, preds_clf_test)  

    # hier metrics for subclass
    if hier == "nace_21_code":
        res_sub_tr, res_cl_tr, res_gro_tr, res_div_tr = metrics_levels(target=train_labels_arr, pred=pred_labels_train)
        res_sub_te, res_cl_te, res_gro_te, res_div_te = metrics_levels(target=test_labels_arr, pred=pred_labels_test)
        res_sub_tr_dum, res_cl_tr_dum, res_gro_tr_dum, res_div_tr_dum = metrics_levels(target=train_labels_arr, pred=df_dummy_train)
        res_sub_te_dum, res_cl_te_dum, res_gro_te_dum, res_div_te_dum = metrics_levels(target=test_labels_arr, pred=df_dummy_test)

        
        with PdfPages(f"{res}train_results_sub.pdf") as pdf:
            pdf.savefig(df_to_table(res_sub_tr, "Subclass Results"))
            pdf.savefig(df_to_table(res_cl_tr, "Class Results"))
            pdf.savefig(df_to_table(res_gro_tr, "Group Results"))
            pdf.savefig(df_to_table(res_div_tr, "Division Results"))
    
        with PdfPages(f"{res}test_results_sub.pdf") as pdf:
            pdf.savefig(df_to_table(res_sub_te, "Subclass Results"))
            pdf.savefig(df_to_table(res_cl_te, "Class Results"))
            pdf.savefig(df_to_table(res_gro_te, "Group Results"))
            pdf.savefig(df_to_table(res_div_te, "Division Results"))

        with PdfPages(f"{res}dummy_train_results_sub.pdf") as pdf:
            pdf.savefig(df_to_table(res_sub_tr, "Subclass Results"))
            pdf.savefig(df_to_table(res_cl_tr, "Class Results"))
            pdf.savefig(df_to_table(res_gro_tr, "Group Results"))
            pdf.savefig(df_to_table(res_div_tr, "Division Results"))
    
        with PdfPages(f"{res}dummy_test_results_sub.pdf") as pdf:
            pdf.savefig(df_to_table(res_sub_te, "Subclass Results"))
            pdf.savefig(df_to_table(res_cl_te, "Class Results"))
            pdf.savefig(df_to_table(res_gro_te, "Group Results"))
            pdf.savefig(df_to_table(res_div_te, "Division Results"))
    
     # analyzing wrong predictions for train and test
    df_res_train = wrong_preds_df(pred_labels_train, train_labels_arr, train_input_txt, map_hier)
    df_res_test = wrong_preds_df(pred_labels_test, test_labels_arr, test_input_txt, map_hier)


    # saving the results
    df_res_train.to_csv(f"{res}{hier}_df_wrong_res_train.csv", index=False)
    df_results_train.to_csv(f"{res}{hier}_metrics_train.csv")
    df_res_test.to_csv(f"{res}{hier}_df_wrong_res_test.csv", index=False)
    df_results_test.to_csv(f"{res}{hier}_metrics_test.csv")

    #df_res_train.to_csv(f"{res}dummy_df_wrong_res_train.csv", index=False)
    df_dummy_train.to_csv(f"{res}dummy_metrics_train.csv")
    #df_res_test.to_csv(f"{res}dummy_df_wrong_res_test.csv", index=False)
    df_dummy_test.to_csv(f"{res}dummy_metrics_test.csv")

    

    
    

    
