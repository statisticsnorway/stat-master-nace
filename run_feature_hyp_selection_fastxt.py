import json
import argparse
import pandas as pd
import numpy as np
import os
import fasttext
from src.metrics import metrics
from matplotlib.backends.backend_pdf import PdfPages

from src.config import RES_FASTXT_FLAT, RANDOM_STATE, THREAD, DATASETS, DATA_FX_TR_VAL_TE, MODELS_FASTXT, JSON_FILES,RES_CV_TEXT,RES_CV_TEXT_NAME, RES_AUTO_TEXT_NAME,RES_AUTO_TEXT
from src.utils.baseline_utils import fasttext_dataprep, output_prep, pred_prep
from src.models.baseline import tune_fasttext_cv, fasttext_train_fn
from src.utils.utils import seed_everything



# --- argument parser ---
parser = argparse.ArgumentParser(description="Run FastText flat model")
parser.add_argument("--stage", type=str, choices=["features", "hyptune"], required=True,
                    help="Choose stage: 'test' for trying different columns, 'hyptune' for hyperparameter tuning.")
parser.add_argument("--mode", type=str, choices=["autotune", "cv"],# default=None,
                    help="Choose tuning method for hyptune stage: 'autotune' or 'cv'.")
parser.add_argument("--input_colm", type=str, nargs="+", default=['company_activity','company_name','company_purpose'],
                    help="Specify which input columns to train on.")
parser.add_argument("--evaluate_test",action="store_true",
                    help="Evaluate on test set (ONLY for final model).")

args = parser.parse_args()

testing=args.input_colm

print('input cols ', args.input_colm)
# --- Parameters Fasttext ---
FIXED_PARAMS = {
    "lr": 0.1,
    "epoch": 5,
    "dim": 100,
    "wordNgrams": 1,
    "loss": "softmax",
    "minCount": 1
}

seed_value = RANDOM_STATE
thread = THREAD
seed_everything(seed_value)


cols_name = "_".join(sorted(args.input_colm))
exp_name = f"{args.stage}_{args.mode}_{cols_name}_clean"
print(f"\nRunning experiment: {exp_name}\n")



# ===============================
# Paths
# ===============================
train_file = os.path.join(DATA_FX_TR_VAL_TE, f"train_{exp_name}")
val_file   = os.path.join(DATA_FX_TR_VAL_TE, f"val_{exp_name}")
#test_file  = os.path.join(DATA_FX_TR_VAL_TE, f"test_{exp_name}.txt")

model_path = os.path.join(MODELS_FASTXT, f"model_fasttext_{exp_name}.bin")
params_path = os.path.join(JSON_FILES, f"best_params_{exp_name}.json")

results_dir = os.path.join(RES_FASTXT_FLAT, "flat_fastxt_preps")
os.makedirs(results_dir, exist_ok=True)

metrics_test_path = os.path.join(results_dir, f"metrics_test_{exp_name}.csv")
metrics_val_path  = os.path.join(results_dir, f"metrics_val_{exp_name}.csv")



# ===============================
# Load data
# ===============================
dtype_map = {
    'company_activity': str,
    'company_name': str,
    'company_purpose': str,
    'nace_21_code': str,
    'nace_21_description_nb': str
}



# --- Import data ---
train = pd.read_csv(
    os.path.join(DATA_FX_TR_VAL_TE, f"train.csv"),
    dtype=dtype_map,
    keep_default_na=False, na_values=[]
).fillna("").set_index('orgnr')

val = pd.read_csv(
    os.path.join(DATA_FX_TR_VAL_TE, f"val.csv"),
    dtype=dtype_map,
    keep_default_na=False, na_values=[]
).fillna("").set_index('orgnr')




# ===============================
# Prepare FastText files
# ===============================
train_ = fasttext_dataprep(
    df=train,
    columns=['nace_21_code'] + args.input_colm,
    df_file=train_file
)

val_ = fasttext_dataprep(
    df=val,
    columns=['nace_21_code'] + args.input_colm,
    df_file=val_file
)


# ===============================
# Prepare prediction inputs
# ===============================
val_input_txt, val_labels = pred_prep(
    val_, args.input_colm, 'nace_21_code')



# ===============================
# Train / Tune model
# ===============================
if args.stage == "hyptune":

    if args.mode == "autotune":

        model = fasttext.train_supervised(
            input=f"{train_file}.txt",
            autotuneValidationFile=f"{val_file}.txt",
            seed=RANDOM_STATE,
            thread=thread
        )

        model.save_model(model_path)
        used_params = {"method": "autotune"}

    elif args.mode == "cv":

        best_params = tune_fasttext_cv(
            df_train=train_,
            n_splits=5,
            input_cols=args.input_colm,
            output_cols='nace_21_code',
            seed=RANDOM_STATE,
            thread=thread,
            n_trials=20,
        )

        with open(params_path, "w") as f:
            json.dump(best_params, f, indent=4)

        model = fasttext_train_fn(
            train_file=f"{train_file}.txt",
            seed=RANDOM_STATE,
            thread=thread,
            best_params=best_params,
            model_file=model_path
        )
        used_params=best_params

elif args.stage == "features":
    # features stage → load trained model
    model = fasttext.train_supervised(
            input=f"{train_file}.txt",
            seed=RANDOM_STATE,
            thread=thread,
            **FIXED_PARAMS
        )
    used_params = FIXED_PARAMS

else:
    print("Stage has to be specified")

# ===============================
# Predictions
# ===============================
pred_val, _ = model.predict(val_input_txt)
pred_val = output_prep(pred_val)
val_labels_arr = np.array(val_labels)


# ===============================
# Metrics
# ===============================
df_results_val = metrics(val_labels_arr, pred_val)
df_results_val.to_csv(metrics_val_path, index=True)


if args.evaluate_test:
    
    test_file  = os.path.join(DATA_FX_TR_VAL_TE, f"test_{exp_name}")
    test = pd.read_csv(
    os.path.join(DATA_FX_TR_VAL_TE, f"test.csv"),
    dtype=dtype_map,
    keep_default_na=False, na_values=[]
    ).fillna("").set_index('orgnr')

    test_ = fasttext_dataprep(
        df=test,
        columns=['nace_21_code'] + args.input_colm,
        df_file=f"{test_file}.txt"
    )

    test_input_txt, test_labels= pred_prep(
        test_, args.input_colm, 'nace_21_code')

    pred_test, _ = model.predict(test_input_txt)
        
    pred_test = output_prep(pred_test)
    test_labels_arr = np.array(test_labels)
    df_results_test = metrics(test_labels_arr, pred_test)

    df_results_test.to_csv(metrics_test_path, index=False)

print(f"Validation results saved → {metrics_val_path}")