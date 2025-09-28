# imported libraries
import fasttext
import optuna
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score
from src.config import DATA_PATH, OLD_DATA, TRANSITION_DATA_PATH, HIERARCHY_DATA, RANDOM_STATE, SAVE_PATH
from src.utils.baseline_utils import output_prep
import sklearn.metrics as m



def objective_cv(trial, df_train, input_cols, output_cols, n_splits=3):
    """Optuna objective using simple k-fold CV on training data."""
    # Suggested hyperparameters
    lr = trial.suggest_float("lr", 0.01, 0.5, log=True)
    epoch = trial.suggest_int("epoch", 5, 30)
    wordNgrams = trial.suggest_int("wordNgrams", 1, 3)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    X = df_train[input_cols].astype(str).agg(' '.join, axis=1).tolist()
    y = df_train[output_cols[0]].astype(str).tolist()
    
    for train_idx, val_idx in skf.split(X, y):
        X_train_fold = [X[i] for i in train_idx]
        y_train_fold = [y[i] for i in train_idx]
        X_val_fold = [X[i] for i in val_idx]
        y_val_fold = [y[i] for i in val_idx]
        
        # Preparing temporary training file for FastText
        temp_train_df = pd.DataFrame({"text": X_train_fold, "label": y_train_fold})
        temp_train_df["fasttext_format"] = "__label__" + temp_train_df["label"] + " " + temp_train_df["text"]
        temp_train_file = "temp_train.txt"
        temp_train_df["fasttext_format"].to_csv(temp_train_file, index=False, header=False)
        
        # Training FastText
        model = fasttext.train_supervised(
            input=temp_train_file,
            lr=lr,
            epoch=epoch,
            wordNgrams=wordNgrams,
            verbose=0
        )
        
        # Predicting and evaluating on validation fold
        pred_labels = output_prep(model.predict(X_val_fold)[0])
        y_val_fold_arr = np.array(y_val_fold)  
        
        f1_macro_score = m.f1_score(y_val_fold_arr, pred_labels, zero_division=np.nan, average='macro')
        scores.append(f1_macro_score)
        
        os.remove(temp_train_file)  # cleanup
    
    return np.mean(scores)  # average CV score

def tune_fasttext_cv(df_train, input_cols, output_cols, n_trials=20, n_splits=3):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective_cv(trial, df_train, input_cols, output_cols, n_splits), n_trials=n_trials)
    print("Best hyperparameters:", study.best_params)
    return study.best_params

def fasttext_train_fn(train_file, best_params, model_file):
    model = fasttext.train_supervised(
        input=train_file,
        lr=best_params["lr"],
        epoch=best_params["epoch"],
        wordNgrams=best_params["wordNgrams"]
    )
    model.save_model(model_file)
    return model






# Hierarchical model

def hier_fasttext():
    model_div = run_fasttext_model(train_div, val_div)
    
    
def fasttext_each_hier():
    ...
