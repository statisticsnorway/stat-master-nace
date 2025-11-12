# imported libraries
import fasttext
import optuna
import numpy as np
import pandas as pd
import os
import json

from sklearn.model_selection import StratifiedKFold
from src.config import M_F_H, DATA, JSON_FILES
from src.utils.baseline_utils import output_prep
import sklearn.metrics as m

"""
def run_fasttext_model(model_file, train_file, val_file, seed, thread=None):
    #if os.path.exists(f"{SAVE_PATH}/{model_file}.bin"):
    #    model = fasttext.load_model(f"{SAVE_PATH}/{model_file}.bin")
        
    #else:
    # Skipgram model, finetuned:
    model = fasttext.train_supervised(input=f"{SAVE_PATH}/{train_file}.txt", 
                                      autotuneValidationFile=f"{SAVE_PATH}/{val_file}.txt",# Hyperparameter tuning by using "autotuneValidationFile" parameter
                                     seed=seed, thread=thread) 
    #Saving the model
    model.save_model(f"{SAVE_PATH}/{model_file}.bin")
    return model
"""


def objective_cv(trial, df_train, input_cols, output_cols, seed, thread, n_splits=3):
    """Optuna objective using simple k-fold CV on training data."""
    # Suggested hyperparameters
    lr = trial.suggest_float("lr", 0.01, 0.2, log=True)
    epoch = trial.suggest_int("epoch", 5, 30)
    wordNgrams = trial.suggest_int("wordNgrams", 1, 3)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []

    X = df_train[input_cols[0]].astype(str).tolist()
    y = df_train[output_cols[0]].astype(str).tolist()
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold = [X[i] for i in train_idx]
        y_train_fold = [y[i] for i in train_idx]
        X_val_fold = [X[i] for i in val_idx]
        y_val_fold = [y[i] for i in val_idx]
        
        # Temporary training file for FastText
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
            verbose=0,
            thread=thread
        )
        
        # Predicting and evaluating on validation fold
        pred_labels = output_prep(model.predict(X_val_fold)[0])
        y_val_fold_arr = np.array(y_val_fold)          
        f1_macro_score = m.f1_score(y_val_fold_arr, pred_labels, zero_division=np.nan, average='macro')
        scores.append(f1_macro_score)
        
        # Pruning optuna
        trial.report(f1_macro_score, step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        os.remove(temp_train_file)  # cleanup
    
    return np.mean(scores)  # average CV score

def tune_fasttext_cv(df_train, input_cols, output_cols, seed, thread=4, n_trials=20, n_splits=3):
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    
    # optuna pruning for time saving by halting trials that are unlikely to produce good results
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5, n_startup_trials=5)
    study.optimize(lambda trial: objective_cv(trial, df_train, input_cols, output_cols, n_splits, seed, thread), n_trials=n_trials)
    print("Best hyperparameters:", study.best_params)
    return study.best_params

def fasttext_train_fn(train_file, best_params, seed, model_file=None, thread=4):
    model = fasttext.train_supervised(
        input=train_file,
        lr=best_params["lr"],
        epoch=best_params["epoch"],
        wordNgrams=best_params["wordNgrams"],
        seed=seed, 
        thread=thread
    )
    if model_file!=None:
        model.save_model(model_file)
    return model


