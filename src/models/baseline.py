# imported libraries
import fasttext
import optuna
import numpy as np
import pandas as pd
import os
import tempfile

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score
from src.config import DATA_PATH, OLD_DATA, TRANSITION_DATA_PATH, HIERARCHY_DATA, RANDOM_STATE, SAVE_PATH
from src.utils.baseline_utils import output_prep
import sklearn.metrics as m


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
    
    for train_idx, val_idx in skf.split(X, y):
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
        
        os.remove(temp_train_file)  # cleanup
    
    return np.mean(scores)  # average CV score

def tune_fasttext_cv(df_train, input_cols, output_cols, seed, thread=None, n_trials=20, n_splits=3):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective_cv(trial, df_train, input_cols, output_cols, n_splits, seed, thread), n_trials=n_trials)
    print("Best hyperparameters:", study.best_params)
    return study.best_params

def fasttext_train_fn(train_file, best_params, seed, model_file=None, thread=None):
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


# Hierarchical model
########### bytte til fasttext_train_fn for å velge beste parameter for hver modell? ############################
def train_hier_fasttext(df, input_col, label_hier, seed, thread=None):
    models={}
    temp_dir=tempfile.mkdtemp()
    
    # First level
    sec = label_hier[0]
    train_file = f"{SAVE_PATH}/data_fasttext/train_fasttext_{sec}.txt"
    models[sec] = fasttext.train_supervised(input=train_file, seed=seed, thread=thread)
    
    # deeper levels
    for i in range(1, len(label_hier)):
        parent_label = label_hier[i-1]
        current_label = label_hier[i]
        
        for parent, group_df in df.groupby(parent):
            group_df = group_df[[input_col, current_label]].copy()
            group_df[current_label] = "__label__" + group_df[current_label].astype(str)
            
            train_path = os.path.join(temp_dir, f"{current_label}_{parent}.txt")
            group_df[[current_label, input_col]].to_csv(train_path, index=False, sep=" ", header=False)
            
            models[current_label][parent]=fasttext.train_supervised(input=train_path, seed=seed, thread=thread)
            
    return models


def predict_hier_fasttext(models:dict, text:list[str]):
    # Level 1
    sec = models['section'].predict(text)[0].replace("__label__", "")

    # Level 2
    div = models['division'][sec].predict(text)[0].replace("__label__", "")

    # Level 3
    grp = models['group'][div].predict(text)[0].replace("__label__", "")

    # Level 4
    clas = models['class'][grp].predict(text)[0].replace("__label__", "")

    # Level 5
    sub = models['subclass'][clas].predict(text)[0].replace("__label__", "")

    return sub

 



    
