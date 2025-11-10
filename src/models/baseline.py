# imported libraries
import fasttext
import optuna
import numpy as np
import pandas as pd
import os
#import tempfile
import json
import shutil


from sklearn.model_selection import StratifiedKFold
from src.config import M_F_H
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


# Hierarchical model

def train_hier_fasttext(df, input_col, label_hiers, best_params, seed, thread=1, save_dir=f"{M_F_H}"):
    models_paths={}
    models_non_path={}
    temp_dir = os.path.expanduser("~/HPLT-project/stat-master-nace/data/temp_fastxt")
    
    # deeper levels
    for i in range(1, len(label_hiers)):
        parent_label = label_hiers[i-1]
        current_label = label_hiers[i]
        models_paths[current_label] = {}
        models_non_path[current_label] = {}
        
        for parent, group_df in df.groupby(parent_label):
    
            group_df = group_df[input_col + [current_label]].copy()
            #group_df[f"fasttext_format"] = ("__label__" + group_df[[current_label] + input_col].agg(' '.join, axis=1))

            # Create fasttext format
            group_df["fasttext_format"] = "__label__" + group_df[[current_label] + input_col].agg(' '.join, axis=1)
            # Remove extra spaces
            group_df["fasttext_format"] = group_df["fasttext_format"].str.replace(r'\s+', ' ', regex=True).str.strip()
        
            print(parent)
            print(current_label)
            print(group_df)
            
            train_path = os.path.join(temp_dir, f"{current_label}_{parent}.txt")
            print('train_path')
            print(train_path)

            
            group_df[f"fasttext_format"].to_csv(train_path, index=False, header=False, encoding="utf-8")

            model_path = os.path.join(save_dir, f"{current_label}_{parent}.bin")
            try:
                model = fasttext.train_supervised(input=train_path,
                                                lr=best_params["lr"],
                                                epoch=best_params["epoch"],
                                                wordNgrams=best_params["wordNgrams"],
                                                thread=thread)
            except:
                models_non_path[current_label][parent] = 'NaN'
                print('❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌')
                continue

            model.save_model(model_path)
            models_paths[current_label][parent] = model_path

            del model   

    with open(f'fasttext_hier_model_paths.json', 'w') as f:
        json.dump(models_paths, f, indent=4)

    with open(f'NaN_fasttext_hier_model_paths.json', 'w') as f:
        json.dump(models_non_path, f, indent=4)
    
    #shutil.rmtree(temp_dir)        
    return models_paths

def load_hier_fasttext_models(models_paths):
    models = {}
    
    # First level
    models['section'] = fasttext.load_model(models_paths['section'])

    # Deeper levels
    for level, value in models_paths.items():
        print(level)
        if isinstance(value, dict):
            models[level] = {}
            print(value)
            for parent, path in value.items():
                models[level][parent] = fasttext.load_model(path)

    return models


def predict_hier_fasttext(models:dict, text:list[str]):
    sec = list(np.char.replace(np.ravel(np.array(models['section'].predict(text)[0])), "__label__", ""))
    #print(sec)   

    div = list(np.char.replace(np.ravel(np.array([models['division'][s].predict(t)[0] for s, t in zip(sec, text)])), "__label__", ""))
    #print(div)

    grp = list(np.char.replace(np.ravel(np.array([models['group'][d].predict(t)[0] for d, t in zip(div, text)])), "__label__", ""))
    #print(grp)

    clas = list(np.char.replace(np.ravel(np.array([models['class'][g].predict(t)[0] for g, t in zip(grp, text)])), "__label__", ""))
    #print(clas)

    sub = [models['nace_21_code'][c].predict(t)[0] for c, t in zip(clas, text)]
    #print(sub)
    return sub




    
