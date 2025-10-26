# imported libraries
import fasttext
import optuna
import numpy as np
import pandas as pd
import os
import tempfile
import json

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score
from src.config import DATA_PATH, OLD_DATA, TRANSITION_DATA_PATH, HIERARCHY_DATA, RANDOM_STATE, SAVE_PATH
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
def train_hier_fasttext(df, input_col, label_hiers, best_params, seed, thread=4, save_dir=f"{SAVE_PATH}/models_fasttext_hier/"):
    models_paths={}
    temp_dir=tempfile.mkdtemp()
    
    # First level
    sec = label_hiers[0]
    train_file = f"{SAVE_PATH}/data_fasttext/train_fasttext_{sec}.txt"
    model_path = os.path.join(save_dir, f"{sec}.bin")
    model = fasttext.train_supervised(input=train_file, 
                                            lr=best_params["lr"],
                                            epoch=best_params["epoch"],
                                            wordNgrams=best_params["wordNgrams"],
                                            seed=seed, 
                                            thread=thread,
                                            )
    
    model.save_model(model_path)
    models_paths[sec]=model_path
    del model #freeing RAM
    
    # deeper levels
    for i in range(1, len(label_hiers)):
        parent_label = label_hiers[i-1]
        current_label = label_hiers[i]
        models_paths[current_label] = {}
        
        for parent, group_df in df.groupby(parent_label):
            group_df = group_df[input_col + [current_label]].copy()
            group_df[f"fasttext_format"] = ("__label__" + group_df[[current_label] + input_col].astype(str).agg(' '.join, axis=1))
  

            print(parent)
            print(current_label)
            print(group_df)
            
            train_path = os.path.join(temp_dir, f"{current_label}_{parent}.txt")
            group_df[f"fasttext_format"].to_csv(train_path, index=False, header=False)
            
            model_path = os.path.join(save_dir, f"{current_label}_{parent}.bin")

            model = fasttext.train_supervised(input=train_path,
                                              lr=best_params["lr"],
                                              epoch=best_params["epoch"],
                                              wordNgrams=best_params["wordNgrams"],
                                              seed=seed,
                                              thread=thread)
            model.save_model(model_path)
            models_paths[current_label][parent] = model_path
            del model   
            
    with open(f'fasttext_hier_model_paths.json', 'w') as f:
        json.dump(models_paths, f, indent=4)
    return models_paths

def load_hier_fasttext_models(models_paths):
    models = {}

    # First level
    first_level = list(models_paths.keys())[0]
    models[first_level] = fasttext.load_model(models_paths[first_level])

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
    sec = models['section'].predict(text)[0][0].replace("__label__", "")

    div = models['division'][sec].predict(text)[0][0].replace("__label__", "")

    grp = models['group'][div].predict(text)[0][0].replace("__label__", "")

    clas = models['class'][grp].predict(text)[0][0].replace("__label__", "")

    sub = models['subclass'][clas].predict(text)[0][0].replace("__label__", "")

    return sub


def predict_hier_fasttext_lazy(models_paths, text):
    # ---- Predict section ----
    section_model = fasttext.load_model(models_paths['section'])
    print(section_model)
    print(section_model.predict(text)[0][0])
    sec = section_model.predict(text)[0][0].replace("__label__", "")
    del section_model 

    
    #  division 
    div_path = models_paths['division'].get(sec)
    if not div_path or not os.path.exists(div_path):
        return {"section": sec}

    division_model = fasttext.load_model(div_path)
    div = division_model.predict(text)[0][0].replace("__label__", "")
    del division_model

    # group 
    grp_path = models_paths['group'].get(div)
    if not grp_path or not os.path.exists(grp_path):
        return {"section": sec, "division": div}

    group_model = fasttext.load_model(grp_path)
    grp = group_model.predict(text)[0][0].replace("__label__", "")
    del group_model

    # class 
    cls_path = models_paths['class'].get(grp)
    if not cls_path or not os.path.exists(cls_path):
        return {"section": sec, "division": div, "group": grp}

    cls_model = fasttext.load_model(cls_path)
    clas = cls_model.predict(text)[0][0].replace("__label__", "")
    del cls_model

    # subclass
    sub_path = models_paths['sn2025_1'].get(clas)
    if not sub_path or not os.path.exists(sub_path):
        return {"section": sec, "division": div, "group": grp, "class": clas}

    sub_model = fasttext.load_model(sub_path)
    sub = sub_model.predict(text)[0][0].replace("__label__", "")
    print(sub_model)
    del sub_model
    
    return sub



    
