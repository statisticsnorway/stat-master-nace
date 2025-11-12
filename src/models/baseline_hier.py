# Hierarchical model

import fasttext
import optuna
import numpy as np
import pandas as pd
import os
import json

from src.config import M_F_H, DATA, JSON_FILES
import sklearn.metrics as m


def train_hier_fasttext(df, input_col, label_hiers, best_params, seed, thread=1, save_dir=f"{M_F_H}"):
    models_paths={}
    models_non_path={}
    temp_dir = os.path.expanduser(f"{DATA}temp_fastxt")
    
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

    with open(f'{JSON_FILES}fasttext_hier_model_paths.json', 'w') as f:
        json.dump(models_paths, f, indent=4)

    with open(f'{JSON_FILES}NaN_fasttext_hier_model_paths.json', 'w') as f:
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




    
