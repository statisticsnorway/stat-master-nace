# Hierarchical model

import fasttext
import optuna
import numpy as np
import pandas as pd
import os
import json
from collections import defaultdict

from src.config import M_F_H, DATA_FASTXT, JSON_FILES, DATA_FX_TR_VAL_TE
import sklearn.metrics as m


def train_hier_fasttext(df, input_col, label_hiers, best_params, seed, thread=1, save_dir=M_F_H):
    models_paths={}
    models_non_path={}
    temp_dir = os.path.expanduser(os.path.join(DATA_FASTXT,"temp_fastxt")) # where the aggregated datasets are saved
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    # deeper levels
    for i in range(1, len(label_hiers)):
        parent_label = label_hiers[i-1]
        current_label = label_hiers[i]
        models_paths[current_label] = {}
        models_non_path[current_label] = {}
        
        if parent_label == 'root':
            model_path = os.path.join(save_dir, f"{current_label}.bin")
            train_file = os.path.join(DATA_FX_TR_VAL_TE, f'train_{current_label}.txt')
            model = fasttext.train_supervised(
                        input=train_file,
                        lr=best_params["lr"],
                        epoch=best_params["epoch"],
                        wordNgrams=best_params["wordNgrams"],
                        seed=seed, 
                        thread=thread
                    )
            
            model.save_model(model_path)
            models_paths[current_label] = {parent_label: model_path}
            del model
            continue

        for parent, group_df in df.groupby(parent_label):
            model_path = os.path.join(save_dir, f"{current_label}_{parent}.bin")
            train_path = os.path.join(temp_dir, f"{current_label}_{parent}.txt")

            if not os.path.exists(train_path):
                group_df = group_df[input_col + [current_label]].copy()
                #group_df[f"fasttext_format"] = ("__label__" + group_df[[current_label] + input_col].agg(' '.join, axis=1))

                # Create fasttext format
                group_df["fasttext_format"] = "__label__" + group_df[[current_label] + input_col].agg(' '.join, axis=1)
                # Remove extra spaces
                group_df["fasttext_format"] = group_df["fasttext_format"].str.replace(r'\s+', ' ', regex=True).str.strip()
            
                print(parent)
                print(current_label)
                print(group_df)
                
                print('train_path')
                print(train_path)
                group_df[f"fasttext_format"].to_csv(train_path, index=False, header=False, encoding="utf-8")

            try:
                model = fasttext.train_supervised(input=train_path,
                                                lr=best_params["lr"],
                                                epoch=best_params["epoch"],
                                                wordNgrams=best_params["wordNgrams"],
                                                thread=thread,
                                                seed=seed)
            except:
                models_non_path[current_label][parent] = 'NaN'
                print('❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌')
                continue

            model.save_model(model_path)
            models_paths[current_label][parent] = model_path

    with open(os.path.join(JSON_FILES,'fasttext_hier_model_paths.json'), 'w') as f:
        json.dump(models_paths, f, indent=4)

    with open(os.path.join(JSON_FILES,'NaN_fasttext_hier_model_paths.json'), 'w') as f:
        json.dump(models_non_path, f, indent=4)

    try:
        del model
    except NameError:
        pass
    #shutil.rmtree(temp_dir)        
    return models_paths


def load_hier_fasttext_models(models_paths):
    models = {}

    # Load root section model
    models['section'] = {'root': fasttext.load_model(models_paths['section']['root'])}

    # Load only deeper levels
    for level, value in models_paths.items():
        if level == 'section':  # skip root, already loaded
            continue
        models[level] = {}
        for parent, path in value.items():
            if path != 'NaN':  # skip failed models
                models[level][parent] = path  # just store the path for now
                # do NOT load yet to save RAM

    return models


"""
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
"""

def batch_level(text, parent_labels, model_dict, out_labels, out_probs):

    groups = defaultdict(list)

    # group indices by parent
    for i, parent in enumerate(parent_labels):
        groups[parent].append(i)

    # predict once per parent model
    for parent, idxs in groups.items():
        subset_text = [text[i] for i in idxs]

        path = model_dict[parent]  # path string
        model = fasttext.load_model(path)  # load model on demand
        labs, probs = model.predict(subset_text)
        del model  # free RAM

        labs = np.char.replace(np.ravel(labs), "__label__", "")
        probs = np.ravel(probs)

        out_labels[idxs] = labs
        out_probs[idxs] = probs

    return out_labels, out_probs

def predict_hier_fasttext(models, text, eps=1e-12):
    n = len(text)
    sec = np.empty(n, dtype=object)
    div = np.empty(n, dtype=object)
    grp = np.empty(n, dtype=object)
    clas = np.empty(n, dtype=object)
    sub = np.empty(n, dtype=object)

    p_sec = np.zeros(n)
    p_div = np.zeros(n)
    p_grp = np.zeros(n)
    p_clas = np.zeros(n)
    p_sub = np.zeros(n)

    # Root level
    root_model = models['section']['root']
    labels, probs = root_model.predict(text)
    del root_model  # free RAM if you want
    sec[:] = np.char.replace(np.ravel(labels), "__label__", "")
    p_sec[:] = np.ravel(probs)

    # Other levels — paths only
    div, p_div = batch_level(text, sec, models['division'], div, p_div)
    print('div', div)
    grp, p_grp = batch_level(text, div, models['group'], grp, p_grp)
    print('grp', grp)
    clas, p_clas = batch_level(text, grp, models['class'], clas, p_clas)
    print('clas', clas)
    sub, p_sub = batch_level(text, clas, models['nace_21_code'], sub, p_sub)

    log_path_prob = (
        np.log(p_sec + eps)
        + np.log(p_div + eps)
        + np.log(p_grp + eps)
        + np.log(p_clas + eps)
        + np.log(p_sub + eps)
    )
    path_prob = np.exp(log_path_prob)

    return sub, path_prob



    
