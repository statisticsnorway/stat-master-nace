import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn.metrics as m
from sklearn.preprocessing import OneHotEncoder 


def brier_multi(targets, probs):    
    #making sure targets and probs are 1D arrays
    targets = np.array(targets).reshape(-1, 1)
    probs = np.array(probs).reshape(-1, 1)
    
    # 2D array for fitting
    all_labels = np.concatenate((targets, probs))
    all_labels = all_labels.reshape(-1, 1)
    
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(all_labels)
    
    ohe_targets = encoder.transform(targets)
    ohe_probs = encoder.transform(probs)
    return np.mean(np.sum(np.square(ohe_probs - ohe_targets), axis=1))



def metrics(target, pred):
    target, pred = np.array(target), np.array(pred)
    results = {}

    for avg in ['macro', 'micro', 'weighted']:
        results[avg] = {
            "f1": m.f1_score(target, pred, zero_division=np.nan, average=avg),
            "recall": m.recall_score(target, pred, zero_division=np.nan, average=avg),
            "precision": m.precision_score(target, pred, zero_division=np.nan, average=avg),
        }
    results['brier score'] = {"score":brier_multi(target, pred)}

    # Converts to DataFrame
    df_results = pd.DataFrame(results).T  # .T transposes so metrics are rows
    df_results = df_results.set_index(np.array(['macro', 'micro', 'weighted', 'score']))
    df_results.index.name = "average"
    return df_results


def metrics_levels(target:list[str], pred:list[str]):
    """
    target: Subclass targets.
    preds: Subclass predictions.
    
    Aggregates subclasses into higher hierarchies and evaluates the metrics on each hierarchy level.
    """
    target, pred = np.array(target), np.array(pred)

    cl_t = np.array([s[:-1] for s in target])
    cl_p = np.array([s[:-1] for s in pred])
    
    gro_t = np.array([s[:-2] for s in target])
    gro_p = np.array([s[:-2] for s in pred])
    
    div_t = np.array([s[:2] for s in target])
    div_p = np.array([s[:2] for s in pred])
    
    res_sub=metrics(target, pred)
    res_cl=metrics(cl_t, cl_p)
    res_gro=metrics(gro_t, gro_p)
    res_div=metrics(div_t, div_p)
    
    return res_sub, res_cl, res_gro, res_div
    

def df_to_table(df, title=""):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')
    ax.axis('tight')
    ax.table(cellText=df.values, 
             colLabels=df.columns, 
             rowLabels=df.index, 
             loc='center')
    plt.title(title)
    return fig