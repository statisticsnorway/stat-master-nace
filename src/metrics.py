import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn.metrics as m
from sklearn.preprocessing import OneHotEncoder 
from src.config import HIERARCHY_DATA

df_hier=pd.read_csv(HIERARCHY_DATA, dtype={'code':str}, sep=";", encoding="latin1")
map_sec = dict(zip(df_hier[df_hier['level']==2]["code"], df_hier[df_hier['level']==2]["parentCode"]))

def get_ancestors(node, map_sec):
    # Returns a set of the node and all its ancestors
    ancestors = set()

    ancestors.add(node)

    parts = node.split('.')
    if len(parts) == 2:
        cl = parts[0] + '.' + parts[1][:-1]
        grp = parts[0] + '.' + parts[1][:-2]
        div = parts[0]
        sec = map_sec[parts[0]]
    else:
        ValueError('Not a valid code')
    return {node, cl, grp, div, sec}

def hierarchical_f1(y_true, y_pred, map_sec=map_sec):
    # y_true, y_pred are lists of labels
    # hierarchy is a dict: {child: parent}
    
    true_set = set()
    pred_set = set()
    
    for t,p in zip(y_true,y_pred):
        true_set.update(get_ancestors(t,map_sec))
        pred_set.update(get_ancestors(p,map_sec))
        
    tp = len(true_set.intersection(pred_set))
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1


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
    #print('target \n', target, 'pred \n', pred)
    for avg in ['macro', 'weighted']:
        results[avg] = {
            "f1": m.f1_score(target, pred, zero_division=np.nan, average=avg),
            #"recall": m.recall_score(target, pred, zero_division=np.nan, average=avg),
            #"precision": m.precision_score(target, pred, zero_division=np.nan, average=avg),
        }
    results['score'] = {'brier_score':brier_multi(target, pred)}

    if len(target[0]) == 6:
        results['score']['HF1'] = hierarchical_f1(target,pred)
    
    # Converts to DataFrame
    df_results = pd.DataFrame(results).T  # .T transposes so metrics are rows
    #df_results = df_results.set_index(np.array(['macro', 'micro', 'weighted', 'score']))
    df_results.index.name = "average"
    return df_results


def metrics_levels(target:list[str], pred:list[str], map_sec=map_sec):
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

    sec_t = np.array([map_sec[t] for t in div_t])
    sec_p = np.array([map_sec[p] for p in div_p])
    
    res_sub=metrics(target, pred)
    res_cl=metrics(cl_t, cl_p)
    res_gro=metrics(gro_t, gro_p)
    res_div=metrics(div_t, div_p)
    res_sec=metrics(sec_t,sec_p)
    
    return res_sub, res_cl, res_gro, res_div, res_sec

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

def mean_ci(values):
    mean = np.mean(values)
    low, high = np.percentile(values, [2.5, 97.5])
    return mean, low, high, f"{mean:.4f} ± {(high-low)/2:.4f}"



