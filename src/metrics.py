import numpy as np
import pandas as pd
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
    results['brier score'] = brier_multi(target, pred)

    # Convert to DataFrame
    df_results = pd.DataFrame(results).T  # .T transposes so metrics are rows
    return df_results

