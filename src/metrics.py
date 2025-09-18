import sklearn.metrics as m
from sklearn.preprocessing import OneHotEncoder 


def brier_multi(targets, probs):
    ohe_targets = OneHotEncoder().fit_transform(targets.reshape(-1, 1))
    ohe_probs = OneHotEncoder().fit_transform(probs.reshape(-1, 1))
    return np.mean(np.sum(np.square(ohe_probs - ohe_targets), axis=1))



def metrics(target, preds):
    results = {}

    for avg in ['macro', 'micro', 'weighted']:
        results[avg] = {
            "f1": m.f1_score(target, pred, zero_division=np.nan, average=avg),
            "recall": m.recall_score(target, pred, zero_division=np.nan, average=avg),
            "precision": m.precsion_score(target, pred, zero_division=np.nan, average=avg),
        }
    results['brier score'] = m.brier_multi(target, pred)

    # Convert to DataFrame
    df_results = pd.DataFrame(results).T  # .T transposes so metrics are rows
    print(df_results)

