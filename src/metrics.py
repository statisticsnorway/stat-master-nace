from sklearn.metrics import f1_score

labels, probs = model.predict(val_text) 
# clean prediction labels
pred_labels = [label[0].replace('__label__', '') for label in labels]

results = {}

for metric in ['macro', 'micro', 'weighted']:
    results[metric] = {
        "f1": m.f1_score(val_labels, pred_labels, zero_division=np.nan, average=metric),
        "recall": m.recall_score(val_labels, pred_labels, zero_division=np.nan, average=metric),
        "precision": m.precsion_score(val_labels, pred_labels, zero_division=np.nan, average=metric),
        #brier_report = m.brier_score_loss(val_labels, pred_labels)
        
    }

# Convert to DataFrame
df_results = pd.DataFrame(results).T  # .T transposes so metrics are rows
print(df_results)

#print(brier_report)