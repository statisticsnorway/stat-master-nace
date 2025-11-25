import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import RES_FASTXT


def wrong_preds_df(pred_labels:list[str], true_labels:list[str], input_text:list[str], mapping:dict)->pd.DataFrame:
    """All the wrong predictions and the true labels are placed in a dataframe"""
    input_text = np.array(input_text)
    
    # filtering to only wrong classified values
    input_text_wp = input_text[pred_labels != true_labels]
    wrong_pred = pred_labels[pred_labels != true_labels]    
    true_code = true_labels[pred_labels != true_labels]

    # new DataFrame
    df_wrong_preds = pd.DataFrame({
        'input text': input_text_wp,
        'wrong predictions':wrong_pred, 
        'prediction name':[mapping.get(x) for x in wrong_pred], 
        'true codes':true_code, 
        'code name':[mapping.get(x) for x in true_code]})
    df_wrong_preds=df_wrong_preds.drop_duplicates()
    return df_wrong_preds


def error_rate_classes_hist(df_wrong_preds:pd.DataFrame, index_type:str='int', n:int=0):
    cnt_grps = df_wrong_preds['true codes'].value_counts()
    if index_type == 'str':
        if n==0:
            plt.barh(cnt_grps.index.astype(str), cnt_grps.values)

        else:
            min_idx = cnt_grps.nsmallest(n).index
            max_idx = cnt_grps.nlargest(n).index
            plt.bar(cnt_grps.index[min_idx], cnt_grps.loc[min_idx])
            plt.bar(cnt_grps.index[max_idx], cnt_grps.loc[max_idx])
            

    elif index_type == 'int':
        if n!=0:
            plt.hist(cnt_grps.index.astype(str), cnt_grps.values)

        else:
            min_idx = cnt_grps.nsmallest(n).index
            max_idx = cnt_grps.nlargest(n).index
            plt.hist(cnt_grps.index[min_idx], cnt_grps.loc[min_idx])
            plt.hist(cnt_grps.index[max_idx], cnt_grps.loc[max_idx])

    plt.title(f"Error counts per class")
    plt.savefig("true_codes_counts.png")
    return max_idx

def len_error_rate(df_wrong_preds:pd.DataFrame, index_type:str='int', threshold:int=0):
    """ comparing error rate for short vs. long input texts.
    """ 
    
    cnt_grps = df_wrong_preds.groupby(['true codes'])
    df_wrong_preds[len(df_wrong_preds['input text']) < threshold]



    if index_type == 'str':
        if threshold==0:
            plt.barh(cnt_grps.index.astype(str), cnt_grps.values)
        else:
            min_idx = cnt_grps.nsmallest(n).index
            max_idx = cnt_grps.nlargest(n).index
            plt.bar(cnt_grps.index[min_idx], cnt_grps.loc[min_idx])
            plt.bar(cnt_grps.index[max_idx], cnt_grps.loc[max_idx])
            

    elif index_type == 'int':
        if threshold==0:
            plt.hist(cnt_grps.index.astype(str), cnt_grps.values)

        else:
            min_idx = cnt_grps.nsmallest(n).index
            max_idx = cnt_grps.nlargest(n).index
            plt.hist(cnt_grps.index[min_idx], cnt_grps.loc[min_idx])
            plt.hist(cnt_grps.index[max_idx], cnt_grps.loc[max_idx])

    plt.title(f"Input length of errors per gold label")
    plt.savefig("true_codes_counts.png")
    return max_idx


######## use f1 macro and micro score for this ############
def global_local_error():
    """ whether mistakes are global (wrong top-level) or local (within same branch).
    """
    ...

def vocab_richness_error():
    """ how much info a text has
    """
    ...

if __name__=="__main__":
    df_wrong_preds=pd.read_csv(f"{RES_FASTXT}cv_text_navn/section_df_wrong_res_test.csv",  dtype=str)
  
    max_indx = error_rate_classes_hist(df_wrong_preds=df_wrong_preds, index_type='str')