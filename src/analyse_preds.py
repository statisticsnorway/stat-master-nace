import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import RES_HIER_M, DATA_FX_TR_VAL_TE, RES_HIER_M


def wrong_preds_df(pred_labels:list[str], true_labels:list[str], input_text:list[str], mapping:dict)->pd.DataFrame:
    """All the wrong predictions and the true labels are placed in a dataframe"""
    input_text = np.array(input_text)
    
    # filtering to only wrong classified values
    input_text_wp = input_text[pred_labels != true_labels]
    wrong_pred = pred_labels[pred_labels != true_labels]    
    gold_labels = true_labels[pred_labels != true_labels]

    # new DataFrame
    df_wrong_preds = pd.DataFrame({
        'input text': input_text_wp,
        'wrong predictions':wrong_pred, 
        'prediction name':[mapping.get(x) for x in wrong_pred], 
        'gold labels':gold_labels, 
        'label name':[mapping.get(x) for x in gold_labels]})
    df_wrong_preds_dist=df_wrong_preds.drop_duplicates()
    return df_wrong_preds, df_wrong_preds_dist


def all_preds_df(pred_labels:list[str], true_labels:list[str], input_text:list[str], mapping:dict)->pd.DataFrame:
    """All the wrong predictions and the true labels are placed in a dataframe"""
    input_text = np.array(input_text)
    
    # filtering to only wrong classified values
    input_text_wp = input_text[pred_labels != true_labels]
    wrong_pred = pred_labels[pred_labels != true_labels]    
    gold_labels = true_labels[pred_labels != true_labels]

    # new dataframes
    df_preds = pd.DataFrame({
        'input text': input_text_wp,
        'predictions':pred_labels, 
        'prediction name':[mapping.get(x) for x in pred_labels], 
        'gold labels':gold_labels, 
        'label name':[mapping.get(x) for x in gold_labels]})

    return df_preds, df_preds.drop_duplicates()


def tot_cnt_grps(df, label):
    return df[label].value_counts()


def tot_len_error_grps(df, label):
    return df[label].value_counts()


def error_rate_classes_hist(df_wrong_preds:pd.DataFrame, tot_cnt_grps:pd.Series, index_type:str='int', n:int=0):
    print('tot_cnt_grps')
    #print(tot_cnt_grps.loc['01.280'])
    cnt_grps = df_wrong_preds['true codes'].value_counts()/tot_cnt_grps
    #print('cnt_grps/tot',  df_wrong_preds['true codes'].value_counts().loc['01.280'])
    max_cnt_grps = cnt_grps.nlargest(5)
    min_cnt_grps = cnt_grps.nsmallest(5)

    if index_type == 'str':
        if n==0:
            plt.barh(cnt_grps.index.astype(str), cnt_grps.values)

        else:
            max_cnt_grps = cnt_grps.nlargest(n).index
            min_idx = cnt_grps.nsmallest(n).index
            plt.bar(cnt_grps.index[min_idx], cnt_grps.loc[min_idx])
            plt.bar(cnt_grps.index[max_idx], cnt_grps.loc[max_idx])
            

    elif index_type == 'int':
        if n==0:
            plt.hist(cnt_grps.index.astype(str), cnt_grps.values)
        else:
            max_cnt_grps = cnt_grps.nlargest(n).index
            min_idx = cnt_grps.nsmallest(n).index
            plt.hist(cnt_grps.index[min_idx], cnt_grps.loc[min_idx])
            plt.hist(cnt_grps.index[max_idx], cnt_grps.loc[max_idx])

    plt.title(f"Error counts per class")
    plt.savefig("true_codes_counts.png")
    return max_cnt_grps, min_cnt_grps


# we estimate the sequence length as the number of tokens after tokenizing the dataset
def len_error_rate(df_wrong_preds:pd.DataFrame, tot_cnt_grps:pd.Series, index_type:str='int', plot:bool=False, n:int=0):
    """ comparing error rate for short vs. long input texts.
    """ 
    
    #cnt_grps = df_wrong_preds.groupby(['true codes'])
    #df_wrong_preds[len(df_wrong_preds['input text']) < threshold]

    # tokens = words
    df_wrong_preds["num_tokens"] = df_wrong_preds["input text"].str.split().str.len()
    print('df_wrong_preds', df_wrong_preds)
    df_wrong_preds = df_wrong_preds/tot_cnt_grps

    # Aggregate min, mean, max per class
    length_stats = df_wrong_preds.groupby('true codes')['num_tokens'].agg(
        min_tokens='min',
        avg_tokens='mean',
        max_tokens='max'
    ).reset_index()

    if plot==False:
        return length_stats

    #TODO: finish implementing the plot 
    elif plot==True:

        if index_type == 'str':
            if n==0:
                plt.barh(length_stats.index.astype(str), cnt_grps.values)

            else:
                min_idx = length_stats.nsmallest(n).index
                max_idx = length_stats.nlargest(n).index
                plt.bar(length_stats.index[min_idx], length_stats.loc[min_idx])
                plt.bar(length_stats.index[max_idx], length_stats.loc[max_idx])
                        
        elif index_type == 'int':
            if n==0:
                plt.hist(length_stats.index.astype(str), length_stats.values)

            else:
                min_idx = length_stats.nsmallest(n).index
                max_idx = length_stats.nlargest(n).index
                plt.hist(length_stats.index[min_idx], length_stats.loc[min_idx])
                plt.hist(length_stats.index[max_idx], length_stats.loc[max_idx])
                

        plt.title(f"Input length of errors per gold label")
        plt.savefig("input_length.png")
    return length_stats


def stat_length_input_per_class():
    """ getting the distribution for the length of the input text per class in one hierarchy level.¨
    """
    ...

def semantically_similiar_labels():
    """ Labels with 
    """
    ...


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
    label = 'nace_21_code'
    df_test=pd.read_csv(f"{DATA_FX_TR_VAL_TE}test.csv",  dtype=str)
    df_wrong_preds=pd.read_csv(f"{RES_HIER_M}hier_df_wrong_res_test.csv",  dtype=str)
    
    test_tot_cnt = tot_cnt_grps(df_test, label)

    max_vals, min_vals = error_rate_classes_hist(df_wrong_preds=df_wrong_preds, tot_cnt_grps=test_tot_cnt, index_type='str')
    print('max error rates')
    print(max_vals)
    
    length_stats = len_error_rate(df_wrong_preds=df_wrong_preds, tot_cnt_grps=test_tot_cnt, index_type='str')
    print('min length statistics')
    print(length_stats.nsmallest(5, 'avg_tokens'))

    print('max length statistics')
    print(length_stats.nlargest(5, 'avg_tokens'))

    print('length statistics of labels with the most errors')
    print(length_stats.loc[length_stats['true codes'].isin(max_vals.index)])

    print('min error rates')
    print(min_vals)

    print('length statistics of labels with the least errors')
    print(length_stats.loc[length_stats['true codes'].isin(min_vals.index)])

    # TODO: Prove that the vagueness of the input texts are the problem
    # TODO: 