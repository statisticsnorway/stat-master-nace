# imported libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocess import load_mapping
from src.config import SAVE_PATH



### Fasttext data preparation
# Format for FastText

def fasttext_dataprep(df: pd.DataFrame, columns:list[str])-> pd.DataFrame:
    df = df.copy()
    df["fasttext_format"] = ("__label__" + df[columns].astype(str).agg(' '.join, axis=1))
    return df 

def splitting_dataset(df:pd.DataFrame, statify_column:str)->pd.DataFrame:
    # train vs test
    train, temp = train_test_split(df, test_size=0.4, random_state=42, stratify=df[statify_column])
    #### stratified cross validation instead of validation set

    # test vs validation
    test, val = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp[statify_column])
    
    # Save to a text file
    train["fasttext_format"].to_csv(f"{SAVE_PATH}/train_fasttext.txt", index=False, header=False)
    val["fasttext_format"].to_csv(f"{SAVE_PATH}/val_fasttext.txt", index=False, header=False)
    test["fasttext_format"].to_csv(f"{SAVE_PATH}/test_fasttext.txt", index=False, header=False)
    return train, val, test
    
def pred_prep(df:pd.DataFrame, input_cols:list[str], output_cols:list[str])->list[str]:
    labels = df[output_cols[0]].astype(str).tolist()
    input_txt = (df[input_cols].astype(str).agg(' '.join, axis=1)).tolist()
    return input_txt, labels


def fasttext_input(df:pd.DataFrame, columns:list[str], statify_column:str, 
                   input_cols_val:list[str], output_cols_val:list[str], 
                   input_cols_test:list[str], output_cols_test:list[str]):

    df_prep = fasttext_dataprep(df, columns)
    train, val, test=splitting_dataset(df_prep, statify_column)
    val_input_txt, val_labels=pred_prep(val, input_cols_val, output_cols_val)
    test_input_txt, test_labels=pred_prep(test, input_cols_test, output_cols_test)
    return val_input_txt, val_labels, test_input_txt, test_labels
    


### Fasttext result preparation
def output_prep(pred_labels:list[str], true_labels:list[str]):
    # arrays of the true and predicted values
    # clean prediction labels
    pred_labels = [label[0].replace('__label__', '') for label in pred_labels]
    pred_labels, true_labels = np.array(pred_labels), np.array(true_labels)
    return pred_labels, true_labels

def wrong_preds_df(pred_labels:list[str], true_labels:list[str], input_text:list[str], mapping_file:str)->pd.DataFrame:
    """All the wrong predictions and the true labels are placed in a dataframe"""
    input_text = np.array(input_text)
    
    # filtering to only wrong classified values
    val_text_wp = input_text[pred_labels != true_labels]
    wrong_pred = pred_labels[pred_labels != true_labels]    
    true_code = true_labels[pred_labels != true_labels]

    # building mapping dictionaries
    #map_sn07 = dict(zip(df['SN2007'], df['SN2007 Tittel']))
    map_file = load_mapping(mapping_file)

    # new DataFrame
    df_wrong_preds = pd.DataFrame({
        'input text': val_text_wp,
        'wrong predictions':wrong_pred, 
        'prediction name':[map_file.get(x) for x in wrong_pred], 
        'true codes':true_code, 
        'code name':[map_file.get(x) for x in true_code]})
    df_wrong_preds=df_wrong_preds.drop_duplicates()
    return df_wrong_preds




if __name__=="__main__":
    val_input_txt, val_labels, test_input_txt, test_labels = fasttext_input(
        df=df_sn07, columns=["SN2007","tekst","navn"], statify_column="SN2007", 
        input_cols_val=["tekst","navn"], output_cols_val=["SN2007"], 
        input_cols_test=["tekst","navn"], output_cols_test=["SN2007", 'SN2007 Tittel'])