# imported libraries
import os
import numpy as np
import pandas as pd
import fasttext

from sklearn.model_selection import train_test_split
from src.config import MODELS_FASTXT



def fasttext_dataprep(df: pd.DataFrame, columns:list[str])-> pd.DataFrame:
    df = df.copy()
    df[columns] = df[columns].astype(str).replace(r'\b(nan|none|na)\b', '', regex=True)
    # Create fasttext format
    df["fasttext_format"] = "__label__" + df[columns].agg(' '.join, axis=1)
    # Remove extra spaces
    df["fasttext_format"] = df["fasttext_format"].str.replace(r'\s+', ' ', regex=True).str.strip()
    return df 

def splitting_dataset(df:pd.DataFrame, statify_column:str, train_file:str, test_file:str, seed:int, val_file:str = False)->pd.DataFrame:
        
    # train vs test
    train, temp = train_test_split(df, test_size=0.4, random_state=seed, stratify=df[statify_column])
    #### stratified cross validation instead of validation set
    
    if val_file:
        # test vs validation
        test, val = train_test_split(temp, test_size=0.5, random_state=seed, stratify=temp[statify_column])
        val["fasttext_format"].to_csv(f"{val_file}.txt", index=False, header=False)
        
    else: 
        test=temp
        
    train["fasttext_format"].to_csv(f"{train_file}.txt", index=False, header=False)
    test["fasttext_format"].to_csv(f"{test_file}.txt", index=False, header=False)
    
    return (train, val, test) if val_file else (train, test)
    
def pred_prep(df:pd.DataFrame, input_cols:list[str], output_cols:list[str])->list[str]:
    labels = df[output_cols[0]].astype(str).tolist()
    if len(input_cols) >1:
        input_txt = (df[input_cols].astype(str).agg(' '.join, axis=1)).tolist()
        df = df.copy()
        df["tekst og navn"] = input_txt
    else: 
        input_txt = df[input_cols[0]].astype(str).tolist()
    return input_txt, labels, df


def fasttext_input(df:pd.DataFrame, columns:list[str], statify_column:str, seed:int,
                  train_file='train_fasttext', test_file='test_fasttext', val_file= False): #'val_fasttext'

    df_prep = fasttext_dataprep(df, columns)
    
    if val_file:
        train, val, test=splitting_dataset(
            df_prep, statify_column, seed=seed,
            train_file=f"{train_file}",
            val_file=f"{val_file}", 
            test_file=f"{test_file}")
        return train, val, test 
        
    else:
        train, test=splitting_dataset(
            df_prep, statify_column, seed=seed,
            train_file=f"{train_file}", 
            test_file=f"{test_file}")
        return train, test
    


### Fasttext result preparation
def output_prep(labels:list[str]):
    labels = np.char.replace(np.ravel(np.array(labels)), "__label__", "")

    #labels = [l[0].replace('__label__', '') for l in labels]
    #labels = np.array(labels)
    return labels

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

def hyper_params(model_file):

    model = fasttext.load_model(f"{MODELS_FASTXT}{model_file}.bin")
    args = model.f.getArgs()

    print("lr:", args.lr)
    print("dim:", args.dim)
    print("epoch:", args.epoch)
    print("wordNgrams:", args.wordNgrams)
    print("loss:", args.loss)


