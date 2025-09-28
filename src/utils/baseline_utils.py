# imported libraries
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import SAVE_PATH



def fasttext_dataprep(df: pd.DataFrame, columns:list[str])-> pd.DataFrame:
    df = df.copy()
    df["fasttext_format"] = ("__label__" + df[columns].astype(str).agg(' '.join, axis=1))
    return df 

def splitting_dataset(df:pd.DataFrame, statify_column:str, train_file:str, test_file:str, val_file:str = False)->pd.DataFrame:
    os.makedirs(os.path.dirname(f"{SAVE_PATH}/{train_file}.txt"), exist_ok=True)
    os.makedirs(os.path.dirname(f"{SAVE_PATH}/{test_file}.txt"), exist_ok=True)
    if val_file:
        os.makedirs(os.path.dirname(f"{SAVE_PATH}/{val_file}.txt"), exist_ok=True)        
        
        
    # train vs test
    train, temp = train_test_split(df, test_size=0.4, random_state=42, stratify=df[statify_column])
    #### stratified cross validation instead of validation set
    
    if val_file:
        # test vs validation
        test, val = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp[statify_column])
        val["fasttext_format"].to_csv(f"{SAVE_PATH}/{val_file}.txt", index=False, header=False)
        
    else: 
        test=temp
        
    train["fasttext_format"].to_csv(f"{SAVE_PATH}/{train_file}.txt", index=False, header=False)
    test["fasttext_format"].to_csv(f"{SAVE_PATH}/{test_file}.txt", index=False, header=False)
    
    return (train, val, test) if val_file else (train, test)
    
def pred_prep(df:pd.DataFrame, input_cols:list[str], output_cols:list[str])->list[str]:
    labels = df[output_cols[0]].astype(str).tolist()
    input_txt = (df[input_cols].astype(str).agg(' '.join, axis=1)).tolist()
    df = df.copy()
    df[f"{input_cols[0]} and {input_cols[1]}"] = input_txt
    return input_txt, labels, df


def fasttext_input(df:pd.DataFrame, columns:list[str], statify_column:str, 
                   #input_cols_train:list[str], output_cols_train:list[str], 
                   #input_cols_val:list[str] = False, output_cols_val:list[str] = False, 
                   #input_cols_test:list[str], output_cols_test:list[str], 
                  train_file='train_fasttext', test_file='test_fasttext', val_file= False): #'val_fasttext'

    df_prep = fasttext_dataprep(df, columns)
    
    if val_file:
        train, val, test=splitting_dataset(
            df_prep, statify_column,train_file=f"{train_file}", 
            val_file=f"{val_file}", 
            test_file=f"{test_file}")
        #val_input_txt, val_labels=pred_prep(val, input_cols_val, output_cols_val)
        #train_input_txt, train_labels=pred_prep(train, input_cols_train, output_cols_train)
        #test_input_txt, test_labels=pred_prep(test, input_cols_test, output_cols_test)
        return train, val, test #train_input_txt, train_labels, val_input_txt, val_labels, test_input_txt, test_labels
        
    else:
        train, test=splitting_dataset(
            df_prep, statify_column, train_file=f"{train_file}", 
            test_file=f"{test_file}")

        #train_input_txt, train_labels=pred_prep(train, input_cols_train, output_cols_train)
        #test_input_txt, test_labels=pred_prep(test, input_cols_test, output_cols_test)
        return train, test #train_input_txt, train_labels, test_input_txt, test_labels
    


### Fasttext result preparation
def output_prep(labels:list[str]):
    labels = [l[0].replace('__label__', '') for l in labels]
    labels = np.array(labels)
    return labels

def wrong_preds_df(pred_labels:list[str], true_labels:list[str], input_text:list[str], map_file:str)->pd.DataFrame:
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
        'prediction name':[map_file.get(x) for x in wrong_pred], 
        'true codes':true_code, 
        'code name':[map_file.get(x) for x in true_code]})
    df_wrong_preds=df_wrong_preds.drop_duplicates()
    return df_wrong_preds

def hyper_params(model_file):

    model = fasttext.load_model(f"{SAVE_PATH}/model_fasttext/{model_file}.bin")
    args = model.f.getArgs()

    print("lr:", args.lr)
    print("dim:", args.dim)
    print("epoch:", args.epoch)
    print("wordNgrams:", args.wordNgrams)
    print("loss:", args.loss)


