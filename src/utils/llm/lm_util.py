import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from src.config import MODELS_FASTXT, DATA_LM_TR_TE, DATA_LM_TR_VAL_TE, RANDOM_STATE, DATA

seed = RANDOM_STATE


def splitting_dataset(df:pd.DataFrame, statify_column:str, train_file:str, test_file:str, seed:int, val_file:str = False)->pd.DataFrame:
        
    # train vs test
    train, temp = train_test_split(df, test_size=0.4, random_state=seed, stratify=df[statify_column])
    #### stratified cross validation instead of validation set
    
    if val_file==False:
        test=temp
        train.to_csv(f"{DATA_LM_TR_TE}train.csv")
        test.to_csv(f"{DATA_LM_TR_TE}test.csv")
    
    else: 
        # test vs validation
        test, val = train_test_split(temp, test_size=0.5, random_state=seed, stratify=temp[statify_column])
        val.to_csv(f"{DATA_LM_TR_VAL_TE}val.csv")
        train.to_csv(f"{DATA_LM_TR_VAL_TE}train.csv")
        test.to_csv(f"{DATA_LM_TR_VAL_TE}test.csv")
    
    return (train, val, test) if val_file else (train, test)
    



if __name__=='__main__':
    df = pd.read_csv(f"{DATA}data_prep_lm.csv", dtype={'company_activity':str,'company_name':str,'division':str, 'group':str, 'class':str, 'nace_21_code':str,'nace_21_description_nb':str}, keep_default_na=False, na_values=[]).fillna("")
    # Getting the train, val and test sets for the LM model
    splitting_dataset(df=df, statify_column='nace_21_code', train_file='train', test_file='test', seed=seed, val_file=False)
    splitting_dataset(df=df, statify_column='nace_21_code', train_file='train', test_file='test', seed=seed, val_file=True)
