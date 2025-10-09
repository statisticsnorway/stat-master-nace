# imported libraries
import numpy as np
import pandas as pd



# Herfra kan du gruppere på over og underkategorier og få en oversikt over hvor mange virksomheter som befinner seg under hver kategori. Kan du herfra kjøre en enkel klassifiseringsmodell?

def count_per_category(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """ 
    Counts the number of occurances per category in a given column of a DataFrame.
    """
    return df.groupby(col).count().reset_index().fillna(0)


# Beskriv datasettet etter f.eks:
#   - hierarkisk oppbygning
#   - antall per kategori på flere nivå.

def explore_data_transition(df):
    
    #1:N or N:1 relationships
    groups_07_25 = count_per_category(df[["SN2007", "SN2025"]],"SN2007") 
    groups_25_07 = count_per_category(df[["SN2007", "SN2025"]],"SN2025") 


    
    

    # Number of codes that have changed in SN2025
    unmatched_07_25 = df["SN2007"] != df["SN2025"]
    count_07_25_unmatched = unmatched_07_25.sum()

    # Number of new codes in 2025
    new = df["SN2007"].isna()

    # Number of changed codes in 2025
    edit = df[
        df["SN2007"].notna()
        & df["SN2025"].notna()
        & (df["SN2007"] != df["SN2025"])]
    
    # Number of deleted codes in 2025
    deleted = df["SN2025"].isna()

    # Number of codes that have same code number, but different description
    new_desc = df[
        (df["SN2007"] == df["SN2025"])
        & (df["SN2007 Tittel"] != df["SN2025 Tittel"])]
        
        
    print('1:N or N:1 relationships')
    print(groups_07_25)
    print(groups_25_07)
    print('Number of NACE codes that have changed in the 2025 sett')
    print(count_07_25_unmatched) #783 av 1241
    print('new',new.sum()) #new 3
    print('edit',edit.shape[0]) #edit 780
    print('del',deleted.sum()) #del 0
    print('new_desc',new_desc.shape[0]) #new_desc 455



def freq_table(df)
    # - sammenheng mellom enkeltord og kategori. tf-idf, bow
    