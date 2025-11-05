# imported libraries
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# Herfra kan du gruppere på over og underkategorier og få en oversikt over hvor mange virksomheter som befinner seg under hver kategori. Kan du herfra kjøre en enkel klassifiseringsmodell?

def count_per_category(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """ 
    Counts the number of occurances per category in a given column of a DataFrame.
    """
    return df.groupby(col).count().reset_index().fillna(0)


# Beskriv datasettet etter f.eks:
#   - hierarkisk oppbygning
#   - antall per kategori på flere nivå.

def explore_data_transition(df: pd.DataFrame):
    
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
    print(unmatched_07_25)
    print('new',new.sum()) #new 3
    print(new)
    print('edit',edit.shape[0]) #edit 780
    print(edit)
    print('del',deleted.sum()) #del 0
    print(deleted)
    print('new_desc',new_desc.shape[0]) #new_desc 455
    print(new_desc)



def freq_table(df:pd.DataFrame, mapping: dict, input_cols:list[str], hier:str, n=5) -> pd.DataFrame:
    """
    Tf-idf to understand the freqeunce of each word in the input and which 
    significantly influences prediction for each class.
    """    
    label=df[hier].dropna().unique()
    inpt=(df[input_cols].astype(str).agg(' '.join, axis=1)).tolist()   
    
    vectorizer = TfidfVectorizer(max_features=5000)
    vectorizer.fit(inpt)
    words = np.array(vectorizer.get_feature_names_out())
    
    tfidf_means=[]
    #classes = []
    
    for l in label:
        df_label = df[df[hier]== l]
        tfidf = vectorizer.transform((df_label[input_cols].astype(str).agg(' '.join, axis=1)).values.tolist())
        tfidf_mean = tfidf.mean(axis=0).A1 # Along the rows. Taking mean of frequencies throughout documents per class
        tfidf_means.append(tfidf_mean)
        #classes.append(l)

    tfidf_means_df = pd.DataFrame(tfidf_means, 
                                  index=[f"{mapping.get(x)}, ({x})" for x in label],
                                  columns=vectorizer.get_feature_names_out())
    
    max_tfidf = pd.DataFrame({cls: tfidf_means_df.loc[cls].nlargest(n).to_dict() for cls in tfidf_means_df.index}).T
    return tfidf_means_df, max_tfidf




    
    
    
    
        