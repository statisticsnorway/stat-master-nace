# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: stat-master-nace
#     language: python
#     name: stat-master-nace
# ---

# %% [markdown]
# # Data exploring

# %%
# imported libraries
import fasttext
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.model_selection import train_test_split
import sklearn.metrics as m

# Treningsdata
df = pd.read_parquet("/ssb/stamme01/data811/NACE/data/one_to_many.parquet")
df_new = pd.read_parquet("/ssb/stamme01/data811/NACE/data/foretak_med_formaal_sn2025.parquet")

# Overgangssett
df_overgang = pd.read_csv(
    "/ssb/stamme01/data811/NACE/data/SN2025-SN2007_23.06.2025.csv", sep=";")


# NACE 2007 Hierarchi
df_hier = pd.read_csv("/ssb/stamme01/data811/NACE/data/nace25_hierarki.csv", sep=";", encoding="latin-1")


# Hent ut data for gamle nace-koder, org-nr og fritekst.
# Antar gamle nace-koder er SN07
df_sn07 = df[["orgnr", "tekst", "navn", "SN2007"]]


# Herfra kan du gruppere på over og underkategorier og få en oversikt over hvor mange virksomheter som befinner seg under hver kategori. Kan du herfra kjøre en enkel klassifiseringsmodell?

# Beskriv datasettet etter f.eks:
#   - hierarkisk oppbygning
groups = df_sn07.groupby("SN2007")["navn"]


#   - antall per kategori på flere nivå.
counted_groups = groups.count()


#   - sammenheng mellom enkeltord og kategori.


# utforske koblingen mellom SN07 og SN25. Hvilke har 1:N koblinger og hvilke har ikke noen koblinger.
groups_07_25 = df[["SN2007", "SN2025"]].groupby("SN2007")
groups_07_25.count()  # Shape is (106, 1)

groups_25_07 = df[["SN2007", "SN2025"]].groupby("SN2025")
groups_25_07.count()  # Shape is (45, 1)
print('done')

# %%
from faker.providers.person.no_NO import Provider
import re
import string

first_names_menn_df = pd.read_excel("/home/stud-msh/stat-master-nace/data/manneNavn.xlsx")
first_names_kvinner_df = pd.read_excel("/home/stud-msh/stat-master-nace/data/kvinneNavn.xlsx")
first_names_df = pd.concat([first_names_menn_df, first_names_kvinner_df], ignore_index=True)

last_names_df = pd.read_excel("/home/stud-msh/stat-master-nace/data/etternavn.xlsx")

# Converting to sets
first_names = set(first_names_df.iloc[:, 0].dropna().str.strip())
last_names = set(last_names_df.iloc[:, 0].dropna().str.strip())

# Combining into one set
all_names = first_names.union(last_names)

# Compiling regex 
pattern = r"\b(" + "|".join(map(re.escape, all_names)) + r")\b"
name_regex = re.compile(pattern, flags=re.IGNORECASE)

all_names_lower = {n.lower() for n in all_names}

def remove_names(text):
    # Remove punctuation (replace with spaces)
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    # Remove names
    words = [w for w in text.split() if w.lower() not in all_names_lower]
    return " ".join(words)

def column_subset(df):
    """Choosing a subset of columns """
    #return df[["orgnr", "tekst", "navn", "SN2007"]]
    return df[["tekst", "navn", "sn2025_1"]]

def cleaning_df(df:pd.DataFrame) -> pd.DataFrame:
    df.copy()
        
    if set(["tekst", "navn", "sn2025_1"]).issubset(set(df.columns)):
        df = df[df.groupby("sn2025_1")["navn"].transform("count") >10]
        df = df[df['sn2025_1']!='00.000']
    
    # Filtering out names
    df['navn'] = df['navn'].apply(remove_names)
    
    return df


# %%
cleaning_df(df_new)['navn']

# %%
df_overgang.head() #[df_overgang['SN2025']=='00.000']

# %%
# ----------- df_overgang datasettet --------------

# 1) Hvordan kategoriene i 2007 er bygd opp, hvordan kan man beskrive strukturen på disse (visualisering)?

df_overgang[df_overgang["SN2007"] == "* Har ingen korrespondanse i SN2007"]
df_overgang = df_overgang.replace(
    to_replace="* Har ingen korrespondanse i SN2007", value=np.nan
)
df_overgang = df_overgang.astype({"SN2025": str})



# %%
# 2) Hvor mange av NACE-kodene som har endret seg i 2025 settet.
unmatched_07_25 = df_overgang["SN2007"] != df_overgang["SN2025"]
count_07_25_unmatched = unmatched_07_25.sum()

# Antall nye koder i 2025:
new = df_overgang["SN2007"].isna()

# Antall endrede koder i 2025:
edit = df_overgang[
    df_overgang["SN2007"].notna()
    & df_overgang["SN2025"].notna()
    & (df_overgang["SN2007"] != df_overgang["SN2025"])]


# Antall koder som ikke finnes lenger i 2025
deleted = df_overgang["SN2025"].isna()

# Antall koder som har samme kode nummer, men annerledes beskrivelse
new_desc = df_overgang[
    (df_overgang["SN2007"] == df_overgang["SN2025"])
    & (df_overgang["SN2007 Tittel"] != df_overgang["SN2025 Tittel"])]

# %%
df_overgang.head()

# %%
print(df_sn07)
print("number of companies under each NACE category")
print(counted_groups)

# %%
print("Number of NACE codes that have changed in the 2025 sett")
print(count_07_25_unmatched)  # 783 av 1241
print("new", new.sum())  # new 3
print("edit", edit.shape[0])  # edit 780
print("del", deleted.sum())  # del 0
print("new_desc", new_desc.shape[0])  # new_desc 455

# %% [markdown]
# ## Fasttext hyperparameter tuning
# FastText's autotune feature allows you to find automatically the best hyperparameters for your dataset. Hyperparameters are learning rate and epochs.
# - er fasttext bedre på å predikere tall eller text? 
# - prøve å bytte ut kodene med navn istedet. kanskje bedre med engelsk navn?
# - prøve ut fasttext hierarki prediksjoner
#
#

# %% [markdown]
# # Visualising the distribution

# %%
# Hvilke hovedgrupper er størst? 62

# %% [markdown]
#
# Checking the distribution of the classes on Subclass category 10.201

# %% [markdown]
#
# Checking the distribution of the classes on Division category 10

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(f"/ssb/stamme01/data811/NACE/data_fasttext/data_preprocessed.csv")
n=5

hier = ["section", "division", "group", "class", "sn2025_1"]
for i in range(len(hier)):
        node = hier[i]
        
        div_dist = df[node].value_counts()
        min_idx = div_dist.nsmallest(n).index
        max_idx = div_dist.nlargest(n).index
    
        df_min = div_dist.nsmallest(n).reset_index()
        df_max = div_dist.nlargest(n).reset_index()
    
        df_min.columns = ['sn', 'nr of instances']
        df_max.columns = ['sn', 'nr of instances']
        
        print(df_min)
        print(df_max)
    
        div_dist = div_dist.sort_index()
        plt.scatter(div_dist.index, div_dist.values, label="All values")
        plt.scatter(min_idx, div_dist.loc[min_idx], color='red', label="Lowest")
        plt.scatter(max_idx, div_dist.loc[max_idx], color='green', label="Highest")
        plt.legend()
        plt.xlabel(node)
        plt.title(f"Distribution of {node}")
        plt.show()


# %% [markdown]
# Hvilke grupper er det som blir mest oppsplittet? 46, men denne gruppen har ikke størst mengde datapunkter. Gruppe 62 har størst mengde datapunkter, men kun 1 oppsplitting. 85 og 26 har 3 og 2 oppsplittinger i nevnt rekkefølge, men er de gruppene med færrest datapunkter. 
#
# Finn måter å visualisere hierarkisk informasjon på en oversiktlig måte. 
# Hvilke områder er det som ser vanskelige ut når det kommer til å gi prediksjoner? De som har færrest datapunkter men mer enn 1 oppsplittinger.

# %%
for i in range(len(hier)-1):
    parent = hier[i]
    child = hier[i+1]

    cnt_grps = df.groupby(parent)[child].nunique()
    min_idx = cnt_grps.nsmallest(n).index
    max_idx = cnt_grps.nlargest(n).index
    
    df_min = cnt_grps.nsmallest(n).reset_index()
    df_max = cnt_grps.nlargest(n).reset_index()
    
    df_min.columns = ['sn', 'nr of groups']
    df_max.columns = ['sn', 'nr of groups']
    
    print(df_min)
    print(df_max)
    
    cnt_grps=cnt_grps.sort_index()
    
    plt.scatter(cnt_grps.index, cnt_grps.values, label="All values")
    plt.scatter(min_idx, cnt_grps.loc[min_idx], color='red', label="Lowest")
    plt.scatter(max_idx, cnt_grps.loc[max_idx], color='green', label="Highest")
    plt.xlabel(parent)
    plt.title(f"Count of splits in {parent}")
    plt.legend()
    plt.show()
