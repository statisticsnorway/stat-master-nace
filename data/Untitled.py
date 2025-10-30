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
from datetime import date
from io import StringIO
import requests

# Treningsdata
df = pd.read_parquet("/ssb/stamme01/data811/NACE/data/one_to_many.parquet")
df_new = pd.read_parquet("/ssb/stamme01/data811/NACE/data/foretak_med_formaal_sn2025.parquet")

# Overgangssett
df_overgang = pd.read_csv(
    "/ssb/stamme01/data811/NACE/data/SN2025-SN2007_23.06.2025.csv", sep=";")


# NACE 2025 Hierarchi
today = date.today().isoformat()
HIERARCHY_DATA = f"https://data.ssb.no/api/klass/v1/classifications/6/codesAt.csv?date={today}&language=en"
df_hier = pd.read_csv(StringIO(requests.get(HIERARCHY_DATA).text), delimiter=',')


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
from io import StringIO
import requests
import csv
from datetime import date

#date = "2025-10-26"
today = date.today().isoformat()
print(today)
url = f"https://data.ssb.no/api/klass/v1/classifications/6/codesAt.csv?date={today}&language=en"

df = pd.read_csv(StringIO(requests.get(url).text), delimiter=',')
#print(df)
#print(df.columns)
df[df['code']=='M']

# %% [markdown]
# # Visualising the distribution

# %% [markdown] jp-MarkdownHeadingCollapsed=true
#
# ## distribution of the classes on all levels (not pruned)

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(f"/ssb/stamme01/data811/NACE/data_fasttext/data_preprocessed.csv")
n=5
##### scatter or bar plots? ###############
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
        if hier[i] == 'section':
            plt.bar(div_dist.index, div_dist.values, alpha=0.7, label="All values")
        else:
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

# %% [markdown]
# ## Counts of splits for codes on the hier dataset with the hier structure of the codes
# ### True hierarchy depth by pruning redundant depths vs not pruning

# %% [markdown]
# ### Original data with unpruned depth

# %%

levels = df_hier['level'].unique()
parents = df_hier['parentCode'].unique()
n=5

for i in range(len(levels)-1):
    df_hier_level = df_hier[df_hier['level']==levels[i]]
    #rint(df_hier_level)
    
    child = df_hier_level['code']
    parent = df_hier_level['parentCode']

    #cnt_grps = df.groupby(parent)[child].nunique()
    cnt_grps =df_hier_level.groupby(['parentCode'])['code'].nunique()
    min_idx = cnt_grps.nsmallest(n).index
    max_idx = cnt_grps.nlargest(n).index
    
    df_min = cnt_grps.nsmallest(n).reset_index()
    df_max = cnt_grps.nlargest(n).reset_index()
    
    df_min.columns = ['sn', 'nr of groups']
    df_max.columns = ['sn', 'nr of groups']
    
    print(df_min)
    print(df_max)
    
    cnt_grps=cnt_grps.sort_index()
    
    if levels[i]==2:
        plt.bar(cnt_grps.index, cnt_grps.values, alpha=0.7, label="All values")

    else:
        plt.bar(cnt_grps.index.astype(float), cnt_grps.values, alpha=0.7, label="All values")
    """
    if hier[i] == 'section':
            plt.bar(cnt_grps.index, cnt_grps.values, alpha=0.7, label="All values")
    else:
        plt.scatter(cnt_grps.index, cnt_grps.values, label="All values")
    """
    #plt.scatter(min_idx, cnt_grps.loc[min_idx], color='red', label="Lowest")
    #plt.scatter(max_idx, cnt_grps.loc[max_idx], color='green', label="Highest")
    #plt.xlabel(parent)
    plt.title(f"Count of splits in {levels[i]}")
    #plt.legend()
    plt.show()

# %% [markdown]
# ### Original data with unpruned depth

# %%
import re

def prune_tree(df, cols=["section", "division", "group", "class", "sn2025_1"]):
    df = df.copy()

    if cols==None:
        # --- Vectorized pruning for level 5 ---
        mask_lvl5 = (df["level"] == 5)
        mask_remove_lvl5 = df.loc[mask_lvl5, "code"].str.contains(r"\.\d*0+$", na=False)
        print(df.loc[mask_lvl5 & mask_remove_lvl5, "code"])
        df.loc[mask_lvl5 & mask_remove_lvl5, "code"] = None

        # --- Vectorized pruning for deeper levels ---
        levels = sorted(df["level"].unique(), reverse=True)[:-1]
        for l in levels[1:]:
            mask_next = (df["level"] == l + 1) & (df["code"].isna())
            parent_codes = df.loc[mask_next, "parentCode"].unique()
            if len(parent_codes) > 0:
                pattern = "|".join([re.escape(p) for p in parent_codes])
                mask_to_remove = (
                    (df["level"] == l)
                    & df["code"].notna()
                    & df["code"].str.contains(pattern)
                    & df["code"].str.contains(r"\.\d*0+$", na=False)
                )
                #print(df.loc[mask_to_remove, "code"])
                df.loc[mask_to_remove, "code"] = None                

    else:
        for i, col in enumerate(cols[:-1]):
            next_col = cols[i + 1]

            # Identifying which codes end with zeros
            mask_ends_with_zeros = df[col].astype(str).str.contains(r"\.\d*0+$", na=False)

            # Getting all codes that have at least one child
            valid_children_prefixes = (
                df[next_col]
                .dropna()
                .astype(str)
                .apply(lambda x: x.split(".")[0])  # generalize prefix
            )

            # Keeping rows that have no child with that prefix
            has_child = df[col].astype(str).isin(valid_children_prefixes)

            # Removeing only if ends with zeros AND has no child
            mask_to_remove = mask_ends_with_zeros & ~has_child
            df.loc[mask_to_remove, col] = None

        last_col = cols[-1]
        df[last_col] = df[last_col].mask(df[last_col].astype(str).str.contains(r"\.\d*0+$", na=False))

    return df





# %%
df_hier_pruned = prune_tree(df_hier, cols=None)
levels = df_hier_pruned['level'].unique()
levels
#df_hie[(df_hie['level']!=1) & (df_hie['level']!=2) & (df_hie['level']!=3) & (df_hie['level']!=4) & (df_hie['level']!=5)]

# %%

levels = df_hier_pruned['level'].unique()
parents = df_hier_pruned['parentCode'].unique()
n=5

for i in range(len(levels)-1):
    df_hier_pruned_level = df_hier_pruned[df_hier_pruned['level']==levels[i]]
    #rint(df_hier_level)
    
    child = df_hier_pruned_level['code']
    parent = df_hier_pruned_level['parentCode']

    #cnt_grps = df.groupby(parent)[child].nunique()
    cnt_grps =df_hier_pruned_level.groupby(['parentCode'])['code'].nunique()
    min_idx = cnt_grps.nsmallest(n).index
    max_idx = cnt_grps.nlargest(n).index
    
    df_min = cnt_grps.nsmallest(n).reset_index()
    df_max = cnt_grps.nlargest(n).reset_index()
    
    df_min.columns = ['sn', 'nr of groups']
    df_max.columns = ['sn', 'nr of groups']
    
    print(df_min)
    print(df_max)
    
    cnt_grps=cnt_grps.sort_index()
    
    if levels[i]==2:
        plt.bar(cnt_grps.index, cnt_grps.values, alpha=0.7, label="All values")

    else:
        plt.bar(cnt_grps.index.astype(float), cnt_grps.values, alpha=0.7, label="All values")
    """
    if hier[i] == 'section':
            plt.bar(cnt_grps.index, cnt_grps.values, alpha=0.7, label="All values")
    else:
        plt.scatter(cnt_grps.index, cnt_grps.values, label="All values")
    """
    #plt.scatter(min_idx, cnt_grps.loc[min_idx], color='red', label="Lowest")
    #plt.scatter(max_idx, cnt_grps.loc[max_idx], color='green', label="Highest")
    #plt.xlabel(parent)
    plt.title(f"Count of splits in {levels[i]-1}")
    #plt.legend()
    plt.show()

# %% [markdown]
# Most of the codes stops splitting at group level. So the tree doesnt get deeper than group level of hierarchy, while some very few still split on class level with '85.59' being the one with most splits, i.e. 7 splits. This means that only few codes have deep hierarchies if we consider any level beyound 'group' as deep, thus only those few to be hard to predict for a simple model like fasttext. the '85.59' would be even more challenging since it the many splits may be difficult to distinuish between in such a low level with possibly minor differences. I need to inspect titles of the '85.59' leafs as an example to understand how minor the difference between them are.

# %% [markdown]
# ## Distribution of classes on pruned hierarchy of company dataset

# %%
df = pd.read_csv(f"/ssb/stamme01/data811/NACE/data_fasttext/data_preprocessed.csv", dtype={'division':str, 'group':str, 'class':str, 'sn2025_1':str})
df.info()

# %%
df = prune_tree(df)
df.info()


# %%
# from preprocess.py file
def derive_hier(df: pd.DataFrame, subclass_col:str, section_map):
    """  ["section", "division", "group", "class"] """
    df = df.copy()
    df["division"] = df[subclass_col].str[:2]
    df["group"]    = df[subclass_col].str[:4]
    df["class"]    = df[subclass_col].str[:5]
    df["section"] = df["division"].map(section_map)
    return df

map_sec = dict(zip(df_hier["code"], df_hier["parentCode"]))
df_hier_hiers = derive_hier(df_hier[df_hier['level']==5], 'code', map_sec)
df_hier_hiers.head()

# %%
n=5


#cnt_grps = df.groupby(parent)[child].nunique()
cnt_grps = df_hier_hiers.groupby(['division'])['code'].nunique()
min_idx_c = cnt_grps.nsmallest(n).index
max_idx_c = cnt_grps.nlargest(n).index
print(cnt_grps)

df_min_c = cnt_grps.nsmallest(n).reset_index()
df_max_c = cnt_grps.nlargest(n).reset_index()

df_min_c.columns = ['sn', 'nr of groups']
df_max_c.columns = ['sn', 'nr of groups']

cnt_grps=cnt_grps.sort_index()


node = 'division'

div_dist = df[node].value_counts()
min_idx = div_dist.nsmallest(n).index
max_idx = div_dist.nlargest(n).index

df_min = div_dist.nsmallest(n).reset_index()
df_max = div_dist.nlargest(n).reset_index()

df_min.columns = ['sn', 'nr of instances']
df_max.columns = ['sn', 'nr of instances']


div_dist = div_dist.sort_index()
#print(div_dist)
highlight_min_vals = [div_dist.get(x, 0) for x in min_idx_c]
highlight_max_vals = [div_dist.get(x, 0) for x in max_idx_c]


plt.bar(div_dist.index.astype(float), div_dist.values, alpha=0.7, label="All values", zorder=1)
if node != "sn2025_1": 
    plt.bar(max_idx_c.astype(float), highlight_max_vals, color='red', label=cnt_grps[max_idx_c], zorder=2)            
    plt.bar(min_idx_c.astype(float), highlight_min_vals, color='green', label=cnt_grps[min_idx_c], zorder=2)
    plt.legend()


plt.xlabel(node)
plt.title(f"Distribution of {node} with nr of subclass splits")
plt.show()

# %%
levels = df_hier['level'].unique()
parents = df_hier['parentCode'].unique()
    
n=5
##### scatter or bar plots? ###############
hier = ["section", "division", "group", "class", "sn2025_1"]
for i,j in zip(range(len(hier)), range(len(levels))):
    
    df_hier_level = df_hier[df_hier['level']==levels[j]]
    
    child = df_hier_level['code']
    parent = df_hier_level['parentCode']

    #cnt_grps = df.groupby(parent)[child].nunique()
    cnt_grps =df_hier_level.groupby(['parentCode'])['code'].nunique()
    min_idx_c = cnt_grps.nsmallest(n).index
    max_idx_c = cnt_grps.nlargest(n).index
    
    df_min_c = cnt_grps.nsmallest(n).reset_index()
    df_max_c = cnt_grps.nlargest(n).reset_index()
    
    df_min_c.columns = ['sn', 'nr of groups']
    df_max_c.columns = ['sn', 'nr of groups']
    
    cnt_grps=cnt_grps.sort_index()
    
      
    node = hier[i]

    div_dist = df[node].value_counts()
    min_idx = div_dist.nsmallest(n).index
    max_idx = div_dist.nlargest(n).index
 
    df_min = div_dist.nsmallest(n).reset_index()
    df_max = div_dist.nlargest(n).reset_index()

    df_min.columns = ['sn', 'nr of instances']
    df_max.columns = ['sn', 'nr of instances']

    
    div_dist = div_dist.sort_index()
    highlight_min_vals = [div_dist.get(x, 0) for x in min_idx_c]
    highlight_max_vals = [div_dist.get(x, 0) for x in max_idx_c]
    if node == 'section':
        plt.bar(div_dist.index, div_dist.values, alpha=0.7, label="All values", zorder=1)
        plt.bar(max_idx_c, highlight_max_vals, color='red',  label=cnt_grps[max_idx_c], zorder=2)
        plt.bar(min_idx_c, highlight_min_vals, color='green', label=cnt_grps[min_idx_c], zorder=2)
        plt.legend()
        
    else:
        plt.bar(div_dist.index.astype(float), div_dist.values, alpha=0.7, label="All values", zorder=1)
        if node != "sn2025_1": 
            plt.bar(max_idx_c.astype(float), highlight_max_vals, color='red', label=cnt_grps[max_idx_c], zorder=2)            
            plt.bar(min_idx_c.astype(float), highlight_min_vals, color='green', label=cnt_grps[min_idx_c], zorder=2)
            plt.legend()
            
        else:
            continue

    plt.xlabel(node)
    plt.title(f"Distribution of {node}")
    plt.show()

# %%
levels = df_hier['level'].unique()
parents = df_hier['parentCode'].unique()
    
n=5
##### scatter or bar plots? ###############
hier = ["section", "division", "group", "class", "sn2025_1"]
for i,j in zip(range(len(hier)), range(len(levels))):
    
    df_hier_level = df_hier[df_hier['level']==levels[j]]
    
    child = df_hier_level['code']
    parent = df_hier_level['parentCode']

    #cnt_grps = df.groupby(parent)[child].nunique()
    cnt_grps =df_hier_level.groupby(['parentCode'])['code'].nunique()
    min_idx_c = cnt_grps.nsmallest(n).index
    max_idx_c = cnt_grps.nlargest(n).index
    
    df_min_c = cnt_grps.nsmallest(n).reset_index()
    df_max_c = cnt_grps.nlargest(n).reset_index()
    
    df_min_c.columns = ['sn', 'nr of groups']
    df_max_c.columns = ['sn', 'nr of groups']
    
    cnt_grps=cnt_grps.sort_index()
    
      
    node = hier[i]

    div_dist = df[node].value_counts()
    min_idx = div_dist.nsmallest(n).index
    max_idx = div_dist.nlargest(n).index
 
    df_min = div_dist.nsmallest(n).reset_index()
    df_max = div_dist.nlargest(n).reset_index()

    df_min.columns = ['sn', 'nr of instances']
    df_max.columns = ['sn', 'nr of instances']

    
    div_dist = div_dist.sort_index()
    highlight_min_vals = [div_dist.get(x, 0) for x in min_idx_c]
    highlight_max_vals = [div_dist.get(x, 0) for x in max_idx_c]
    if node == 'section':
        plt.bar(div_dist.index, div_dist.values, alpha=0.7, label="All values", zorder=1)
        plt.bar(max_idx_c, highlight_max_vals, color='red',  label=cnt_grps[max_idx_c], zorder=2)
        plt.bar(min_idx_c, highlight_min_vals, color='green', label=cnt_grps[min_idx_c], zorder=2)
        plt.legend()
        
    else:
        plt.bar(div_dist.index.astype(float), div_dist.values, alpha=0.7, label="All values", zorder=1)
        if node != "sn2025_1": 
            plt.bar(max_idx_c.astype(float), highlight_max_vals, color='red', label=cnt_grps[max_idx_c], zorder=2)            
            plt.bar(min_idx_c.astype(float), highlight_min_vals, color='green', label=cnt_grps[min_idx_c], zorder=2)
            plt.legend()
            
        else:
            continue

    plt.xlabel(node)
    plt.title(f"Distribution of {node}")
    plt.show()

# %%
levels = df_hier['level'].unique()
parents = df_hier['parentCode'].unique()
    
n=5
##### scatter or bar plots? ###############
hier = ["section", "division", "group", "class", "sn2025_1"]
for i,j in zip(range(len(hier)), range(len(levels))):
    
    df_hier_level = df_hier[df_hier['level']==levels[j]]
    
    child = df_hier_level['code']
    parent = df_hier_level['parentCode']

    #cnt_grps = df.groupby(parent)[child].nunique()
    cnt_grps =df_hier_level.groupby(['parentCode'])['code'].nunique()
    min_idx_c = cnt_grps.nsmallest(n).index
    max_idx_c = cnt_grps.nlargest(n).index
    
    df_min_c = cnt_grps.nsmallest(n).reset_index()
    df_max_c = cnt_grps.nlargest(n).reset_index()
    
    df_min_c.columns = ['sn', 'nr of groups']
    df_max_c.columns = ['sn', 'nr of groups']
    
    cnt_grps=cnt_grps.sort_index()
    
      
    node = hier[i]

    div_dist = df[node].value_counts()
    min_idx = div_dist.nsmallest(n).index
    max_idx = div_dist.nlargest(n).index
 
    df_min = div_dist.nsmallest(n).reset_index()
    df_max = div_dist.nlargest(n).reset_index()

    df_min.columns = ['sn', 'nr of instances']
    df_max.columns = ['sn', 'nr of instances']

    
    div_dist = div_dist.sort_index()
    highlight_min_vals = [div_dist.get(x, 0) for x in min_idx_c]
    highlight_max_vals = [div_dist.get(x, 0) for x in max_idx_c]
    if node == 'section':
        plt.bar(div_dist.index, div_dist.values, alpha=0.7, label="All values", zorder=1)
        plt.bar(max_idx_c, highlight_max_vals, color='red',  label=cnt_grps[max_idx_c], zorder=2)
        plt.bar(min_idx_c, highlight_min_vals, color='green', label=cnt_grps[min_idx_c], zorder=2)
        plt.legend()
        
    else:
        plt.bar(div_dist.index.astype(float), div_dist.values, alpha=0.7, label="All values", zorder=1)
        if node != "sn2025_1": 
            plt.bar(max_idx_c.astype(float), highlight_max_vals, color='red', label=cnt_grps[max_idx_c], zorder=2)            
            plt.bar(min_idx_c.astype(float), highlight_min_vals, color='green', label=cnt_grps[min_idx_c], zorder=2)
            plt.legend()
            
        else:
            continue

    plt.xlabel(node)
    plt.title(f"Distribution of {node}")
    plt.show()

# %%
levels = df_hier['level'].unique()
parents = df_hier['parentCode'].unique()
    
n=5
##### scatter or bar plots? ###############
hier = ["section", "division", "group", "class", "sn2025_1"]
for i,j in zip(range(len(hier)), range(len(levels))):
    
    df_hier_level = df_hier[df_hier['level']==levels[j]]
    
    child = df_hier_level['code']
    parent = df_hier_level['parentCode']

    #cnt_grps = df.groupby(parent)[child].nunique()
    cnt_grps =df_hier_level.groupby(['parentCode'])['code'].nunique()
    min_idx_c = cnt_grps.nsmallest(n).index
    max_idx_c = cnt_grps.nlargest(n).index
    
    df_min_c = cnt_grps.nsmallest(n).reset_index()
    df_max_c = cnt_grps.nlargest(n).reset_index()
    
    df_min_c.columns = ['sn', 'nr of groups']
    df_max_c.columns = ['sn', 'nr of groups']
    
    cnt_grps=cnt_grps.sort_index()
    
      
    node = hier[i]

    div_dist = df[node].value_counts()
    min_idx = div_dist.nsmallest(n).index
    max_idx = div_dist.nlargest(n).index
 
    df_min = div_dist.nsmallest(n).reset_index()
    df_max = div_dist.nlargest(n).reset_index()

    df_min.columns = ['sn', 'nr of instances']
    df_max.columns = ['sn', 'nr of instances']

    
    div_dist = div_dist.sort_index()
    highlight_min_vals = [div_dist.get(x, 0) for x in min_idx_c]
    highlight_max_vals = [div_dist.get(x, 0) for x in max_idx_c]
    if node == 'section':
        plt.bar(div_dist.index, div_dist.values, alpha=0.7, label="All values", zorder=1)
        plt.bar(max_idx_c, highlight_max_vals, color='red',  label=cnt_grps[max_idx_c], zorder=2)
        plt.bar(min_idx_c, highlight_min_vals, color='green', label=cnt_grps[min_idx_c], zorder=2)
        plt.legend()
        
    else:
        plt.bar(div_dist.index.astype(float), div_dist.values, alpha=0.7, label="All values", zorder=1)
        if node != "sn2025_1": 
            plt.bar(max_idx_c.astype(float), highlight_max_vals, color='red', label=cnt_grps[max_idx_c], zorder=2)            
            plt.bar(min_idx_c.astype(float), highlight_min_vals, color='green', label=cnt_grps[min_idx_c], zorder=2)
            plt.legend()
            
        else:
            continue

    plt.xlabel(node)
    plt.title(f"Distribution of {node}")
    plt.show()

# %%
levels = df_hier['level'].unique()
parents = df_hier['parentCode'].unique()
    
n=5
##### scatter or bar plots? ###############
hier = ["section", "division", "group", "class", "sn2025_1"]
for i,j in zip(range(len(hier)), range(len(levels))):
    
    df_hier_level = df_hier[df_hier['level']==levels[j]]
    
    child = df_hier_level['code']
    parent = df_hier_level['parentCode']

    #cnt_grps = df.groupby(parent)[child].nunique()
    cnt_grps =df_hier_level.groupby(['parentCode'])['code'].nunique()
    min_idx_c = cnt_grps.nsmallest(n).index
    max_idx_c = cnt_grps.nlargest(n).index
    
    df_min_c = cnt_grps.nsmallest(n).reset_index()
    df_max_c = cnt_grps.nlargest(n).reset_index()
    
    df_min_c.columns = ['sn', 'nr of groups']
    df_max_c.columns = ['sn', 'nr of groups']
    
    cnt_grps=cnt_grps.sort_index()
    
      
    node = hier[i]

    div_dist = df[node].value_counts()
    min_idx = div_dist.nsmallest(n).index
    max_idx = div_dist.nlargest(n).index
 
    df_min = div_dist.nsmallest(n).reset_index()
    df_max = div_dist.nlargest(n).reset_index()

    df_min.columns = ['sn', 'nr of instances']
    df_max.columns = ['sn', 'nr of instances']

    
    div_dist = div_dist.sort_index()
    highlight_min_vals = [div_dist.get(x, 0) for x in min_idx_c]
    highlight_max_vals = [div_dist.get(x, 0) for x in max_idx_c]
    if node == 'section':
        plt.bar(div_dist.index, div_dist.values, alpha=0.7, label="All values", zorder=1)
        plt.bar(max_idx_c, highlight_max_vals, color='red',  label=cnt_grps[max_idx_c], zorder=2)
        plt.bar(min_idx_c, highlight_min_vals, color='green', label=cnt_grps[min_idx_c], zorder=2)
        plt.legend()
        
    else:
        plt.bar(div_dist.index.astype(float), div_dist.values, alpha=0.7, label="All values", zorder=1)
        if node != "sn2025_1": 
            plt.bar(max_idx_c.astype(float), highlight_max_vals, color='red', label=cnt_grps[max_idx_c], zorder=2)            
            plt.bar(min_idx_c.astype(float), highlight_min_vals, color='green', label=cnt_grps[min_idx_c], zorder=2)
            plt.legend()
            
        else:
            continue

    plt.xlabel(node)
    plt.title(f"Distribution of {node}")
    plt.show()

# %% [markdown]
# ## Counts of splits for codes on the company dataset

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
    
    plt.bar(cnt_grps.index, cnt_grps.values, alpha=0.7, label="All values")
    """
    if hier[i] == 'section':
            plt.bar(cnt_grps.index, cnt_grps.values, alpha=0.7, label="All values")
    else:
        plt.scatter(cnt_grps.index, cnt_grps.values, label="All values")
    """
    plt.scatter(min_idx, cnt_grps.loc[min_idx], color='red', label="Lowest")
    plt.scatter(max_idx, cnt_grps.loc[max_idx], color='green', label="Highest")
    plt.xlabel(parent)
    plt.title(f"Count of splits in {parent}")
    plt.legend()
    plt.show()

# %% [markdown]
# ## distribution agains count of groups

# %%
for i in range(1,len(hier)):
    parent = hier[i]
    if parent == 'subclass':
        pass
    child = hier[i+1]

    cnt_grps = df.groupby(parent)[child].nunique()
    min_idx_cnt = cnt_grps.nsmallest(n).index
    max_idx_cnt = cnt_grps.nlargest(n).index
    
    #cnt_min = cnt_grps.nsmallest(n).reset_index()
    cnt_max = cnt_grps.nlargest(n).reset_index()
    
    node = hier[i]

    div_dist = df[node].value_counts()
    min_idx_div = div_dist.nsmallest(n).index
    max_idx_div = div_dist.nlargest(n).index

    df_min = div_dist.nsmallest(n).reset_index()
    df_max = div_dist.nlargest(n).reset_index()
    print(max_idx_cnt)
    
    print(cnt_max)
    
    plt.scatter(div_dist.values, cnt_grps.values, alpha=0.7)
    #plt.scatter(div_dist.loc[min_idx_div], cnt_grps.loc[min_idx_div], color='red', label="Lowest distribution")
    #plt.scatter(div_dist.loc[max_idx_cnt], cnt_grps.loc[max_idx_cnt], color='orange', label="Highest count")
    plt.ylabel('cnt_grps')
    plt.xlabel('div_dist')
    plt.title(f"Data abundance vs hierarchy complexity of {parent}")
    
    
    """
    # lowest div_dist
    for idx in min_idx_div:
        if idx in div_dist.index and idx in cnt_grps.index:
            plt.text(div_dist.loc[idx], cnt_grps.loc[idx],
                     f"{idx} ↓", color='red', fontsize=9, fontweight='bold')
    
    # highest cnt_grps
    for idx in max_idx_cnt:
        if idx in div_dist.index and idx in cnt_grps.index:
            plt.text(div_dist.loc[idx], cnt_grps.loc[idx],
                     f"{idx} ↑", color='green', fontsize=9, fontweight='bold')
    """
    plt.tight_layout()
    plt.show()

# %%
from sklearn.feature_extraction.text import TfidfVectorizer


    
def freq_table(df:pd.DataFrame, input_cols:list[str], hier:str) -> pd.DataFrame:
    """
    Tf-idf to understand the freqeunce of each word in the input and which 
    significantly influences prediction for each class.
    """
    label=df[hier]
    inpt=(df[input_cols].astype(str).agg(' '.join, axis=1)).tolist()   
    
    vectorizer = TfidfVectorizer(max_features=5000)
    vectorizer.fit(inpt)
    words = np.array(vectorizer.get_feature_names_out())
    
    tfidf_means=[]
    classes = []
    
    for l in label.unique():
        df_label = df[df[hier]== l]
        tfidf = vectorizer.transform((df_label[input_cols].astype(str).agg(' '.join, axis=1)).tolist())
        tfidf_mean = tfidf.mean(axis=0).A1 # Along the row
        tfidf_means.append(tfidf_mean)
        classes.append(l)
    
    tfidf_means_df = pd.DataFrame(tfidf_means, index=classes, columns=vectorizer.get_feature_names_out())
    return tfidf_means_df




# %%

for hier in hierarchies:
    tfidf_means_df = freq_table(df=df, input_cols=['tekst','navn'], hier=hier)
    print(hier)
    print(tfidf_means_df)

# %%
