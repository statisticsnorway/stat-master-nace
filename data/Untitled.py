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

# Overgangssett
df_overgang = pd.read_csv(
    "/ssb/stamme01/data811/NACE/data/SN2025-SN2007_23.06.2025.csv", sep=";"
)


# NACE 2007 Hierarchi
#df_hier = pd.read_csv("/ssb/stamme01/data811/NACE/data//nace07_hierarki.csv", sep=";", encoding="latin-1")


# Hent ut data for gamle nace-koder, org-nr og fritekst.
# Antar gamle nace-koder er SN07
df_sn07 = df[["orgnr", "tekst", "navn", "SN2007"]]


# Herfra kan du gruppere på over og underkategorier og få en oversikt over hvor mange virksomheter som befinner seg under hver kategori. Kan du herfra kjøre en enkel klassifiseringsmodell?

# Beskriv datasettet etter f.eks:
#   - hierarkisk oppbygning
groups = df_sn07.groupby("SN2007")


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
df = pd.read_parquet("/ssb/stamme01/data811/NACE/data/one_to_many.parquet")
df.head()

# %%
df_hier.head()

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
    & (df_overgang["SN2007"] != df_overgang["SN2025"])
]


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
# # Fasttext

# %%
# Fasttext data preparation

# Format for FastText
df_sn07["fasttext_format"] = (
    "__label__" + df_sn07["SN2007"] + " " + df_sn07["tekst"] + " " + df_sn07["navn"]
)

# train vs test
train, temp = train_test_split(df_sn07, test_size=0.4, random_state=42)#, stratify=df_sn07['SN2007'])
#### stratified cross validation instead of validation set

# test vs validation
test, val = train_test_split(temp, test_size=0.5, random_state=42)#, stratify=temp['SN2007'])

# Save to a text file
data = '/ssb/stamme01/data811/NACE/'

train["fasttext_format"].to_csv(f"{data}/train_fasttext.txt", index=False, header=False)
val["fasttext_format"].to_csv(f"{data}/val_fasttext.txt", index=False, header=False)
test["fasttext_format"].to_csv(f"{data}/test_fasttext.txt", index=False, header=False)

# Input for the prediction method:
val_labels = val["SN2007"].tolist()
val_text = (val["tekst"] + " " + val["navn"]).tolist()

test_labels = test["SN2007"].tolist()
test_text = (test["tekst"] + " " + test["navn"]).tolist()



# %%
test.index = pd.RangeIndex(start=0, stop=len(test))
test.iloc[1]

# %% [markdown]
# ## Fasttext hyperparameter tuning
# FastText's autotune feature allows you to find automatically the best hyperparameters for your dataset. Hyperparameters are learning rate and epochs.
# - er fasttext bedre på å predikere tall eller text? 
# - prøve å bytte ut kodene med navn istedet. kanskje bedre med engelsk navn?
# - prøve ut fasttext hierarki prediksjoner
#
#

# %% [markdown]
# ### Training and predicting on codes

# %%
# Kjøre dataene inn på fasttext modellen (og andre relevante modeller) og se hva slags resultater vi får

# Skipgram model, finetuned:
model = fasttext.train_supervised(input=f"{data}/train_fasttext.txt", autotuneValidationFile=f"{data}/val_fasttext.txt") # Hyperparameter tuning by using "autotuneValidationFile" parameter

#Saving the model
model.save_model(f"{data}/model_nace.bin")

# using saved model
model = fasttext.load_model(f"{data}/model_nace.bin")




# %% [markdown]
# #### Metrics
# Kjøre med scikit learn i tillegg **weighted-** og **macro F1 score** for å fokusere mer på laveste nivåene.

# %%
from sklearn.metrics import f1_score

labels, probs = model.predict(val_text) 
# clean prediction labels
pred_labels = [label[0].replace('__label__', '') for label in labels]

results = {}

for metric in ['macro', 'micro', 'weighted']:
    results[metric] = {
        "f1": m.f1_score(val_labels, pred_labels, zero_division=np.nan, average=metric),
        "recall": m.recall_score(val_labels, pred_labels, zero_division=np.nan, average=metric),
        "precision": m.precsion_score(val_labels, pred_labels, zero_division=np.nan, average=metric),
        #brier_report = m.brier_score_loss(val_labels, pred_labels)
        
    }

# Convert to DataFrame
df_results = pd.DataFrame(results).T  # .T transposes so metrics are rows
print(df_results)

#print(brier_report)

# %% [markdown]
# #### Finding out which labels are classified wrong

# %%
################# Gjøre dette om til en funksjon s################

# arrays of the truee and predicted values
pred_labels=np.array(pred_labels)
val_labels=np.array(val_labels)

# filtering to only wrong classified values
val_text_wp = val_text[pred_labels != val_labels]
wrong_pred = pred_labels[pred_labels != val_labels]    
true_code = val_labels[pred_labels != val_labels]

# building mapping dictionaries
map_sn07 = dict(zip(df_overgang['SN2007'], df_overgang['SN2007 Tittel']))

# new DataFrame
df_wrong_preds = pd.DataFrame({
    'input text': val_text_wp,
    'wrong predictions':wrong_pred, 
    'prediction name':[map_sn07.get(x) for x in wrong_pred], 
    'true codes':true_code, 
    'code name':[map_sn07.get(x) for x in true_code]})
df_wrong_preds=df_wrong_preds.drop_duplicates()
df_wrong_preds

# %% [markdown]
# ### Training and predicting on code names

# %%

# %% [markdown]
# ## Hierarchical fasttext

# %%
model = 

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # Visualising the distribution

# %%
# Hvilke hovedgrupper er størst? 62

# %% [markdown]
#
# Checking the distribution of the classes on Subclass category 10.201

# %%
train["Subclass"] = train["SN2007"].astype(float)
train["Subclass"]

# %% [markdown]
#
# Checking the distribution of the classes on Class category 10.20

# %%
train["Class"] = round(train["Subclass"], 2).astype(str)
train["Class"]

# %% [markdown]
#
# Checking the distribution of the classes on Group category 10.2

# %%
train["Group"] = round(train["Subclass"], 1).astype(str)
train["Group"]

# %%
counts_gr = train["Group"].value_counts()
plt.bar(counts_gr.index, counts_gr.values)
plt.xlabel("Category")
plt.ylabel("Count")
plt.title("Count of Values in Category Column")
plt.xticks(rotation=90)  # Rotates x-axis labels by 45 degrees
plt.show()

# %% [markdown]
#
# Checking the distribution of the classes on Division category 10

# %%
train["Division"] = train["Subclass"].astype(int).astype(str)
train["Division"]

# %%
import seaborn as sns
from matplotlib.patches import Patch

tr_div_gr = train.groupby("Division")["Group"].nunique().sort_values(ascending=False)
div_dist = train["Division"].value_counts().reindex(tr_div_gr.index)

# generating colors(one per bar)
colors = cm.plasma(tr_div_gr.values / tr_div_gr.values.max())


plt.figure(figsize=(12, 6))
plt.bar(div_dist.index, div_dist.values, color=colors)
plt.yscale("log")

plt.xlabel("Division")
plt.ylabel("Count")
plt.title("Distribution of the Division and Number of Hierarchical Splits")
plt.xticks(rotation=90)

unique_counts = sorted(tr_div_gr.unique())
count_to_color = {
    count: cm.plasma(count / tr_div_gr.values.max()) for count in unique_counts
}
legend_handles = [
    Patch(color=count_to_color[count], label=f"{count} groups")
    for count in unique_counts
]

plt.legend(
    handles=legend_handles,
    title="Number of Groups",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
)

plt.tight_layout()


# %%
# This plot is the best one in my opinion
#!!!!!!!!!!!! Not good practice to have 2 y axis!!!!!!!!!!!!!!!!!!!

# Data
tr_div_gr = train.groupby("Division")["Group"].nunique().sort_values(ascending=False)
div_dist = train["Division"].value_counts().reindex(tr_div_gr.index)

fig, ax1 = plt.subplots(figsize=(12, 6))

# Bar plot for counts
ax1.bar(tr_div_gr.index, div_dist.values)
ax1.set_xlabel("Division")
ax1.set_ylabel("Number of Groups", color="black")
ax1.tick_params(axis="y", labelcolor="black")
plt.xticks(rotation=90)
plt.title("Count of Groups per Division")

# secondary y-axis to indicate number of groups
ax2 = ax1.twinx()
ax2.plot(tr_div_gr.index, tr_div_gr.values, "o-", color="red", label="Number of Groups")
ax2.set_ylabel("Distinict Number of Groups", color="red")
ax2.tick_params(axis="y", labelcolor="red")

plt.tight_layout()
plt.show()


# %%
# Data
tr_div_gr = train.groupby("Group")["Class"].nunique().sort_values(ascending=False)
div_dist = train["Group"].value_counts().reindex(tr_div_gr.index)

fig, ax1 = plt.subplots(figsize=(12, 6))

# Bar plot for counts
ax1.bar(tr_div_gr.index, div_dist.values)


# Normalising the plots?
# plt.yscale('log')

ax1.set_xlabel("Group")
ax1.set_ylabel("Number of Classes", color="black")
ax1.tick_params(axis="y", labelcolor="black")
plt.xticks(rotation=90)
plt.title("Count of Classes per Groups")

# secondary y-axis to indicate number of groups
ax2 = ax1.twinx()
ax2.plot(tr_div_gr.index, tr_div_gr.values, "-o", color="red", label="Number of Groups")
ax2.set_ylabel("Distinict Number of Classes", color="red")
ax2.tick_params(axis="y", labelcolor="red")
# Only integers on y-axis (red)
from matplotlib.ticker import MaxNLocator

ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

plt.tight_layout()
plt.show()

# %%
# This plot is the best one in my opinion
#!!!!!!!!!!!! Not good practice to have 2 y axis!!!!!!!!!!!!!!!!!!!


# Data
tr_div_gr = train.groupby("Class")["Subclass"].nunique().sort_values(ascending=False)
div_dist = train["Class"].value_counts().reindex(tr_div_gr.index)

fig, ax1 = plt.subplots(figsize=(12, 6))

# Bar plot for counts
ax1.bar(tr_div_gr.index, div_dist.values)


# Normalising the plots?
# plt.yscale('log')

ax1.set_xlabel("Classes")
ax1.set_ylabel("Number of Subclasses", color="black")
ax1.tick_params(axis="y", labelcolor="black")
plt.xticks(rotation=90)
plt.title("Count of Subclasses per Classes")

# secondary y-axis to indicate number of groups
ax2 = ax1.twinx()
ax2.plot(tr_div_gr.index, tr_div_gr.values, "-o", color="red", label="Number of Groups")
ax2.set_ylabel("Distinict Number of Subclasses", color="red")
ax2.tick_params(axis="y", labelcolor="red")
# Only integers on y-axis (red)
from matplotlib.ticker import MaxNLocator

ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

plt.tight_layout()
plt.show()

# %%
train.columns

# %% [markdown]
# Hvilke grupper er det som blir mest oppsplittet? 46, men denne gruppen har ikke størst mengde datapunkter. Gruppe 62 har størst mengde datapunkter, men kun 1 oppsplitting. 85 og 26 har 3 og 2 oppsplittinger i nevnt rekkefølge, men er de gruppene med færrest datapunkter. 
#
# Finn måter å visualisere hierarkisk informasjon på en oversiktlig måte. 
# Hvilke områder er det som ser vanskelige ut når det kommer til å gi prediksjoner? De som har færrest datapunkter men mer enn 1 oppsplittinger.

# %%
df_hier_sorted = df_hier.sort_values(by="level", ascending=True, inplace=False)
df_hier_sorted

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # Visualizing the hierarchy of the NACE07 codes

# %%
# Er det noen fellestrekk i friteksten i enkelte grupper som kan beskrives?

df_hier_sorted = df_hier.sort_values(by="level", ascending=True, inplace=False)

from anytree import Node, RenderTree
from anytree.exporter import DotExporter

# Creating the tree through a dictionary of nodes
nodes = {}
roots = []

dummy_root = Node("AllRoots")
for _, row in df_hier_sorted.iterrows():
    if row["parentCode"] is np.nan:
        nodes[row["code"]] = Node(row["code"], parent=dummy_root, alias=row["name"])

        roots.append(nodes[row["code"]])
    else:
        nodes[row["code"]] = Node(
            row["code"], parent=nodes[row["parentCode"]], alias=row["name"]
        )

roots = sorted(roots, key=lambda r: r.alias)

# Printing hierarchy
"""
for root in roots:
    for pre, fill, node in RenderTree(root):
        print(f"{pre}{node.alias}")
"""

###### Need to install the graphviz software in the virtual environment

# Exporting tree to Graphviz and render PNG
# DotExporter(dummy_root).to_picture("/home/stud-msh/stat-master-nace/src/data/forest.png")
DotExporter(
    dummy_root,
    nodeattrfunc=lambda node: f'label="{getattr(node, "alias", node.name)}"',
).to_dotfile("forest.dot")

import pydot

(graph,) = pydot.graph_from_dot_file("forest.dot")
graph.write_png("forest.png")
Image("forest.png")  # displays in Jupyter/Colab


# DotExporter(dummy_root,
#            nodeattrfunc=lambda node: 'shape=box, style=filled, fillcolor=lightblue' if node.children else 'shape=ellipse, fillcolor=lightgreen',
#            edgeattrfunc=lambda parent, child: 'color=gray').to_picture("forest_hidden_root.png")


# %%

# %%

# %%

# %%

# %%

# %%

# %%
