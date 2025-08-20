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

# %%
# imported libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.cm as cm
import fasttext


# Treningsdata
df = pd.read_parquet("/ssb/stamme01/data811/NACE/data/one_to_many.parquet")

# Overgangssett
df_overgang = pd.read_csv("/ssb/stamme01/data811/NACE/data/SN2025-SN2007_23.06.2025.csv", sep = ';')


# Hent ut data for gamle nace-koder, org-nr og fritekst.
# Antar gamle nace-koder er SN07
df_sn07 = df[['orgnr', 'tekst', 'SN2007']]


# Herfra kan du gruppere på over og underkategorier og få en oversikt over hvor mange virksomheter som befinner seg under hver kategori. Kan du herfra kjøre en enkel klassifiseringsmodell?

# Beskriv datasettet etter f.eks:
#   - hierarkisk oppbygning
groups = df_sn07.groupby('SN2007')


#   - antall per kategori på flere nivå.
counted_groups = groups.count()


#   - sammenheng mellom enkeltord og kategori.



# utforske koblingen mellom SN07 og SN25. Hvilke har 1:N koblinger og hvilke har ikke noen koblinger.
groups_07_25=df[['SN2007', 'SN2025']].groupby('SN2007')
groups_07_25.count() # Shape is (106, 1)

groups_25_07=df[['SN2007', 'SN2025']].groupby('SN2025')
groups_25_07.count() # Shape is (45, 1)



# %%
df.head()

# %%

#----------- df_overgang datasettet --------------

#1) Hvordan kategoriene i 2007 er bygd opp, hvordan kan man beskrive strukturen på disse (visualisering)? 

df_overgang[df_overgang['SN2007']=='* Har ingen korrespondanse i SN2007']
df_overgang = df_overgang.replace(to_replace='* Har ingen korrespondanse i SN2007',value=np.nan)
df_overgang = df_overgang.astype({'SN2007': float})

df_overgang['SN2007'].astype(float)
df_overgang['SN2025'].astype(float)



# %%

# 2) Hvor mange av NACE-kodene som har endret seg i 2025 settet. 
unmatched_07_25 = (df_overgang['SN2007'] != df_overgang['SN2025'])
count_07_25_unmatched = unmatched_07_25.sum()

# Antall nye koder i 2025:
new = df_overgang['SN2007'].isna()

# Antall endrede koder i 2025:
edit =df_overgang[
    df_overgang['SN2007'].notna() &
    df_overgang['SN2025'].notna() &
    (df_overgang['SN2007'] != df_overgang['SN2025'])]


# Antall koder som ikke finnes lenger i 2025
deleted = df_overgang['SN2025'].isna()

# Antall koder som har samme kode nummer, men annerledes beskrivelse
new_desc = df_overgang[(df_overgang['SN2007'] == df_overgang['SN2025']) & 
                       (df_overgang['SN2007 Tittel'] != df_overgang['SN2025 Tittel'])]

# %%
df.head()

# %%
df_overgang.head()

# %%
print(df_sn07)
print('number of companies under each NACE category')
print(counted_groups)



# %%
print('Number of NACE codes that have changed in the 2025 sett')
print(count_07_25_unmatched) #783 av 1241
print('new',new.sum()) #new 3
print('edit',edit.shape[0]) #edit 780
print('del',deleted.sum()) #del 0
print('new_desc',new_desc.shape[0]) #new_desc 455

# %%
# checking which codes have the 1:N or N:1 relationship.



# %%
#Fasttext 

train = df_sn07[:int(df_sn07.shape[0]*2/3)].copy()
test = df_sn07[int(df_sn07.shape[0]/3):].copy()
pred = test['tekst'].tolist()
print(repr(pred[1]))

# %%
test.index = pd.RangeIndex(start=0, stop=len(test))
test.iloc[1]

# %%
# Kjøre dataene inn på fasttext modellen (og andre relevante modeller) og se hva slags resultater vi får

# Format for FastText
train['fasttext_format'] = '__label__' + train['SN2007'] + ' ' + train['tekst']

# Save to a text file
output_file = 'fasttext_input.txt'
train['fasttext_format'].to_csv(output_file,  index=False, header=False)


# Skipgram model :
model = fasttext.train_supervised(input='fasttext_input.txt')
labels, probabilities = model.predict([pred[1]]) #if wanting to predict more than one label set k= number of labels.

print(labels)


# %%
#Hvilke hovedgrupper er størst? 62

# %% [markdown]
#
# Checking the distribution of the classes on Subclass category 10.201

# %%
train['Subclass'] = train['SN2007'].astype(float)
train['Subclass']

# %% [markdown]
#
# Checking the distribution of the classes on Class category 10.20

# %%
train['Class'] = round(train['Subclass'], 2).astype(str)
train['Class']

# %% [markdown]
#
# Checking the distribution of the classes on Group category 10.2

# %%
train['Group'] = round(train['Subclass'], 1).astype(str)
train['Group']

# %%
counts_gr = train['Group'].value_counts()
plt.bar(counts_gr.index, counts_gr.values)
plt.xlabel('Category')
plt.ylabel('Count')
plt.title('Count of Values in Category Column')
plt.xticks(rotation=90) # Rotates x-axis labels by 45 degrees
plt.show()

# %% [markdown]
#
# Checking the distribution of the classes on Division category 10

# %%
train['Division'] = train['Subclass'].astype(int).astype(str)
train['Division']

# %%

import seaborn as sns
from matplotlib.patches import Patch


tr_div_gr = train.groupby("Division")["Group"].nunique().sort_values(ascending=False)
div_dist = train['Division'].value_counts().reindex(tr_div_gr.index)

# generating colors(one per bar)
colors = cm.plasma(tr_div_gr.values / tr_div_gr.values.max())


plt.figure(figsize=(12,6))
plt.bar(div_dist.index, div_dist.values, color=colors)
plt.yscale('log')

plt.xlabel('Dicis')
plt.ylabel('Count')
plt.title('Distribution of the Division and Number of Hierarchical Splits')
plt.xticks(rotation=90) 

unique_counts = sorted(tr_div_gr.unique())
count_to_color = {count: cm.plasma(count / tr_div_gr.values.max()) for count in unique_counts}
legend_handles = [Patch(color=count_to_color[count], label=f"{count} groups") 
                  for count in unique_counts]

plt.legend(handles=legend_handles, title= "Number of Groups", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()


# %%
#This plot is the best one in my opinion


# Data
tr_div_gr = train.groupby("Division")["Group"].nunique().sort_values(ascending=False)
div_dist = train['Division'].value_counts().reindex(tr_div_gr.index)

fig, ax1 = plt.subplots(figsize=(12,6))

# Bar plot for counts
ax1.bar(tr_div_gr.index, div_dist.values)
ax1.set_xlabel('Division')
ax1.set_ylabel('Number of Groups', color='black')
ax1.tick_params(axis='y', labelcolor='black')
plt.xticks(rotation=90)
plt.title("Count of Groups per Division")

# secondary y-axis to indicate number of groups
ax2 = ax1.twinx()
ax2.plot(tr_div_gr.index, tr_div_gr.values, 'o-', color='red', label='Number of Groups')
ax2.set_ylabel('Distinict Number of Groups', color='red')
ax2.tick_params(axis='y', labelcolor='red')

plt.tight_layout()
plt.show()


# %%
#This plot is the best one in my opinion


# Data
tr_div_gr = train.groupby("Class")["Subclass"].nunique().sort_values(ascending=False)
div_dist = train['Class'].value_counts().reindex(tr_div_gr.index)

fig, ax1 = plt.subplots(figsize=(12,6))

# Bar plot for counts
ax1.bar(tr_div_gr.index, div_dist.values)


#Normalising the plots?
#plt.yscale('log')

ax1.set_xlabel('Classes')
ax1.set_ylabel('Number of Subclasses', color='black')
ax1.tick_params(axis='y', labelcolor='black')
plt.xticks(rotation=90)
plt.title("Count of Subclasses per Classes")

# secondary y-axis to indicate number of groups
ax2 = ax1.twinx()
ax2.plot(tr_div_gr.index, tr_div_gr.values, '-o', color='red', label='Number of Groups')
ax2.set_ylabel('Distinict Number of Subclasses', color='red')
ax2.tick_params(axis='y', labelcolor='red')
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
#Er det noen fellestrekk i friteksten i enkelte grupper som kan beskrives? 

# %%
