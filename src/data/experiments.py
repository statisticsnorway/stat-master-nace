# imported libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator
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
counted-groups = groups.count()


#   - sammenheng mellom enkeltord og kategori.



# utforske koblingen mellom SN07 og SN25. Hvilke har 1:N koblinger og hvilke har ikke noen koblinger.
groups_07_25=df[['SN2007', 'SN2025']].groupby('SN2007')
groups_07_25.count() # Shape is (106, 1)

groups_25_07=df[['SN2007', 'SN2025']].groupby('SN2025')
groups_25_07.count() # Shape is (45, 1)


# Kjøre dataene inn på fasttext modellen (og andre relevante modeller) og se hva slags resultater vi får 

train = df_sn07[:int(df_sn07.shape[0]*2/3)].copy()
test = df_sn07[int(df_sn07.shape[0]/3):].copy()
pred = test['tekst'].tolist()

test.index = pd.RangeIndex(start=0, stop=len(test))
#test.iloc[1]

# Format for FastText
train['fasttext_format'] = '__label__' + train['SN2007'] + ' ' + train['tekst']

# Save to a text file
output_file = 'fasttext_input.txt'
train['fasttext_format'].to_csv(output_file,  index=False, header=False)


# Skipgram model :
model = fasttext.train_supervised(input='fasttext_input.txt')
labels, probabilities = model.predict([pred[1]]) #if wanting to predict more than one label set k= number of labels.
#print(labels)

#----------- df_overgang datasettet --------------

#1) Hvordan kategoriene i 2007 er bygd opp, hvordan kan man beskrive strukturen på disse (visualisering)? 

df_overgang[df_overgang['SN2007']=='* Har ingen korrespondanse i SN2007']
df_overgang = df_overgang.replace(to_replace='* Har ingen korrespondanse i SN2007',value=np.nan)
df_overgang = df_overgang.astype({'SN2007': float})

df_overgang['SN2007'].astype(float)
df_overgang['SN2025'].astype(float)



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


#Det høres også greit ut å sjekke hvilke koder som har blitt splittet og samlet i 25-versjonen. Men begynn med å se på 2007-data. 

#Hvilke hovedgrupper er størst? 

#Hvilke grupper er det som blir mest oppsplittet? 

#Finn måter å visualisere hierarkisk informasjon på en oversiktlig måte. Hvilke områder er det som ser vanskelige ut når det kommer til å gi prediksjoner? 

#Er det noen fellestrekk i friteksten i enkelte grupper som kan beskrives? 


# Data
tr_div_gr = train.groupby("Division")["Group"].nunique().sort_values(ascending=False)
div_dist = train['Division'].value_counts().reindex(tr_div_gr.index)

def nace_structure_plot():

    fig, ax1 = plt.subplots(figsize=(12,6))

    # Bar plot for counts
    ax1.bar(tr_div_gr.index, div_dist.values)
    ax1.set_xlabel('Division')
    ax1.set_ylabel('Number of Values (count)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    plt.xticks(rotation=90)
    plt.title("Count of Values per Division (colored by number of groups)")

    # secondary y-axis to indicate number of groups
    ax2 = ax1.twinx()
    ax2.plot(tr_div_gr.index, tr_div_gr.values, 'o-', color='red', label='Number of Groups')
    ax2.set_ylabel('Number of Groups', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.tight_layout()
    plt.show()


if __name__=="__main__":
    print(df_sn07)
    print('number of companies under each NACE category')
    print(counted-groups)
    print('Number of NACE codes that have changed in the 2025 sett')
    print(count_07_25_unmatched) #783 av 1241
    print('new',new.sum()) #new 3
    print('edit',edit.shape[0]) #edit 780
    print('del',deleted.sum()) #del 0
    print('new_desc',new_desc.shape[0]) #new_desc 455
    print('prediction input text and label')
    test.iloc[1]
    print('predicted label')
    print(labels)
    nace_structure_plot()

