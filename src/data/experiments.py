# imported libraries
import pandas as pd


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


if __name__=="__main__":
    print(df_sn07)
    print('number of companies under each NACE category')
    print(counted-groups)
    print('Number of NACE codes that have changed in the 2025 sett')
    print(count_07_25_unmatched) #783 av 1241
    