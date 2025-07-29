# imported libraries
import pandas as pd


# Treningsdata
df = pd.read_parquet("/ssb/stamme01/data811/NACE/data/one_to_many.parquet")


# Hent ut data for gamle nace-koder, org-nr og fritekst.
# Antar gamle nace-koder er SN07

df_sn07 = df[['orgnr', 'tekst', 'SN2007']]


# Beskriv datasettet etter f.eks:
#   - hierarkisk oppbygning
#   - antall per kategori på flere nivå.
#   - sammenheng mellom enkeltord og kategori.


# utforske koblingen mellom SN07 og SN25. Hvilke har 1:N koblinger og hvilke har ikke noen koblinger.

# Kjøre dataene inn på fasttext modellen (og andre relevante modeller) og se hva slags resultater vi får 



if __name__=="__main__":
    print(df_sn07)
    