# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python3
#     language: python
#     name: python3
# ---

# %%
import pandas as pd

#treningsdata
df = pd.read_parquet("/ssb/stamme01/data811/NACE/data/one_to_many.parquet")

print(df.head())

# %%
print(df.columns)

# %%
df_sn07 = df[['orgnr', 'tekst', 'SN2007']]
df_sn07
