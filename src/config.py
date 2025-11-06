import os
import pandas as pd
from datetime import date

DATA_BR_TRAIN = 'https://minio.lab.sspcloud.fr/projet-aiml4os-wp10/NorwayData/train_norwaydata.parquet'
DATA_BR_TEST = 'https://minio.lab.sspcloud.fr/projet-aiml4os-wp10/NorwayData/test_norwaydata.parquet'

# Treningsdata
DATA_PATH = ...
OLD_OLD_DATA = "/ssb/stamme01/data811/NACE/data/foretak_med_formaal_sn2025.parquet" 
OLD_DATA = "/ssb/stamme01/data811/NACE/data/one_to_many.parquet" # This includes the connection between companies and SN07 and SN25 code.

# Overgangssett
TRANSITION_DATA_PATH = "/ssb/stamme01/data811/NACE/data/SN2025-SN2007_23.06.2025.csv" # This incluced SN titles 


# NACE 2025 Hierarchi
today = date.today().isoformat()
HIERARCHY_DATA = f"https://data.ssb.no/api/klass/v1/classifications/6/codesAt.csv?date={today}&language=en"
#HIERARCHY_DATA = "/ssb/stamme01/data811/NACE/data/nace25_hierarki.csv"
HIERARCHY_DATA_PRUNED = "/ssb/stamme01/data811/NACE/data/nace25_hierarki_pruned.csv"

# Random state
RANDOM_STATE = 42

# Path to save files
SAVE_PATH = '~/HPLT-project/stat-master-nace/'

