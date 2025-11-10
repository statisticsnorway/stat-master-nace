import os
from datetime import date


# Treningsdata
DATA_BR_TRAIN = 'https://minio.lab.sspcloud.fr/projet-aiml4os-wp10/NorwayData/train_norwaydata.parquet'
DATA_BR_TEST = 'https://minio.lab.sspcloud.fr/projet-aiml4os-wp10/NorwayData/test_norwaydata.parquet'

# Path to save files
SAVE_PATH = os.path.expanduser('~/HPLT-project/stat-master-nace/')

# Overgangssett
TRANSITION_DATA_PATH = "/ssb/stamme01/data811/NACE/data/SN2025-SN2007_23.06.2025.csv" # This incluced SN titles 


# NACE 2025 Hierarchi
today = date.today().isoformat()
HIERARCHY_DATA = f"https://data.ssb.no/api/klass/v1/classifications/6/codesAt.csv?date={today}"

#HIERARCHY_DATA = "/ssb/stamme01/data811/NACE/data/nace25_hierarki.csv"
HIERARCHY_DATA_PRUNED = os.path.expanduser("~/HPLT-project/stat-master-nace/data/nace25_hierarki_pruned.csv")

# Random state
RANDOM_STATE = 42


# folders
DATA = os.path.expanduser('~/HPLT-project/stat-master-nace/data/')
DATA_FASTXT= os.path.expanduser('~/HPLT-project/stat-master-nace/data/data_fastxt/')
MODELS_FASTXT = os.path.expanduser('~/HPLT-project/stat-master-nace/data/models_fastxt/')
M_F_H= os.path.expanduser('~/HPLT-project/stat-master-nace/data/models_fasttext_hier/')


