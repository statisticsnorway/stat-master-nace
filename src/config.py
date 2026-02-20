import os
from datetime import date



# Path to save files
SAVE_PATH = os.path.expanduser('/fp/homes01/u01/ec-minas/stat-master-nace/')
#os.path.expanduser('/flash/project_465002259/minashe/stat-master-nace/')
#os.path.expanduser('~/HPLT-project/stat-master-nace/')

# Overgangssett
TRANSITION_DATA_PATH = "/ssb/stamme01/data811/NACE/data/SN2025-SN2007_23.06.2025.csv" # This incluced SN titles 

#HIERARCHY_DATA = "/ssb/stamme01/data811/NACE/data/nace25_hierarki.csv"
HIERARCHY_DATA_PRUNED = f"{SAVE_PATH}data/nace25_hierarki_pruned.csv"

# Random state
RANDOM_STATE = 42
THREAD = 16

# folders
DATA = f"{SAVE_PATH}data/"


# NACE 2025 Hierarchi
#today = '2026-01-22'#date.today().isoformat()
HIERARCHY_DATA = os.path.join(DATA, 'nace25_hierarki_nor.csv')
#f"https://data.ssb.no/api/klass/v1/classifications/6/codesAt.csv?date={today}"


# NACE data
DATA_BR_TRAIN = os.path.join(DATA, 'train_norwaydata.csv') #'https://minio.lab.sspcloud.fr/projet-aiml4os-wp10/NorwayData/train_norwaydata.parquet'
DATA_BR_TEST = os.path.join(DATA, 'test_norwaydata.csv') #'https://minio.lab.sspcloud.fr/projet-aiml4os-wp10/NorwayData/test_norwaydata.parquet'

# general train val test data
DATASETS=os.path.join(DATA,'datasets')
JSON_FILES=f"{SAVE_PATH}json_files/"


## lm data
DATA_LM=f"{DATA}data_lm/"
DATA_LM_TR_VAL_TE=f"{DATA_LM}tr_val_te/"
DATA_LM_TR_TE=f"{DATA_LM}tr_te/"

## fasttext data
DATA_FASTXT= f"{SAVE_PATH}data/data_fastxt/"
DATA_FX_TR_VAL_TE=f"{DATA_FASTXT}tr_val_te/"
DATA_FX_TR_TE=f"{DATA_FASTXT}tr_te/"


## fasttext models
MODELS_FASTXT = f"{SAVE_PATH}data/models_fastxt/"
M_F_H= f"{SAVE_PATH}data/models_fastxt_hier/"


# Restults folder 
## fasttext models
RES_DUMMY =  f"{SAVE_PATH}results/dummy/"
RES_FASTXT_FLAT = f"{SAVE_PATH}results/fasttext/flat"
RES_HIER_M = f"{SAVE_PATH}results/fasttext/hier_fastxt_model/"

# including name column vs not
RES_AUTO_TEXT_NAME=f"{SAVE_PATH}results/fasttext/autotune_text_navn/"
RES_AUTO_TEXT=f"{SAVE_PATH}results/fasttext/autotune_text/"

# including name column vs not cv
RES_CV_TEXT=f"{SAVE_PATH}results/fasttext/cv_text/"
RES_CV_TEXT_NAME=f"{SAVE_PATH}results/fasttext/cv_text_navn/"

# llm
RES_LM=f"{SAVE_PATH}results/llm/"
