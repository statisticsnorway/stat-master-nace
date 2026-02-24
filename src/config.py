import os
from datetime import date



# Path to save files
SAVE_PATH = os.path.expanduser('/fp/homes01/u01/ec-minas/stat-master-nace/')
#SAVE_RES = os.path.expanduser('/fp/homes01/u01/ec-minas/stat-master-nace/')
SAVE_DATA = os.path.expanduser('/cluster/work/projects/ec403/ec-minashe-master/')
#os.path.expanduser('/flash/project_465002259/minashe/stat-master-nace/')
#os.path.expanduser('~/HPLT-project/stat-master-nace/')

# Overgangssett
TRANSITION_DATA_PATH = "/ssb/stamme01/data811/NACE/data/SN2025-SN2007_23.06.2025.csv" # This incluced SN titles 

#HIERARCHY_DATA = "/ssb/stamme01/data811/NACE/data/nace25_hierarki.csv"
HIERARCHY_DATA_PRUNED = f"{SAVE_DATA}data/nace25_hierarki_pruned.csv"

# Random state
RANDOM_STATE = 42
THREAD = 16

# folders
DATA = os.path.join(SAVE_DATA, 'data')
#f"{SAVE_PATH}data/"


# NACE 2025 Hierarchi
#today = '2026-01-22'#date.today().isoformat()
HIERARCHY_DATA = os.path.join(SAVE_DATA, 'data', 'nace25_hierarki_nor.csv')
#f"https://data.ssb.no/api/klass/v1/classifications/6/codesAt.csv?date={today}"


# NACE data
DATA_BR_TRAIN = os.path.join(DATA, 'train_norwaydata.csv') #'https://minio.lab.sspcloud.fr/projet-aiml4os-wp10/NorwayData/train_norwaydata.parquet'
DATA_BR_TEST = os.path.join(DATA, 'test_norwaydata.csv') #'https://minio.lab.sspcloud.fr/projet-aiml4os-wp10/NorwayData/test_norwaydata.parquet'

# general train val test data
DATASETS=os.path.join(DATA,'datasets')

# json files
JSON_FILES = os.path.join(SAVE_PATH, "json_files")

## lm data
DATA_LM = os.path.join(DATA, "data_lm")
DATA_LM_TR_VAL_TE = os.path.join(DATA_LM, "tr_val_te")
DATA_LM_TR_TE = os.path.join(DATA_LM, "tr_te")

## fasttext data
DATA_FASTXT = os.path.join(DATA, "data_fastxt")
DATA_FX_TR_VAL_TE = os.path.join(DATA_FASTXT, "tr_val_te")
DATA_FX_TR_TE = os.path.join(DATA_FASTXT, "tr_te")

## fasttext models
MODELS_FASTXT = os.path.join(DATA, "models_fastxt")
M_F_H = os.path.join(DATA, "models_fastxt_hier")

# Results folder
RES = os.path.join(SAVE_PATH, "results")

## fasttext results
RES_DUMMY = os.path.join(RES, "dummy")
RES_FASTXT_FLAT = os.path.join(RES, "fasttext", "flat")
RES_HIER_FASTXT = os.path.join(RES, "fasttext", "hier_fastxt_model")

# including name column vs not
RES_AUTO_TEXT_NAME = os.path.join(RES, "fasttext", "autotune_text_navn")
RES_AUTO_TEXT = os.path.join(RES, "fasttext", "autotune_text")

# including name column vs not cv
RES_CV_TEXT = os.path.join(RES, "fasttext", "cv_text")
RES_CV_TEXT_NAME = os.path.join(RES, "fasttext", "cv_text_navn")

# llm
RES_LM = os.path.join(RES, "llm")