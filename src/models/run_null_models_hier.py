# Rerunning the models that return NaN errors in hierarchical fasttext model

import fasttext
import os
import glob
import json
from config import DATA, SAVE_PATH

with open(f"{SAVE_PATH}NaN_fasttext_hier_model_paths.json", "r") as f:
    data = json.load(f)


div = list(data["division"].keys())
gr = list(data["group"].keys())
cls = list(data["class"].keys())
sub = list(data["nace_21_code"].keys())
print(div)

i=3
all = [div, gr, cls, sub][i]
par=['division', 'group', 'class', 'nace_21_code'][i]


for g in all:
    temp_dir = os.path.expanduser(f"{DATA}temp_fastxt/{par}_{g}.txt")
    model_path = os.path.expanduser(f"{DATA}models_fasttext_hier/{par}_{g}.bin")
    print(temp_dir)

    model = fasttext.train_supervised(input=temp_dir,
                                            lr=0.12106896936002161,
                                            epoch=10,
                                            wordNgrams=1,
                                            seed=42,
                                            )
    model.save_model(model_path)

    with open(f"{SAVE_PATH}fasttext_hier_model_paths.json", "r") as f:
        data = json.load(f)
        data[par][g] = model_path

    # Save updated JSON
    with open(f"{SAVE_PATH}fasttext_hier_model_paths.json", "w") as f:
        json.dump(data, f, indent=4, sort_keys=True)
    
    print(model_path)




















"""

def train_hier_fasttext():
    models_paths={}
    temp_dir = os.path.expanduser("~/HPLT-project/stat-master-nace/data/temp_fastxt")
    
    # deeper levels
    for p in glob.glob(f"{temp_dir}/*.txt"):
        train_path = p
        print('train_path')
        print(train_path)

        #model_path = os.path.join(save_dir, f"{current_label}_{parent}.bin")
        
        model = fasttext.train_supervised(input=train_path,
                                        lr=0.01,
                                        epoch=10,
                                        wordNgrams=1,
                                        seed=42,
                                        thread=4)

        #model.save_model(model_path)
        #models_paths[current_label][parent] = model_path
            

    #with open(f'fasttext_hier_model_paths.json', 'w') as f:
    #    json.dump(models_paths, f, indent=4)
    
    #shutil.rmtree(temp_dir)        
    return models_paths

train_hier_fasttext()
"""