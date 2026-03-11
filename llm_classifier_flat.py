"""
Script to NACE codes using the LLM model.
"""

import pandas as pd
import os
from pathlib import Path
import json
import torch
from vllm import LLM, SamplingParams
from src.parser import parse_args
from src.config import HIERARCHY_DATA, RES_LM, JSON_FILES
from src.utils.llm.flat_llm_util import (
    build_prompt,
    llm_call,
    mapping_code_names,
    validate_and_assign
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

args = parse_args()
PROJECT_ROOT = Path(__file__).resolve().parent


df_hier = pd.read_csv(HIERARCHY_DATA, dtype={'code':str}, sep=";", encoding="latin1")


BASE_COLUMNS = [
    'company_activity',
    'company_name',
    'division',
    'group',
    'class',
    'nace_21_code',
    'nace_21_description_nb'
]

PRED_COLUMNS = ['pred_subclass', 'pred_probs']

ALL_COLUMNS = BASE_COLUMNS + PRED_COLUMNS

def run_classify_nace(tokenizer, 
        llm,
        subclasses_name_code,
        subclasses_code,
        map_code_names,
        sampling_params, 
        input_file:str=args.test_data_file, 
        input:str|list[str] = ["company_activity", "company_name"],
        batch_size=args.batch_size, 
        output_file:str=os.path.join(RES_LM, args.output_file_flat),
        checkpoint_file=os.path.join(JSON_FILES ,args.checkpoint_file_flat) #"results/checkpoint_flat.json",
        ):
    start_row = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            start_row = checkpoint.get('last_row')

    
    for batch in pd.read_csv(input_file, 
                             dtype={
                                 'company_activity':str,
                                 'company_name':str,
                                 'division':str, 
                                 'group':str, 
                                 'class':str, 
                                 'nace_21_code':str,
                                 'nace_21_description_nb':str
                                 }, 
                             chunksize=batch_size,
                             keep_default_na=False, 
                             na_values=[],
                             index_col='orgnr',
                             skiprows=range(1, start_row+1)
                             ):
        
    
        batch = batch.fillna("")
        descriptions=batch[input].astype(str).agg(" ".join, axis=1).to_dict()

        # results per company
        batch_results = {idx: {} for idx in batch.index}

        # flat prompts with all subclasses
        prompts = build_prompt(
            descriptions=descriptions,
            options=subclasses_name_code
        )

        preds, probs = llm_call(
            tokenizer=tokenizer,
            llm=llm, 
            sampling_params=sampling_params,
            prompts=prompts,
            subclasses_code =subclasses_code, 
            map_code_names=map_code_names)

        print('########### preds\n',preds)
        print('########### probs\n',probs)


        batch_results = validate_and_assign(
            preds,
            probs,
            batch_results,
        )
        print('batch results\n',batch_results)
        # Add predictions to batch DataFrame
        for idx, res in batch_results.items():
            batch.loc[idx, 'pred_subclass'] = res.get('subclass', '')
            batch.loc[idx, 'pred_probs'] = res.get('subclass_probs', '')
            

        batch = batch.reindex(columns=ALL_COLUMNS)
        assert list(batch.columns) == ALL_COLUMNS, f"Schema mismatch: {batch.columns}"

        # Write to CSV
        batch.to_csv(
            output_file,
            mode='a',
            header=not os.path.exists(output_file),
            index=True
        )

        start_row+=len(batch)
        # update checkpoint
        with open(checkpoint_file, "w") as f:
            json.dump({"last_row": int(start_row)}, f)
    return 'Classification is finished'

if __name__ == "__main__":    
    num_visible = torch.cuda.device_count()
    
    print('num_visible ',num_visible)
    sampling_params = SamplingParams(
        temperature=0, # 0 means deterministic decoding
        max_tokens=50,
        logprobs=1
    )
    model = LLM(args.model_name,
                tensor_parallel_size=num_visible) # bigger models may require more GPUs and higher tensor parallel size

    tokenizer = model.get_tokenizer()
   
    map_name = df_hier[['code', 'name']].set_index('code')['name'].to_dict()
    subclasses_name_code = tuple(df_hier[df_hier['level']==5][['code','name']].astype(str).agg(" - ".join, axis=1))
    subclasses_code = tuple(df_hier[df_hier['level']==5]['code'].astype(str))

    map_code_names = mapping_code_names(df=df_hier[df_hier['level']==5],  subclass_col='code', map_name=map_name)
    message = run_classify_nace(
        tokenizer=tokenizer, 
        llm=model,
        subclasses_name_code=subclasses_name_code,
        subclasses_code=subclasses_code,
        map_code_names=map_code_names,
        sampling_params=sampling_params,
        )
    print(message)
    
