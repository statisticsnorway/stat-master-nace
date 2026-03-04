"""
Script to classify NACE codes using the LLM model with hierarchical prompt.
"""
import pandas as pd
import os
from pathlib import Path
import torch
import json
import math
from vllm import LLM, SamplingParams
from src.parser import parse_args
from src.config import HIERARCHY_DATA, RES_LM, JSON_FILES
from src.utils.llm.hier_llm_util import (
    llm_call,
    derive_hier_names,
    derive_hier,
    mapping_code_names,
    build_parent_child_map,
    companies_at_level,
    prepare_level_prompts,
    validate_and_assign,
)

#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

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

PRED_COLUMNS = ['pred_subclass', 'pred_level_probs', 'joint_prob', 'avg_prob']

ALL_COLUMNS = BASE_COLUMNS + PRED_COLUMNS


def run_classify_nace(tokenizer, 
        llm,
        hierarchy,
        map_code_names,
        sampling_params, 
        input_file:str=os.path.join(RES_LM, args.test_data_file), 
        input:str|list[str] = ["company_activity", "company_name"],
        output_file:str=os.path.join(RES_LM, args.output_file_hier),
        levels=["root", "section", "division", "group", "class", "subclass"],
        checkpoint_file=os.path.join(JSON_FILES, args.checkpoint_file_hier),
        ):
    """current level is the level that presents possible classes at that level that can be classified. 
        next_level is the level that is being classified"""
    start_row=0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            checkpoint = json.load(f)
            start_row=checkpoint.get('last_row')

    map_hier_indx = {level: i for i, level in enumerate(levels)}
    reader = pd.read_csv(input_file, 
                             dtype={
                                 'company_activity':str,
                                 'company_name':str,
                                 'division':str, 
                                 'group':str, 
                                 'class':str, 
                                 'nace_21_code':str,
                                 'nace_21_description_nb':str
                                 }, 
                                 keep_default_na=False, 
                                 na_values=[],
                                 chunksize=args.batch_size,
                                 index_col='orgnr',
                                 skiprows=range(1, start_row+1),
                                )
    for i, batch in enumerate(reader):        
        batch = batch.fillna("")

        if 'pred_subclass' not in batch.columns:
            batch.loc[:,'pred_subclass'] = ""
        descriptions=batch[input].astype(str).agg(" ".join, axis=1).to_dict()
        # results per company
        batch_results = {idx: {} for idx in batch.index}
        level_probs = {idx: {} for idx in batch.index}

        ############### all levels ###################

        for i_level in range(len(levels) - 1):
            current_level = levels[i_level]
            next_level = levels[i_level + 1]

            print('current level', current_level)
            companies = companies_at_level(
                batch_results=batch_results,
                current_level=current_level,
                map_hier_indx=map_hier_indx
            )
            #print('companies ', companies)

            if not companies:
                continue

            prompts, meta = prepare_level_prompts(
                companies=companies,
                batch_results=batch_results,
                hierarchy_code=hierarchy,
                current_level=current_level,
                next_level=next_level,
                descriptions=descriptions,
                map_code_name=map_code_names
            )

            if not prompts:
                continue
            #print('meta \n ', meta)                    
            preds, level_probs = llm_call(
                tokenizer=tokenizer,
                llm=llm,
                sampling_params=sampling_params,
                prompts=prompts,
                current_level=current_level,
                next_level=next_level,
                meta=meta,
                map_code_name=map_code_names,
                level_probs=level_probs,
            )
            print('preds ', preds)
            print('level_probs', level_probs)


            batch_results = validate_and_assign(
                preds=preds,
                meta=meta,
                batch_results=batch_results,
                hierarchy_code=hierarchy,
                current_level=current_level,
                next_level=next_level,
            )
      
        print('batch results', batch_results)
        print('level_probs', level_probs)


        if next_level == 'subclass':
            # Add predictions to batch DataFrame
            for idx in batch_results:
                batch.loc[idx, "pred_subclass"] = batch_results[idx]["subclass"]
                print('level_probs[idx] ', level_probs[idx])
		probs = level_probs[idx].values()

		if any(v is None or v <= 0 for v in probs):
		    joint_logprob = None
		else:
		    joint_logprob = sum(math.log(v) for v in probs)
		    avg_logprob = sum(log_probs) / len(log_probs)

		#batch.loc[idx, 'subclass_prob'] = joint_logprob
		batch.loc[idx, 'joint_prob'] = math.exp(joint_logprob)
		batch.loc[idx, 'avg_prob'] = math.exp(avg_logprob)
                #batch.loc[idx, 'subclass_probs'] =  sum(math.log(v) for v in level_probs[idx].values())
                batch.loc[idx, 'pred_level_probs'] = json.dumps(level_probs[idx])

            print('batch subclass \n', batch)
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
    return None

        
        
if __name__ == "__main__":
    num_visible = torch.cuda.device_count()
    

    llm_fake=False
    if llm_fake:
        model=1
        tokenizer=1
        sampling_params=1
    else:
        model=LLM(
            args.model_name,
            tensor_parallel_size=num_visible) # bigger models may require more GPUs and higher tensor parallel size
        
        tokenizer=model.get_tokenizer()
        
        sampling_params = SamplingParams(
            temperature=0, #deterministic sampling
            max_tokens=20,
            logprobs=1
            )
   
    map_sec = dict(zip(df_hier[df_hier['level']==2]["code"], df_hier[df_hier['level']==2]["parentCode"]))
    map_name = df_hier[['code', 'name']].set_index('code')['name'].to_dict()
    
    df_hier_names = derive_hier_names(df=df_hier[df_hier['level']==5],  subclass_col='code', section_map=map_sec, map_name=map_name)
    df_hier_family = derive_hier(df=df_hier[df_hier['level']==5],  subclass_col='code', section_map=map_sec, map_name=map_name)

    HIERARCHY = build_parent_child_map(df_hier_family)
    map_code_names = mapping_code_names(df=df_hier[df_hier['level']==5],  subclass_col='code', section_map=map_sec, map_name=map_name)
    
    run_classify_nace(
        tokenizer=tokenizer, 
        llm=model, 
        hierarchy=HIERARCHY, 
        map_code_names=map_code_names,
        sampling_params=sampling_params)
    
