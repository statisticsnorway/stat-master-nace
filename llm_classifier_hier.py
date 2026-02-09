"""
Script to classify NACE codes using the LLM model with hierarchical prompt.
"""
import pandas as pd
import os
from pathlib import Path
import torch
import json
from matplotlib.backends.backend_pdf import PdfPages
from vllm import LLM, SamplingParams
from src.parser import parse_args
from src.config import HIERARCHY_DATA, RES_LM
from src.metrics import metrics_levels, df_to_table
from src.utils.llm.hier_llm_util import (
    llm_call,
    sections_df_hier,
    derive_hier_names,
    derive_hier,
    mapping_code_names,
    build_parent_child_map,
    companies_at_level,
    prepare_level_prompts,
    validate_and_assign,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

args = parse_args()
"""
# ================= DEBUG CHECKS =================
print("Starting hierarchical classification")

# Check that input files exist
assert os.path.exists(args.test_data_file), f"{args.test_data_file} not found!"
assert os.path.exists(HIERARCHY_DATA), f"{HIERARCHY_DATA} not found!"
print(f"Found test data file: {args.test_data_file}")
print(f"Found hierarchy file: {HIERARCHY_DATA}")
# ==============================================

"""
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

PRED_COLUMNS = ['pred_subclass']

ALL_COLUMNS = BASE_COLUMNS + PRED_COLUMNS


def run_classify_nace(tokenizer, 
        llm,
        hierarchy,
        map_code_names,
        sampling_params, 
        input_file:str=args.test_data_file, 
        input:str|list[str] = ["company_activity", "company_name"],
        batch_size=1,#args.batch_size, 
        output_file:str=args.output_file_hier,
        levels=["root", "section", "division", "group", "class", "subclass"],
        checkpoint_file="results/checkpoint_hier.json",
        ):
    """current level is the level that presents possible classes at that level that can be classified. 
        next_level is the level that is being classified"""
    start_idx=None
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            checkpoint = json.load(f)
            start_idx=checkpoint.get('last_idx')

    map_hier_indx = {level: i for i, level in enumerate(levels)}

    """
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
                                 index_col=0,
                                )
    for i, batch in enumerate(reader):
    """
    input_df = pd.read_csv(input_file, 
                           dtype={'company_activity':str,
                                  'company_name':str,
                                  'division':str, 
                                  'group':str, 
                                  'class':str, 
                                  'nace_21_code':str,
                                  'nace_21_description_nb':str}, 
                            keep_default_na=False, 
                            na_values=[], 
                            index_col=0).fillna("")
    
    input_df=input_df.iloc[:10]
    for i_row in range(len(input_df)):
        
        batch = input_df.iloc[i_row:i_row + batch_size].copy()
        # continuing from last company processed
        if start_idx is not None and batch.index.max() <= start_idx:
            continue
        if start_idx is not None:
            batch = batch[batch.index > start_idx]
            if batch.empty:
                continue
        
        batch = batch.fillna("")

        if 'pred_subclass' not in batch.columns:
            batch.loc[:,'pred_subclass'] = ""
        descriptions=batch[input].astype(str).agg(" ".join, axis=1).to_dict()
        # results per company
        batch_results = {idx: {} for idx in batch.index}

        ############### all other levels ###################

        for i_level in range(len(levels) - 1):
            current_level = levels[i_level]
            next_level = levels[i_level + 1]

            print('current level', current_level)
            companies = companies_at_level(
                batch_results=batch_results,
                current_level=current_level,
                map_hier_indx=map_hier_indx
            )
            print('companies ', companies)

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

            print('promtpts next level ', prompts)
            
            #### including mapping from code to code-name  


            preds = llm_call(
                tokenizer=tokenizer,
                llm=llm,
                sampling_params=sampling_params,
                prompts=prompts,
                hierarchy_code=hierarchy,
                current_level=current_level,
                meta=meta,
            )
            print('preds ', preds)

            batch_results = validate_and_assign(
                preds=preds,
                meta=meta,
                batch_results=batch_results,
                hierarchy_code=hierarchy,
                current_level=current_level,
                next_level=next_level,
            )
      
        print('batch results', batch_results)

        if current_level == 'subclass':
            # Add predictions to batch DataFrame
            for idx, res in batch_results.items():
                batch.loc[idx, "pred_subclass"] = res.get("subclass", "")

            batch = batch.reindex(columns=ALL_COLUMNS)
            assert list(batch.columns) == ALL_COLUMNS, f"Schema mismatch: {batch.columns}"
            # Write to CSV
            batch.to_csv(
                output_file,
                mode='a',
                header=not os.path.exists(output_file),
                index=True
            )
            
            # update checkpoint
            with open(checkpoint_file, "w") as f:
                json.dump({"last_idx": int(batch.index.max())}, f)
    return None

        
        
if __name__ == "__main__":
    num_visible = torch.cuda.device_count()
    

    llm_fake=False
    if llm_fake:
        model=1
        tokenizer=1
        sampling_params=1
    else:
        model =LLM(args.model_name,
                    #max_model_len= 65536,
                    tensor_parallel_size=num_visible) # bigger models may require more GPUs and higher tensor parallel size
        tokenizer =model.get_tokenizer()
        
        sampling_params = SamplingParams(
            temperature=0.3,
            max_tokens=50,
            logprobs=1
            )
   
    #SECTIONS = sections_df_hier(df_hier)
    map_sec = dict(zip(df_hier[df_hier['level']==2]["code"], df_hier[df_hier['level']==2]["parentCode"]))
    map_name = df_hier[['code', 'name']].set_index('code')['name'].to_dict()
    
    df_hier_names = derive_hier_names(df=df_hier[df_hier['level']==5],  subclass_col='code', section_map=map_sec, map_name=map_name)
    df_hier_family = derive_hier(df=df_hier[df_hier['level']==5],  subclass_col='code', section_map=map_sec, map_name=map_name)

    HIERARCHY = build_parent_child_map(df_hier_family)
    map_code_names = mapping_code_names(df=df_hier[df_hier['level']==5],  subclass_col='code', section_map=map_sec, map_name=map_name)
    
    run_classify_nace(
        tokenizer=tokenizer, 
        llm=model, 
        #sections=SECTIONS, 
        hierarchy=HIERARCHY, 
        map_code_names=map_code_names,
        sampling_params=sampling_params)
    """
    res_sub, res_cl, res_gro, res_div = metrics_levels(target=df_with_res['nace_21_code'], pred=df_with_res['pred_subclass']) # type: ignore
    with PdfPages(f"{RES_LM}test_hier_results.pdf") as pdf:
        pdf.savefig(df_to_table(res_sub, "Subclass Results"))
        pdf.savefig(df_to_table(res_cl, "Class Results"))
        pdf.savefig(df_to_table(res_gro, "Group Results"))
        pdf.savefig(df_to_table(res_div, "Division Results"))"""