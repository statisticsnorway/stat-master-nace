"""
Script to NACE codes using the LLama model.
"""

import pandas as pd
import os
from pathlib import Path
import json
from matplotlib.backends.backend_pdf import PdfPages
from vllm import LLM, SamplingParams
from src.parser import parse_args
from src.config import HIERARCHY_DATA, RES_LM
from src.metrics import metrics_levels, df_to_table
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

sampling_params = SamplingParams(
        temperature=0.3,
        max_tokens=50,
        logprobs=1
    )

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
        subclasses,
        map_code_names,
        input_file:str=args.test_data_file, 
        input:str|list[str] = ["company_activity", "company_name"],
        sampling_params=sampling_params, 
        batch_size=args.batch_size, 
        output_file:str=args.output_file_flat,
        checkpoint_file="checkpoint_flat.json",
        ):
    
    """
    input_df = pd.read_csv(input_file, dtype={'company_activity':str,'company_name':str,'division':str, 'group':str, 'class':str, 'nace_21_code':str,'nace_21_description_nb':str}, keep_default_na=False, na_values=[]).fillna("")
    input_df=input_df.iloc[:200]
    for i in range(0, len(input_df), 2):
        batch = input_df.iloc[i:i + batch_size]
    """
    
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
                             na_values=[]
                             ):
        batch = batch.fillna("")

        descriptions=batch[input].astype(str).agg(" ".join, axis=1).to_dict()

        # results per company
        batch_results = {idx: {} for idx in batch.index}

        # flat prompts with all subclasses
        prompts = build_prompt(
            descriptions=descriptions,
            options=subclasses
        )

        preds = llm_call(
            tokenizer=tokenizer, 
            llm=llm, 
            sampling_params=sampling_params,
            prompts=prompts,
            subclasses=subclasses, 
            map_code_names=map_code_names)

        batch_results = validate_and_assign(
            preds,
            batch_results,
        )
    
        # Add predictions to batch DataFrame
        for idx, res in batch_results.items():
            batch.loc[idx, 'pred_subclass'] = res.get('subclass', '')

        batch = batch.reindex(columns=ALL_COLUMNS)
        assert list(batch.columns) == ALL_COLUMNS, f"Schema mismatch: {batch.columns}"

        # Write to CSV
        batch.to_csv(
            output_file,
            mode='a',
            header=not os.path.exists(output_file),
            index=False
        )
        
        # update checkpoint
        with open(checkpoint_file, "w") as f:
            json.dump({"last_idx": idx}, f)
    return batch

if __name__ == "__main__":
    model = LLM(args.model_name,
            tensor_parallel_size=args.num_gpus) # bigger models may require more GPUs and higher tensor parallel size

    tokenizer = model.get_tokenizer()
   
    map_name = df_hier[['code', 'name']].set_index('code')['name'].to_dict()
    subclasses = tuple(df_hier[df_hier['level']==5][['code','name']].astype(str).agg(" - ".join, axis=1))
    map_code_names = mapping_code_names(df=df_hier[df_hier['level']==5],  subclass_col='code', map_name=map_name)
    df_with_res = run_classify_nace(
        tokenizer=tokenizer, 
        llm=model,
        subclasses=subclasses,
        map_code_names=map_code_names,
        )

    res_sub, res_cl, res_gro, res_div = metrics_levels(
        target=df_with_res['nace_21_code'], 
        pred=df_with_res['pred_subclass']) 

    with PdfPages(f"{RES_LM}test_flat_results.pdf") as pdf:
        pdf.savefig(df_to_table(res_sub, "Subclass Results"))
        pdf.savefig(df_to_table(res_cl, "Class Results"))
        pdf.savefig(df_to_table(res_gro, "Group Results"))
        pdf.savefig(df_to_table(res_div, "Division Results"))
    
