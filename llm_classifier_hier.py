"""
Script to classify NACE codes using the LLama model with hierarchical prompt.
"""
import pandas as pd
import os
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from vllm import LLM, SamplingParams
from src.parser import parse_args
from src.config import HIERARCHY_DATA, RES_LM
from src.metrics import metrics_levels, df_to_table
from src.utils.llm.hier_llm_util import (
    build_prompt,
    llm_call,
    #llm_call_fake,
    sections_df_hier,
    derive_hier_names,
    mapping_code_names,
    build_parent_child_map,
    auto_descend,
    companies_at_level,
    prepare_level_prompts,
    validate_and_assign,
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
        sections,
        hierarchy,
        map_code_names,
        input_file:str=args.test_data_file, 
        input:str|list[str] = ["company_activity", "company_name"],
        sampling_params=sampling_params, 
        batch_size=1,#args.batch_size, 
        output_file:str=args.output_file_hier,
        levels=["section", "division", "group", "class", "subclass"],
        checkpoint_file="checkpoint_hier.json",
        ):
    
    """if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            last_idx = json.load(f)["last_idx"]
    else:
        last_idx = -1"""

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
        if i <= last_idx:
            continue
        
        batch = batch.fillna("")
        """
    input_df = pd.read_csv(input_file, dtype={'company_activity':str,'company_name':str,'division':str, 'group':str, 'class':str, 'nace_21_code':str,'nace_21_description_nb':str}, keep_default_na=False, na_values=[], index_col=0).fillna("")
    input_df=input_df.iloc[:2]
    for i in range(len(input_df)):
        batch = input_df.iloc[i:i + batch_size].copy()

        if 'pred_subclass' not in batch.columns:
            batch.loc[:,'pred_subclass'] = ""
        descriptions=batch[input].astype(str).agg(" ".join, axis=1).to_dict()
        # results per company
        batch_results = {idx: {} for idx in batch.index}


        # Section
        section_prompt = build_prompt(
            descriptions=descriptions,
            level_name="section",
            options=sections
        )

        section_preds = llm_call(
            tokenizer=tokenizer, 
            llm=llm, 
            sampling_params=sampling_params,
            prompts=section_prompt,
            hierarchy=hierarchy, 
            level="section", 
            map_code_names=map_code_names)

        # auto-descend to fill unambiguous levels
        for idx in section_preds:
            pred = section_preds[idx]
            final_code, final_level = auto_descend(pred, hierarchy)
            batch_results[idx][final_level] = final_code
            
        for i in range(len(levels) - 1):
            current_level = levels[i]
            next_level = levels[i + 1]

            companies = companies_at_level(
                batch_results, current_level, map_hier_indx
            )

            if not companies:
                continue

            prompts, meta = prepare_level_prompts(
                companies,
                batch_results,
                hierarchy,
                current_level,
                next_level,
                descriptions
            )

            if not prompts:
                continue

            preds = llm_call(
                tokenizer=tokenizer,
                llm=llm,
                sampling_params=sampling_params,
                prompts=prompts,
                hierarchy=hierarchy,
                level=next_level,
                map_code_names=map_code_names
            )
            batch_results = validate_and_assign(
                preds,
                meta,
                batch_results,
                hierarchy
            )
            

        # Add predictions to batch DataFrame
        for idx, res in batch_results.items():
            batch.loc[idx, "pred_subclass"] = res.get("subclass", "")

        batch = batch.reindex(columns=ALL_COLUMNS)
        assert list(batch.columns) == ALL_COLUMNS, f"Schema mismatch: {batch.columns}"
        print(batch)
        # Write to CSV
        batch.to_csv(
            output_file,
            mode='a',
            header=not os.path.exists(output_file),
            index=False
        )
        """
        # update checkpoint
        with open(checkpoint_file, "w") as f:
            json.dump({"last_idx": i}, f)"""

    return batch

        
        
if __name__ == "__main__":
    model = LLM(args.model_name,
            tensor_parallel_size=args.num_gpus) # bigger models may require more GPUs and higher tensor parallel size
    tokenizer = model.get_tokenizer()
   
    SECTIONS = sections_df_hier(df_hier)
    map_sec = dict(zip(df_hier[df_hier['level']==2]["code"], df_hier[df_hier['level']==2]["parentCode"]))
    map_name = df_hier[['code', 'name']].set_index('code')['name'].to_dict()
    df_hier_hier = derive_hier_names(df=df_hier[df_hier['level']==5],  subclass_col='code', section_map=map_sec, map_name=map_name)
    HIERARCHY = build_parent_child_map(df_hier_hier)
    map_code_names = mapping_code_names(df=df_hier[df_hier['level']==5],  subclass_col='code', section_map=map_sec, map_name=map_name)
    
    df_with_res = run_classify_nace(
        tokenizer=tokenizer, 
        llm=model, 
        sections=SECTIONS, 
        hierarchy=HIERARCHY, 
        map_code_names=map_code_names)
    
    res_sub, res_cl, res_gro, res_div = metrics_levels(target=df_with_res['nace_21_code'], pred=df_with_res['pred_subclass']) # type: ignore
    with PdfPages(f"{RES_LM}test_hier_results.pdf") as pdf:
        pdf.savefig(df_to_table(res_sub, "Subclass Results"))
        pdf.savefig(df_to_table(res_cl, "Class Results"))
        pdf.savefig(df_to_table(res_gro, "Group Results"))
        pdf.savefig(df_to_table(res_div, "Division Results"))