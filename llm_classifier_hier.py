"""
Script to calculate dataset scores using the LLama.
"""
import pandas as pd
import re
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from vllm import LLM, SamplingParams
import multiprocessing as mp
from src.parser import parse_args
from src.config import HIERARCHY_DATA, HIERARCHY_DATA_PRUNED, RES_LM
from src.metrics import metrics_levels, df_to_table

args = parse_args()
PROJECT_ROOT = Path(__file__).resolve().parent


df_hier = pd.read_csv(HIERARCHY_DATA, encoding="latin1")
df_hier_pruned = pd.read_csv(HIERARCHY_DATA_PRUNED, encoding="latin1")

sampling_params = SamplingParams(
        temperature=0.3,
        max_tokens=2048,
        logprobs=1
    )


def build_prompt(descriptions, level_name, options, parent=None):
    prompts=[]
    options_str = "\n".join(f"- {opt}" for opt in options)
    parent_str = f"\nParent category: {parent}" if parent else ""
    
    for description in descriptions:
        prompts.append(f"""
        Company description:
        {description}

        Task:
        Select the most appropriate {level_name}.{parent_str}

        Possible {level_name}s:
        {options_str}

        Answer with exactly ONE code from the list above.
        """.strip())
    return prompts


def extract_class(output_text, hierarchy, level, map_code_names):
    # Removeing "assistant" prefix if present
    assistant_prefix = "assistant\n\n"
    if output_text.startswith(assistant_prefix):
        output_text = output_text[len(assistant_prefix):]

    if output_text is None:
        print(f"No code found in model output:\n{output_text}")
        return None
    
    if output_text not in hierarchy[level]:
        if output_text in map_code_names:
            output_text = map_code_names[output_text]
        else:
            print(f"{output_text} not in HIERARCHY")
            return None
        
    return output_text


def llm_call(tokenizer, llm, sampling_params, prompts, hierarchy, level, map_code_names):
    """
    texts: list of strings
    returns: list of (text, score)
    """
    prompts_tokenized = tokenizer.apply_chat_template(
        [
            [
            {"role": "system", 
            "content": "You are an expert in SN2025 classification."},
            {"role": "user", "content": p}
            ] for p in prompts
        ],
        tokenize=False,
        add_generation_prompt=True
        )
        

    # Runing batched inference
    outputs = llm.generate(prompts_tokenized, sampling_params)

    # Extracting scores
    results = []
    results_probs = []
    for out in outputs:
        raw = out.outputs[0].text
        probs = out.outputs[0].logprobs
        print('--------- raw prompt ------------')
        print(raw)
        nace_code = extract_class(output_text=raw, hierarchy=hierarchy, level=level, map_code_names=map_code_names)
        results.append(nace_code)
        results_probs.append(probs)
        print('############### result logits \n', results_probs) #----------------- legge til sannsynlighetene i run functionen
    return results


"""
def format_output(id, text, score, gold_score=None):
    if score is None:
        return None
        
    data_res = {"id": id, "text": text, "score": score}

    if gold_score is not None:
        data_res["gold score"] = gold_score

    print("----------- OUT PROMPT-----------")
    print(data_res)
    return data_res
"""


def sections_df_hier(df:pd.DataFrame):
    sections = df[df['level']==1][['code','name']].astype(str).agg(" - ".join, axis=1).tolist()
    return sections


def derive_hier_names(df: pd.DataFrame, subclass_col:str, section_map, map_name):
    """  ["section", "division", "group", "class"] """
    df = df.copy()
    codes_div = df[subclass_col].str[:2]
    names_div = codes_div.map(map_name)
    df["division"] = codes_div + " - " + names_div

    codes_sec = codes_div.map(section_map) 
    names_sec = codes_sec.map(map_name)
    df["section"] = codes_sec + " - " + names_sec

    codes_gr = df[subclass_col].str[:4]
    names_gr = codes_gr.map(map_name)
    df["group"] = codes_gr + " - " + names_gr

    codes_cls = df[subclass_col].str[:5]
    names_cls = df[subclass_col].map(map_name)
    df["class"] = codes_cls + " - " + names_cls

    return df

def mapping_code_names(df: pd.DataFrame, subclass_col:str, section_map:dict, map_name:dict):
    """  ["section", "division", "group", "class"] """
    df = df.copy()

    df_div = pd.DataFrame({
    "code": df[subclass_col].str[:2],
    "name": df[subclass_col].str[:2] + " - " + df[subclass_col].str[:2].map(map_name)
    })

    df_sec = pd.DataFrame({
        "code": df[subclass_col].str[:2].map(section_map),
        "name": df[subclass_col].str[:2].map(section_map) + " - " + df[subclass_col].str[:2].map(section_map).map(map_name)
    })

    df_gr = pd.DataFrame({
        "code": df[subclass_col].str[:4],
        "name": df[subclass_col].str[:4] + " - " + df[subclass_col].str[:4].map(map_name)
    })

    df_cls = pd.DataFrame({
        "code": df[subclass_col].str[:5],
        "name": df[subclass_col].str[:5] + " - " + df[subclass_col].map(map_name)
    })

    map_df = pd.concat([df_div, df_sec, df_gr, df_cls], ignore_index=True).drop_duplicates().set_index("code")["name"].to_dict()
    return map_df


def build_parent_child_map(df, hier=['section', 'division', 'group', 'class', 'code']):
    all_parent_child = {}
    for i in range(len(hier)-1):
        parent_child  = (
            df[[hier[i], hier[i+1]]]
            .dropna()
            .groupby(hier[i])[hier[i+1]]
            .apply(lambda x: list(set(x)))
            .to_dict()
        )
        all_parent_child[hier[i]]=parent_child

    return all_parent_child

def auto_descend(code, hierarchy):
    """
    Walk down hierarchy automatically
    while only ONE child exists.
    """
    child_level = {
    "section":"division",
    "division": "group",
    "group": "class",
    "class" :"subclass"
    }
    for level in ["section", "division", "group", "class"]:
        if code in hierarchy[level] and len(hierarchy[level][code]) == 1:
            code = hierarchy[level][code][0]
        else:
            return code, level
    
    return code, child_level[level]



def run_classify_nace(tokenizer, 
                      llm,
                      sections,
                      hierarchy,
                      map_code_names,
                      input_file:str=args.val_data_file, 
                      input:str|list[str] = ["company_activity", "company_name"],
                      sampling_params=sampling_params, 
                      batch_size=args.batch_size, 
                      output_file:str=args.output_file,
                      ):
    #results = {}
    levels = ["section", "division", "group", "class", "subclass"]
    map_hier_indx = {level: i for i, level in enumerate(levels)}

    parent_level = {
        "division": "section",
        "group": "division",
        "class": "group",
        "subclass": "class"
    }

    first_batch = True

    for batch in pd.read_csv(input_file, chunksize=batch_size):

        descriptions=batch[input].fillna("").astype(str).agg(" ".join, axis=1).tolist()
        print('################## descriptions \n', descriptions)
        # results per company
        batch_results = {i: {} for i in range(len(descriptions))}


        # Section
        section_prompt = build_prompt(
            descriptions=descriptions,
            level_name="section",
            options=sections
        )

        print('################## section prompt \n', section_prompt)

        section_preds = llm_call(
            tokenizer=tokenizer, 
            llm=llm, 
            sampling_params=sampling_params,
            prompts=section_prompt,
            hierarchy=hierarchy, 
            level="section", 
            map_code_names=map_code_names)# e.g. "C"
        
        print('################ section peds \n', section_preds)


        prompts = []
        options_list = []

        # auto-descend to fill unambiguous levels
        for i, pred in enumerate(section_preds):
            final_code, final_level = auto_descend(pred, hierarchy)
            batch_results[i][final_level] = final_code

        # building prompts for next predictions
        for i, res in batch_results.items():
            # Skip if already at deepest level
            last_known_level = max(res.keys(), key=lambda k: map_hier_indx[k]) if res else None
            
            if last_known_level == "subclass":
                continue  # nothing left to predict

            # Determine next level to predict
            if last_known_level is None:
                next_level = "section"
                parent = None
                options = list(hierarchy[next_level].keys())
            else:
                last_parent = res[last_known_level]
                next_level_index = map_hier_indx[last_known_level] + 1
                next_level = levels[next_level_index]

                # Check that parent exists in hierarchy
                if last_parent not in hierarchy.get(last_known_level, {}):
                    print(f"{last_parent} not in hierarchy for level {last_known_level}")
                    continue

                options = hierarchy[last_known_level][last_parent]
                if not options:
                    continue  # nothing to predict

            # auto-descend if one child
            if len(options) == 1:
                auto_code, auto_level = auto_descend(options[0], hierarchy)
                batch_results[i][auto_level] = auto_code
                continue  # no prompt needed

            # build prompt for the next level
            buildt_prompt = build_prompt(
                descriptions=descriptions,
                level_name=next_level,
                options=options,
                parent=last_parent
            )
            prompts.extend(buildt_prompt)
            options_list.append(options)

            preds = llm_call(tokenizer=tokenizer, 
                             llm=llm, 
                             sampling_params=sampling_params,
                             prompts=prompts,
                             hierarchy=hierarchy, 
                             level=next_level, 
                             map_code_names=map_code_names
                             )
            

            

            # Validate prediction
            for pred, option in zip(preds, options_list):
                if set(pred).issubset(option):
                    print(f"Invalid prediction at {next_level}: {pred}, using parent {batch_results[i].get(next_level, '')}")
                    pred = batch_results[i].get(next_level, '')

            #if preds not in options_list:
            #    print(f"Invalid prediction at {next_level}: {pred}, using parent {batch_results[i].get(next_level, '')}")
            #    pred = batch_results[i].get(next_level, '')

            final_code = auto_descend(pred, hierarchy)
            batch_results[i][next_level] = final_code

        # Add predictions to batch DataFrame
        for i, res in batch_results.items():
            batch.at[i, f"pred_subclass"] = res.get("subclass", "")

        # Write to CSV
        batch.to_csv(
            output_file,
            mode="w" if first_batch else "a",
            header=first_batch,
            index=False
        )
        first_batch=False
    return batch


        
        
if __name__ == "__main__":
    #mp.set_start_method("spawn", force=True)
    model = LLM(args.model_name,
            tensor_parallel_size=args.num_gpus # bigger models may require more GPUs and higher tensor parallel size
            )

    tokenizer = model.get_tokenizer()
   
    SECTIONS = sections_df_hier(df_hier)
    map_sec = dict(zip(df_hier[df_hier['level']==2]["code"], df_hier[df_hier['level']==2]["parentCode"]))
    map_name = df_hier[['code', 'name']].set_index('code')['name'].to_dict()
    df_hier_hier = derive_hier_names(df=df_hier[df_hier['level']==5],  subclass_col='code', section_map=map_sec, map_name=map_name)
    HIERARCHY = build_parent_child_map(df_hier_hier)
    map_code_names = mapping_code_names(df=df_hier[df_hier['level']==5],  subclass_col='code', section_map=map_sec, map_name=map_name)
    

    df_with_res = run_classify_nace(tokenizer=tokenizer, llm=model, sections=SECTIONS, hierarchy=HIERARCHY, map_code_names=map_code_names)
    print(df_with_res['subclass'])
    res_sub, res_cl, res_gro, res_div = metrics_levels(target=df_with_res['nace_21_code'], pred=df_with_res['subclass']) # type: ignore
    

    with PdfPages(f"{RES_LM}train_hier_results_sub.pdf") as pdf:
        pdf.savefig(df_to_table(res_sub, "Subclass Results"))
        pdf.savefig(df_to_table(res_cl, "Class Results"))
        pdf.savefig(df_to_table(res_gro, "Group Results"))
        pdf.savefig(df_to_table(res_div, "Division Results"))