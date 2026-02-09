import pandas as pd
import re
import os
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
import multiprocessing as mp
from src.parser import parse_args

from sklearn.model_selection import train_test_split
from src.config import MODELS_FASTXT, DATA_LM_TR_TE, DATA_LM_TR_VAL_TE, RANDOM_STATE, DATA

seed = RANDOM_STATE

args = parse_args()
PROJECT_ROOT = Path(__file__).resolve().parent


def splitting_dataset(df:pd.DataFrame, statify_column:str, train_file:str, test_file:str, seed:int, val_file:str = False)->pd.DataFrame:
        
    # train vs test
    train, temp = train_test_split(df, test_size=0.4, random_state=seed, stratify=df[statify_column])
    #### stratified cross validation instead of validation set
    
    if val_file==False:
        test=temp
        train.to_csv(f"{DATA_LM_TR_TE}train.csv")
        test.to_csv(f"{DATA_LM_TR_TE}test.csv")
    
    else: 
        # test vs validation
        test, val = train_test_split(temp, test_size=0.5, random_state=seed, stratify=temp[statify_column])
        val.to_csv(f"{DATA_LM_TR_VAL_TE}val.csv")
        train.to_csv(f"{DATA_LM_TR_VAL_TE}train.csv")
        test.to_csv(f"{DATA_LM_TR_VAL_TE}test.csv")
    
    return (train, val, test) if val_file else (train, test)
    

def build_prompt(descriptions, next_level, meta, map_code_name):
    prompts={}
    
    for indx in descriptions:
        if indx not in meta:
            print(f'meta does not exist for {indx}')
            continue
        parent, options = meta[indx]

        options_str = "\n".join(f"- {map_code_name.get(opt, '')}" for opt in options)
        parent_str = f"\nOverordnet kategori: {map_code_name.get(parent, '')}" if parent else ""
        
        prompts[indx]=(
        f"Bedriftsbeskrivelse:\n{descriptions[indx].strip()}\n\n"
        f"Oppgave:\nVelg den mest passende '{next_level}' klassen.{parent_str}\n\n"
        f"Mulige {next_level} klasser:\n"
        f"{options_str}\n\n"
        "Svar mednøyaktig ÉN kode fra listen ovenfor og velg KUN koden.\n"
        "Ikke inkluder navn, forklaring eller andre tegn.")
    return prompts


def extract_class(output_text, hierarchy_code, current_level, parent):
    " hierarchy is a hierarchy of allowed codes on that level. "

    if output_text is None:
        print(f"No code found in model output:\n{output_text}", flush=True)
        return None

    # Removeing "assistant" prefix if present
    assistant_prefix = "assistant\n\n"
    if output_text.startswith(assistant_prefix):
        output_text = output_text[len(assistant_prefix):]

    # extract matches   
    if output_text in hierarchy_code[current_level]:
        return output_text
  
    #regex extraction
    escaped_codes = [re.escape(code) for code in hierarchy_code[current_level][parent]]
    pattern = r"\b(" + "|".join(escaped_codes) + r")\b"
    match = re.search(pattern, output_text)
    if match:
        output_text = match.group(1)
        return output_text
    # total failure
    print(
        f"Failed to extract valid code at level '{current_level}' from output: '{output_text}'",flush=True)
        
    return None


def llm_call_fake(tokenizer, llm, sampling_params, prompts, hierarchy_code, current_level, meta):
    """
    Fake LLM call that deterministically returns a valid option
    for each prompt, keyed by the original batch index.
    """

    prompt_ids = list(prompts.keys())
    prompt_texts = list(prompts.values())

    results = {}

    for idx, p in zip(prompt_ids, prompt_texts):
        options = hierarchy_code[current_level]

        # Pick first valid option for reproducibility
        if isinstance(options, dict):
            results[idx] = list(options.keys())[0]
        else:
            results[idx] = options[0]

    return results



def llm_call(tokenizer, llm, sampling_params, prompts, hierarchy_code, current_level, meta):
    """
    texts: list of strings
    returns: list of (text, score)
    """
    prompt_ids = list(prompts.keys())
    prompt_texts = list(prompts.values())

    prompts_tokenized = tokenizer.apply_chat_template(
        [
            [
            {"role": "system", 
            "content": "Du er en ekspert på SN2025-klassifisering."},
            {"role": "user", "content": p}
            ] for p in prompt_texts
        ],
        tokenize=False,
        add_generation_prompt=True
        )
    
    # Runing batched inference
    outputs = llm.generate(prompts_tokenized, sampling_params)

    results = {}
    results_probs = {}
    for idx, out in zip(prompt_ids, outputs):
        parent, children = meta[idx]
        raw = out.outputs[0].text
        probs = out.outputs[0].logprobs
        nace_code = extract_class(output_text=raw, 
                                  hierarchy_code=hierarchy_code, 
                                  current_level=current_level, 
                                  parent=parent,
                                  )
        results[idx]=nace_code
        results_probs[idx]=probs
        #print('############### result logits \n', results_probs, flush=True) #----------------- legge til sannsynlighetene i run functionen
    return results


def sections_df_hier(df:pd.DataFrame):
    sections = df[df['level']==1][['code','name']].astype(str).agg(" - ".join, axis=1).tolist()
    return sections


def derive_hier_names(df: pd.DataFrame, subclass_col:str, section_map, map_name):
    """  ["section", "division", "group", "class"] 
    Expanding df to include all levels as separate columns 
    """
    
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
    names_cls = codes_cls.map(map_name)
    df["class"] = codes_cls + " - " + names_cls

    codes_subcls = df[subclass_col]
    names_subcls = df[subclass_col].map(map_name)
    df["subclass"] = codes_subcls + " - " + names_subcls
    return df


def derive_hier(df: pd.DataFrame, subclass_col:str, section_map, map_name):
    """  ["section", "division", "group", "class"] 
    Expanding df to include all levels as separate columns 
    """
    
    df = df.copy()
    codes_div = df[subclass_col].str[:2]
    df["division"] = codes_div 

    codes_sec = codes_div.map(section_map) 
    df["section"] = codes_sec

    codes_gr = df[subclass_col].str[:4]
    df["group"] = codes_gr 

    codes_cls = df[subclass_col].str[:5]
    df["class"] = codes_cls 

    codes_subcls = df[subclass_col]
    df["subclass"] = codes_subcls
    return df



def mapping_code_names(df: pd.DataFrame, subclass_col:str, section_map:dict, map_name:dict):
    "  ['section', 'division', 'group', 'class', 'subclass'] "
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
    df_subcls = pd.DataFrame({
        "code": df[subclass_col],
        "name": df[subclass_col] + " - " + df[subclass_col].map(map_name)
    })

    map_df = pd.concat([df_div, df_sec, df_gr, df_cls, df_subcls], ignore_index=True).drop_duplicates().set_index("code")["name"].to_dict()
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

    all_parent_child['root'] = list(all_parent_child['section'].keys())
    return all_parent_child

def auto_descend(parent, hierarchy_code, next_level):
    """
    Walk down hierarchy automatically
    while only ONE child exists.
    """
    child_level = {
    "root":"section",
    "section":"division",
    "division": "group",
    "group": "class",
    "class" :"subclass"
    }            
    level = next_level

    while level in child_level:
        # no children in this level = stop
        if parent not in hierarchy_code.get(level, {}):
            return parent, level
        
        children = hierarchy_code[level][parent]

        # more than one child = stop
        if len(children) != 1:
            return parent, level
        
        # exactky one child = decend
        parent = children[0]
        level = child_level[level]
        
    return parent, level


def companies_at_level(batch_results, current_level, map_hier_indx):
    """
    Gather all the company indexes from this batch that are at the same level to one list.
    
    :param batch_results: results of predictions at different hierarchies for the batch. 
                          The batch includes companies previous predictions.
    :param level: the last level that has been predicted
    :param map_hier_indx: mapping the level to its corresponding index
    """
    companies = []
    for idx, res in batch_results.items():
        if current_level == 'root':
            companies.append(idx)
        # rest of the levels
        if not res:
            continue

        deepest_level = max(res.keys(), key=lambda k: map_hier_indx[k])
        print('deepest_level_indx ', deepest_level)
        print('current_level_indx ', current_level)
        if deepest_level == current_level:
            companies.append(idx)
    return companies


def prepare_level_prompts(
    companies,
    batch_results,
    hierarchy_code,
    current_level,
    next_level,
    descriptions,
    map_code_name
):
    """
    Build prompt for all the companies in the batch at a specific level.
    
    :param companies: company index
    :param batch_results: results of predictions at different hierarchies for the batch. 
                          The batch includes companies previous predictions.
    :param hierarchy: the dictionary of hierarchies with parent as key and a list of children as value
    :param current_level: The last level that has been predicted
    :param next_level: The level that is being predicted
    :param descriptions: company descriptions and names
    """
    meta = {}
    selected_descp={}
    for company in companies:

        # section level
        if current_level == 'root':
            parent=None
            children = hierarchy_code[current_level]
            selected_descp[company]=descriptions[company]
            meta[company]=(parent, children)

        else:
            if current_level not in batch_results[company]:
                continue
            
            # lower levels (division and down)
            parent = batch_results[company][current_level]

            if parent not in hierarchy_code.get(current_level, {}):
                batch_results[company][next_level] = parent
                continue

            children = hierarchy_code[current_level][parent]

            if len(children) == 1:
                auto_code, auto_level = auto_descend(parent=parent, 
                                                     hierarchy_code=hierarchy_code, 
                                                     current_level=current_level)
                batch_results[company][auto_level] = auto_code
                children = hierarchy_code[auto_level][auto_code]
                continue
            
            selected_descp[company]=descriptions[company]
            meta[company]=(parent, children)

    prompts = build_prompt(
        descriptions=selected_descp,
        next_level=next_level,
        meta=meta,
        map_code_name=map_code_name,
    )
    return prompts, meta


def validate_and_assign(
    preds,
    meta,
    batch_results,
    hierarchy_code,
    current_level,
    next_level
):
    for idx, pred in preds.items():

        if isinstance(pred, list):
            print('list pred', pred)
            pred = pred[0]

        parent, children = meta[idx]

        if isinstance(parent, list):
            print('parent list', parent)
            parent = parent[0]

        if pred in children:
            label = pred
        else:
            label = parent

        if current_level != 'class':
            final_code, final_level = auto_descend(
                parent=label,
                hierarchy_code=hierarchy_code,
                next_level=next_level
            )
        else:
            final_code, final_level = label, next_level

        if not isinstance(final_level, str):
            raise ValueError(f"Invalid final_level: {final_level}")

        batch_results[idx][final_level] = final_code

    return batch_results




if __name__=='__main__':
    df = pd.read_csv(f"{DATA}data_prep_lm.csv", dtype={'company_activity':str,'company_name':str,'division':str, 'group':str, 'class':str, 'nace_21_code':str,'nace_21_description_nb':str}, keep_default_na=False, na_values=[]).fillna("")
    # Getting the train, val and test sets for the LM model
    splitting_dataset(df=df, statify_column='nace_21_code', train_file='train', test_file='test', seed=seed, val_file=False)
    splitting_dataset(df=df, statify_column='nace_21_code', train_file='train', test_file='test', seed=seed, val_file=True)
