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


def extract_class(output_text):
    # Removeing "assistant" prefix if present
    assistant_prefix = "assistant\n\n"
    if output_text.startswith(assistant_prefix):
        output_text = output_text[len(assistant_prefix):]

    if output_text is None:
        print(f"No code found in model output:\n{output_text}")
        return None
    return output_text


def llm_call(tokenizer, llm, sampling_params, prompts):
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
        nace_code = extract_class(raw)
        results.append(nace_code)
        results_probs.append(probs)
        print(results_probs) #----------------- legge til sannsynlighetene i run functionen
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



SECTIONS = [
    "A – Agriculture, forestry and fishing",
    "B – Mining and quarrying",
    "C – Manufacturing",
    "D – Electricity, gas, steam and air conditioning supply",
]
"""


def sections_df_hier(df:pd.DataFrame):
    sections = df[df['level']==1][['code','name']].astype(str).agg(" - ".join, axis=1).tolist()
    return sections


"""
HIERARCHY = {
    # Section → Division
    "C": ["10", "11", "12"],

    # Division → Group
    "10": ["101", "102"],

    # Group → Class
    "101": ["1011", "1012"],

    # Class → Subclass
    "1011": ["10110", "10120"],
}"""


def derive_hier_names(df: pd.DataFrame, subclass_col:str, section_map, map_name):
    """  ["section", "division", "group", "class"] """
    df = df.copy()
    codes_div = df[subclass_col].str[:2]
    names_div = codes_div.map(map_name)
    df["division"] = codes_div + " - " + names_div

    codes_gr = df[subclass_col].str[:4]
    names_gr = codes_gr.map(map_name)
    df["group"] = codes_gr + " - " + names_gr

    codes_cls = df[subclass_col].str[:5]
    names_cls = df[subclass_col].map(map_name)
    df["class"] = codes_cls + " - " + names_cls

    df["section"] = df["division"].map(section_map)
    return df



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
    while code in hierarchy and len(hierarchy[code]) == 1:
        code = hierarchy[code][0]
    return code



def run_classify_nace(tokenizer, 
                      llm,
                      sections,
                      hierarchy,
                      input_file:str=args.val_data_file, 
                      input:str|list[str] = ["company_activity", "company_name"],
                      sampling_params=sampling_params, 
                      batch_size=args.batch_size, 
                      output_file:str=args.output_file,
                      ):
    results = {}
    first_batch = True

    for batch in pd.read_csv(input_file, chunksize=batch_size):

        descriptions=batch[input].fillna("").astype(str).agg(" ".join, axis=1).tolist()
        
        # results per company
        batch_results = {i: {} for i in range(len(descriptions))}


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
            prompts=section_prompt)# e.g. "C"
        
        ### ----------- might need extraction --------------

        for i, pred in enumerate(section_preds):
            final_code = auto_descend(pred, hierarchy)
            batch_results[i]["section"] = final_code

        results["section"] = section_preds
        current = section_preds

        # Division until Subclass
        for level in ["division", "group", "class", "subclass"]:
            # Prepare prompts only for companies that still need prediction
            indices_to_predict = [i for i, res in batch_results.items() if level not in res]

            if not indices_to_predict:
                continue

            prompts = []
            options_list = []

            for i in indices_to_predict:
                parent = batch_results[i].get({
                    "division": "section",
                    "group": "division",
                    "class": "group",
                    "subclass": "class"
                }[level])

                if parent not in hierarchy or len(hierarchy[parent]) == 0:
                    batch_results[i][level] = parent
                    print(f"{current} not in HIERARCHY")
                    continue
                
                prompts.append(build_prompt(
                    descriptions=descriptions,
                    level_name=level,
                    options=hierarchy[parent],
                    parent=current
                ))
                options_list.append(hierarchy[current])

            if not prompts:
                continue

            preds = llm_call(tokenizer=tokenizer, 
                                    llm=llm, 
                                    sampling_params=sampling_params,
                                    prompts=prompts)

            # Save predictions with auto-descend
            for j, i in enumerate(indices_to_predict):
                pred = preds[j]

                # Validate prediction
                if pred not in options_list[j]:
                    print(f"Invalid prediction at {level}: {pred}, using parent {batch_results[i].get(level, '')}")
                    pred = batch_results[i].get(level, '')

                final_code = auto_descend(pred, hierarchy)
                batch_results[i][level] = final_code

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
    map_sec = dict(zip(df_hier["code"], df_hier["parentCode"]))
    map_name = df_hier[['code', 'name']].set_index('code')['name'].to_dict()
    df_hier_hier = derive_hier_names(df=df_hier,  subclass_col='code', section_map=map_sec, map_name=map_name)
    HIERARCHY = build_parent_child_map(df_hier_hier)
    
    df_with_res = run_classify_nace(tokenizer=tokenizer, llm=model, sections=SECTIONS, hierarchy=HIERARCHY)
    print(df_with_res['subclass'])
    res_sub, res_cl, res_gro, res_div = metrics_levels(target=df_with_res['nace_21_code'], pred=df_with_res['subclass']) # type: ignore
    

    with PdfPages(f"{RES_LM}train_hier_results_sub.pdf") as pdf:
        pdf.savefig(df_to_table(res_sub, "Subclass Results"))
        pdf.savefig(df_to_table(res_cl, "Class Results"))
        pdf.savefig(df_to_table(res_gro, "Group Results"))
        pdf.savefig(df_to_table(res_div, "Division Results"))