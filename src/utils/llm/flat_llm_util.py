import pandas as pd
import re
import os
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from vllm import LLM, SamplingParams
import multiprocessing as mp
from src.parser import parse_args

from src.config import RANDOM_STATE

seed = RANDOM_STATE

args = parse_args()
PROJECT_ROOT = Path(__file__).resolve().parent


def build_prompt(descriptions, options):
    prompts={}
    options_str = "\n".join(f"- {opt}" for opt in options)
    
    for indx in descriptions:
        prompts[indx]=(
        f"Company description:\n{descriptions[indx].strip()}\n\n"
        "Task:\nSelect the most appropriate subclass.\n\n"
        "Possible subclasses:\n"
        f"{options_str}\n\n"
        "Answer with exactly ONE code from the list above.")
    return prompts



def extract_class(output_text, subclasses, map_code_names):
    # Removeing "assistant" prefix if present
    assistant_prefix = "assistant\n\n"
    if output_text.startswith(assistant_prefix):
        output_text = output_text[len(assistant_prefix):]

    if output_text is None:
        ##print(f"No code found in model output:\n{output_text}", flush=True)
        return f"No code found in model output"
    
    if output_text not in subclasses:
        if output_text in map_code_names:
            output_text = map_code_names[output_text]
        else:
            ##print(f"{output_text} not in HIERARCHY", flush=True)
            return None
        
    return output_text

def llm_call_fake(tokenizer, llm, sampling_params, prompts, subclasses, map_code_names):
    """
    Fake LLM call that deterministically returns a valid option
    for each prompt, keyed by the original batch index.
    """

    prompt_ids = list(prompts.keys())
    prompt_texts = list(prompts.values())

    results = {}

    for idx, p in zip(prompt_ids, prompt_texts):
        options = subclasses

        # Pick first valid option for reproducibility
        if isinstance(options, dict):
            results[idx] = list(options.keys())[-1]
        else:
            results[idx] = options[0]

    return results

def llm_call(tokenizer, llm, sampling_params, prompts, subclasses, map_code_names):
    """
    Docstring for llm_call
    
    :param tokenizer: the tokenizer corresponding to the model
    :param llm: the large language model
    :param sampling_params: Hyper parameters
    :param prompts: the company name and description prompts
    :param hierarchy: the dictionary of hierarchies with parent as key and a list of children as value
    :param subclass_col_name: Name of the subclass column in the dataframe
    :param map_code_names: dictionary mapping the codes to its corresponding names
    """
   
    prompt_ids = list(prompts.keys())
    prompt_texts = list(prompts.values())

    prompts_tokenized = tokenizer.apply_chat_template(
        [
            [
            {"role": "system", 
            "content": "You are an expert in SN2025 classification."},
            {"role": "user", "content": p}
            ] for p in prompt_texts
        ],
        tokenize=False,
        add_generation_prompt=True
        )
        

    # Runing batched inference
    outputs = llm.generate(prompts_tokenized, sampling_params)

    # Extracting scores
    results = {}
    results_probs = {}
    for idx, out in zip(prompt_ids, outputs):
        raw = out.outputs[0].text
        probs = out.outputs[0].logprobs
        #print('--------- raw prompt ------------ \n', raw, flush=True)
        nace_code = extract_class(output_text=raw, subclasses=subclasses, map_code_names=map_code_names)
        results[idx]=nace_code
        results_probs[idx]=probs
        #print('############### result logits \n', results_probs, flush=True) #----------------- legge til sannsynlighetene i run functionen
    return results


def mapping_code_names(df: pd.DataFrame, subclass_col:str, map_name:dict):
    """
    Docstring for mapping_code_names
    
    :param df: Dataframe with the subclass codes and code names
    :param subclass_col: Name of the subclass column in the dataframe
    :param map_name: Map the code to the name of the code
    """
    df = df.copy()

    df_subcls = pd.DataFrame({
        "code": df[subclass_col],
        "name": df[subclass_col] + " - " + df[subclass_col].map(map_name)
    })

    map_df = df_subcls.drop_duplicates().set_index("code")["name"].to_dict()
    return map_df



def validate_and_assign(
    preds,
    batch_results,
):
    for idx in preds:
        pred=preds[idx]
        # normalization
        if isinstance(pred, list):
            pred = pred[0]

        batch_results[idx]['subclass'] = pred
    return batch_results

