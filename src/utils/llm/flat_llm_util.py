import pandas as pd
from pathlib import Path
import re
from matplotlib.backends.backend_pdf import PdfPages
import multiprocessing as mp
import math
from src.parser import parse_args

from src.config import RANDOM_STATE

seed = RANDOM_STATE

args = parse_args()
PROJECT_ROOT = Path(__file__).resolve().parent

def build_prompt(descriptions, options):
    prompts={}
    options_str = "\n".join(f"- {opt}" for opt in options)

    for indx in descriptions:
        prompts[indx] = (
            "Informasjon om bedriften:\n"
            f"{descriptions[indx].strip()}\n\n"
            f"Oppgave:\n"
            "Velg den NACE-underklassen som best beskriver bedriftens hovedaktivitet.\n\n"
            f"Gyldige NACE-klasser:\n{options_str}\n\n"
            "Regler:\n"
            "- Velg nøyaktig ÉN kode fra listen over og returner KUN koden (f.eks. 01.110).\n"
            "- Hvis ingen kategori passer godt, velg den som er nærmest.\n"
            "- Ikke inkluder forklaring, navn, punktum eller andre tegn.\n\n"
            "Svar:"
        )
    return prompts


def extract_class(output_text, subclasses_code, map_code_names):
    escaped_codes = [re.escape(code) for code in subclasses_code]
    #escaped_codes.append(re.escape("UKJENT"))
    pattern = r"\b(" + "|".join(escaped_codes) + r")\b"

    # Removeing "assistant" prefix if present
    assistant_prefix = "assistant\n\n"
    if output_text.startswith(assistant_prefix):
        output_text = output_text[len(assistant_prefix):]

    if output_text is None:
        print(f"No code found in model output:\n{output_text}", flush=True)
        return "UKJENT"
    
    if output_text not in subclasses_code:
        match = re.search(pattern, output_text)

        # if the output is code and name compines
        if output_text in map_code_names:
            output_text = map_code_names[output_text]
        #elif output_text == 'UKJENT':
        #    return 'UKJENT'
        elif match:
            output_text = match.group(1)
        else:
            print(f"####### {output_text} not in HIERARCHY", flush=True)
            return 'UKJENT'
    return output_text

def compute_code_prob(nace_code, logprob_list):
    """
    Returns probability of extracted code.
    If code is 'UKJENT', returns 0.
    """

    if nace_code == "UKJENT":
        return 0.0

    total_logprob = 0.0
    reconstructed = ""

    for token_dict in logprob_list:
        logprob_obj = list(token_dict.values())[0]
        token = logprob_obj.decoded_token

        # stop at special end token
        if token == "<|im_end|>":
            break

        reconstructed += token

        # Only accumulate logprob if token is part of the code
        if nace_code.startswith(reconstructed):
            total_logprob += logprob_obj.logprob

        # If reconstruction no longer matches code then stop
        elif not nace_code.startswith(reconstructed):
            break

    # If we never fully reconstructed the code → probability 0
    if reconstructed.strip() != nace_code:
        return 0.0

    return math.exp(total_logprob)


def llm_call(tokenizer, llm, sampling_params, prompts, subclasses_code, map_code_names):
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
            "content": "Du er en ekspert på SN2025-klassifisering."},
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

        print('--------- raw prompt ------------ \n', raw, flush=True)
        print('--------- raw probs ------------ \n', probs, flush=True)

        
        nace_code = extract_class(output_text=raw, subclasses_code=subclasses_code, map_code_names=map_code_names)
        nace_probs = compute_code_prob(nace_code=nace_code,logprob_list=probs)
        results[idx]=nace_code
        results_probs[idx]=nace_probs
        #print('############### result logits \n', results_probs, flush=True) #----------------- legge til sannsynlighetene i run functionen
    return results, results_probs


def mapping_code_names(df: pd.DataFrame, subclass_col:str, map_name:dict):
    """
    Docstring for mapping_code_names
    
    :param df: Dataframe with the subclass codes and code names
    :param subclass_col: Name of the subclass column in the dataframe
    :param map_name: Map the code to the name of the code
    """
    df = df.copy()

    df_subcls = pd.DataFrame({
        "name": df[subclass_col] + " - " + df[subclass_col].map(map_name),
        "code": df[subclass_col],
    })

    map_df = df_subcls.drop_duplicates().set_index("name")["code"].to_dict()
    return map_df



def validate_and_assign(
    preds,
    probs,
    batch_results,
):
    for idx in preds:
        pred=preds[idx]
        prob=probs[idx]
        # normalization
        if isinstance(pred, list):
            pred = pred[0]

        batch_results[idx]['subclass'] = pred
        batch_results[idx]['subclass_probs'] = prob
    return batch_results

