import pandas as pd
from pathlib import Path
import re
from matplotlib.backends.backend_pdf import PdfPages
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
        f"Bedriftsbeskrivelse og navn:\n{descriptions[indx].strip()}\n\n"
        "Oppgave:\nVelg den NACE-underklassen som passer best.\n\n"
        "Gyldige NACE-klasser:\n"
        f"{options_str}\n\n"    
        "Regler:\n"
        "- Du må velge nøyaktig ÉN kode fra listen over og velg KUN koden (for eksempel: 01.110).\n"
        #"- Hvis teksten ikke inneholder tilstrekkelig informasjon til å avgjøre riktig klasse, svar: UKJENT.\n"
        "- Svaret skal kun bestå av selve koden \n" # eller ordet UKJENT med blokkbokstaver.\n"
        "- Ikke inkluder navn, forklaring, punktum eller andre tegn.\n\n"

        "Svar:")
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


def llm_call_fake(tokenizer, llm, sampling_params, prompts, subclasses_code, map_code_names):
    """
    Fake LLM call that deterministically returns a valid option
    for each prompt, keyed by the original batch index.
    """

    prompt_ids = list(prompts.keys())
    prompt_texts = list(prompts.values())

    results = {}

    for idx, p in zip(prompt_ids, prompt_texts):
        options = subclasses_code

        # Pick first valid option for reproducibility
        if isinstance(options, dict):
            results[idx] = list(options.keys())[-1]
        else:
            results[idx] = options[0]

    return results

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
        nace_code = extract_class(output_text=raw, subclasses_code=subclasses_code, map_code_names=map_code_names)
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
        "name": df[subclass_col] + " - " + df[subclass_col].map(map_name),
        "code": df[subclass_col],
    })

    map_df = df_subcls.drop_duplicates().set_index("name")["code"].to_dict()
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

