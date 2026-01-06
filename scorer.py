"""
Script to calculate dataset scores using the LLama.
"""

import argparse
import json
import re
import os
from pathlib import Path
import gzip
import zstandard as zstd
import io
from contextlib import contextmanager
import torch
import vllm
from vllm import LLM, SamplingParams
from src.parser import parse_args

args = parse_args()
PROJECT_ROOT = Path(__file__).resolve().parent

PROMPT_PATH = PROJECT_ROOT / args.prompt
PROMPT = PROMPT_PATH.read_text()

OUTPUT_DS_PATH = Path(args.output_dataset_dir)
if args.debug:
    OUTPUT_DS_PATH = PROJECT_ROOT / "data/debug_scored_dataset.jsonl"
output_file=OUTPUT_DS_PATH


model = LLM(args.model_name,
            tensor_parallel_size=args.num_gpus # bigger models may require more GPUs and higher tensor parallel size
            )

tokenizer = model.get_tokenizer()

sampling_params = SamplingParams(
        temperature=0.3,
        max_tokens=2048
    )


def build_prompt(description, level_name, options, parent=None):
    options_str = "\n".join(f"- {opt}" for opt in options)

    parent_str = f"\nParent category: {parent}" if parent else ""

    return f"""
    Company description:
    {description}

    Task:
    Select the most appropriate {level_name}.{parent_str}

    Possible {level_name}s:
    {options_str}

    Answer with exactly ONE code from the list above.
    """.strip()


def extract_class(output_text):
    # Removeing "assistant" prefix if present
    assistant_prefix = "assistant\n\n"
    if output_text.startswith(assistant_prefix):
        output_text = output_text[len(assistant_prefix):]

    # Looking for the score with flexible regex
    pattern =  pattern = r"""(?:Educational\s*score\s*:|
                  I\ would\ give\ an\ educational\ score\ of|
                  I\ would\ give\ it\ a\ total\ score\ of|
                  I\ would\ give\ it\ a\ score\ of|
                  I\ would\ award\ a\ score\ of|
                  I\ would\ assign\ a\ score\ of)
                  \s*([0-9]+(?:\.[0-9]+)?)"""

    match = re.search(pattern, output_text, re.IGNORECASE | re.VERBOSE)

    if not match:
        print(f"No score found in model output:\n{output_text}")
        return None
    return int(float(match.group(1)))


def call_llm(texts: list[str], tokenizer, llm, sampling_params, prompts):
    """
    texts: list of strings
    returns: list of (text, score)
    """




    prompts_tokenized = tokenizer.apply_chat_template(
        [
            {"role": "system", 
            "content": "You are an expert in SN2025 classification."}
            {"role": "user", "content": p}
        ] for p in prompts,

        tokenize=False,
        add_generation_prompt=True
        )
        

    # Runing batched inference
    outputs = llm.generate(prompts_tokenized, sampling_params)

    # Extracting scores
    results = []
    for text, out in zip(texts, outputs):
        raw = out.outputs[0].text
        score = extract_score(raw)
        results.append((text, score))
    return results

def format_output(id, text, score, gold_score=None):
    if score is None:
        return None
        
    data_res = {"id": id, "text": text, "score": score}

    if gold_score is not None:
        data_res["gold score"] = gold_score

    #print("----------- OUT PROMPT-----------")
    #print(data_res)
    return data_res

 
def batch_reader(source, batch_size=args.batch_size):
    batch = []
    for line in source:
        item = json.loads(line) if isinstance(line, str) else line
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch





############### this must be edited. use the hier-df dataframe #################
SECTIONS = [
    "A – Agriculture, forestry and fishing",
    "B – Mining and quarrying",
    "C – Manufacturing",
    "D – Electricity, gas, steam and air conditioning supply",
]


HIERARCHY = {
    # Section → Division
    "C": ["10", "11", "12"],

    # Division → Group
    "10": ["101", "102"],

    # Group → Class
    "101": ["1011", "1012"],

    # Class → Subclass
    "1011": ["10110", "10120"],
}
######## replace these with the code and name of code
#################################################################################


def run_classify_nace(description):
    results = {}

    # Section
    section_prompt = build_prompt(
        description=description,
        level_name="section",
        options=SECTIONS
    )

    section_output = llm_call(section_prompt)
    section_code = section_output.split()[0]  # e.g. "C"

    results["section"] = section_code
    current = section_code

    # Division → Subclass
    for level in ["division", "group", "class", "subclass"]:
        if current not in HIERARCHY:
            break

        options = HIERARCHY[current]

        prompt = build_prompt(
            description=description,
            level_name=level,
            options=options,
            parent=current
        )

        prediction = llm_call(prompt)

        # Validate
        if prediction not in options:
            print(f"Invalid prediction at {level}: {prediction}")
            break

        results[level] = prediction
        current = prediction

    return results



def run_classify_nace(data_file=args.base_dataset_dir, target_dataset=None, output_file=output_file, batch_size=args.batch_size):
    tokenizer = model.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=0.3,
        max_tokens=2048
    )

    if target_dataset is None:
        def infile_iterable():
            with open_file(data_file) as infile:
                yield from batch_reader(infile, batch_size=batch_size)
    else:
        def infile_iterable():
            yield from batch_reader(target_dataset, batch_size=batch_size)

    with open(output_file, "w", encoding="utf-8") as outfile:
        for batch in infile_iterable():
            ids = [item["id"] for item in batch]
            texts = [item["text"] for item in batch]

            if "score" in batch[1].keys():
                golds = [item.get("score") for item in batch]

            scored = call_llm(texts, tokenizer, model, sampling_params)

            if "score" in batch[1].keys():
                # Formating and writing JSON with gold labels
                for ex_id, (text, score), gold in zip(ids, scored, golds):
                    record = format_output(ex_id, text, score, gold)
                    if record is not None:
                        outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
                        outfile.flush()
            else:
                # Formating and writeing JSON without gold labels
                for ex_id, (text, score) in zip(ids, scored):
                    record = format_output(ex_id, text, score)
                    if record is not None:
                        outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
                        outfile.flush()

        
        
if __name__ == "__main__":
    classify_nace()