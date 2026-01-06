import argparse
import torch

def parse_args():   
    MODEL = "meta-llama/Llama-3.1-8B-Instruct"
    data_file =  '/appl/local/openeurollm/training/catalogue/hplt/3.0/sorted/nob_Latn/10_1.jsonl.zst' 

    argparser = argparse.ArgumentParser()
    argparser.add_argument("-b", "--base_dataset_dir", type=str, default=data_file)
    argparser.add_argument("-v", "--val_dataset_dir", type=str, default="NbAiLab/nb-fineweb2-edu-bokmaal")
    argparser.add_argument("-o", "--output_dataset_dir", type=str, default="data/scored_dataset.jsonl")
    argparser.add_argument("-ov", "--output_dataset_dir_val", type=str, default="data/scored_dataset_val.jsonl")
    argparser.add_argument("-p", "--prompt", type=str, default="src/scorer_prompt.txt")
    argparser.add_argument("-m", "--max_retries", type=int, default=2)
    argparser.add_argument("-M", "--model_name", type=str, default=MODEL)
    argparser.add_argument("-gpu", "--num_gpus", type=int, default=1)
    argparser.add_argument("-bs", "--batch_size", type=int, default=32)   
    argparser.add_argument("--debug", action="store_true")
    return argparser.parse_args()
    
