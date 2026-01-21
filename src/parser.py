import argparse
from src.config import DATA_BR_TRAIN, DATA_BR_TEST
# using DATA_BR_TRAIN as validation set to tune hyperparameters.


def parse_args():   
    MODEL = "meta-llama/Llama-3.1-8B-Instruct"

    argparser = argparse.ArgumentParser()
    argparser.add_argument("-vd", "--val_data_file", type=str, default=DATA_BR_TRAIN)
    argparser.add_argument("-td", "--test_data_file", type=str, default=DATA_BR_TEST)
    argparser.add_argument("-o", "--output_file", type=str, default="data/data_lm_preds.csv")
    argparser.add_argument("-p", "--prompt", type=str, default="src/scorer_prompt.txt")
    argparser.add_argument("-m", "--max_retries", type=int, default=2)
    argparser.add_argument("-M", "--model_name", type=str, default=MODEL)
    argparser.add_argument("-gpu", "--num_gpus", type=int, default=4)
    argparser.add_argument("-bs", "--batch_size", type=int, default=32)   
    argparser.add_argument("--debug", action="store_true")
    return argparser.parse_args()
    
