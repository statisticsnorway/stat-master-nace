import argparse
from src.config import RES_LM, DATA_LM_TR_VAL_TE
# using DATA_BR_TRAIN as validation set to tune hyperparameters.


def parse_args():   
    MODEL = "meta-llama/Llama-3.1-8B-Instruct"

    argparser = argparse.ArgumentParser()
    argparser.add_argument("-vd", "--val_data_file", type=str, default=f"{DATA_LM_TR_VAL_TE}val.csv")
    argparser.add_argument("-td", "--test_data_file", type=str, default=f"{DATA_LM_TR_VAL_TE}test.csv")
    argparser.add_argument("-oh", "--output_file_hier", type=str, default=f"{RES_LM}data_lm_preds_hier.csv")
    argparser.add_argument("-of", "--output_file_flat", type=str, default=f"{RES_LM}data_lm_preds_flat.csv")
    argparser.add_argument("-p", "--prompt", type=str, default="src/scorer_prompt.txt")
    argparser.add_argument("-m", "--max_retries", type=int, default=2)
    argparser.add_argument("-M", "--model_name", type=str, default=MODEL)
    argparser.add_argument("-gpu", "--num_gpus", type=int, default=4)
    argparser.add_argument("-bs", "--batch_size", type=int, default=2048) #batch size 32   
    argparser.add_argument("--debug", action="store_true")
    return argparser.parse_args()
    
