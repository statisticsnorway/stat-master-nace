import argparse
import os
from src.config import DATA_LM_TR_VAL_TE
# using DATA_BR_TRAIN as validation set to tune hyperparameters.


def parse_args():   
    MODEL ="Qwen/Qwen2.5-7B-Instruct"

    argparser = argparse.ArgumentParser()
    argparser.add_argument("-vd", "--val_data_file", type=str, default=os.path.join(DATA_LM_TR_VAL_TE, 'val.csv'))
    argparser.add_argument("-td", "--test_data_file", type=str, default=os.path.join(DATA_LM_TR_VAL_TE, 'test.csv'))
    argparser.add_argument("-oh", "--output_file_hier", type=str, default="data_lm_preds_hier.csv")
    argparser.add_argument("-of", "--output_file_flat", type=str, default="data_lm_preds_flat.csv")
    argparser.add_argument("-cfh", "--checkpoint_file_hier", type=str, default="checkpoint_hier.json")
    argparser.add_argument("-cff", "--checkpoint_file_flat", type=str, default="checkpoint_flat.json")
    argparser.add_argument("-M", "--model_name", type=str, default=MODEL, choices=["meta-llama/Llama-3.2-3B-Instruct",
                                                                                   "meta-llama/Llama-3.1-8B-Instruct",
                                                                                   "Qwen/Qwen2.5-3B-Instruct",
                                                                                   "Qwen/Qwen2.5-7B-Instruct",
                                                                                   "Qwen/Qwen2.5-14B-Instruct",
                                                                                   "Qwen/Qwen2.5-32B-Instruct",                                                                                   
                                                                                   ])
    argparser.add_argument("--input_colm", type=str, nargs="+", default=['company_activity','company_name','company_purpose'],
                    help="Specify which input columns to train on.")    
    #argparser.add_argument("-gpu", "--num_gpus", type=int, default=1)
    argparser.add_argument("-bs", "--batch_size", type=int, default=64) #batch size 32   
    argparser.add_argument("--debug", action="store_true")
    return argparser.parse_args()
    
