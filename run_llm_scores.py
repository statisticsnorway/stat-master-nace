# script for calculating the scores for the flat and hierarchical llm model
import pandas as pd
import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from src.config import HIERARCHY_DATA, RES_LM
from src.metrics import metrics_levels, df_to_table

# ===============================================
# reading the csv files of llm classified data
# ===============================================
dtype_map = {
    'company_activity': str,
    'company_name': str,
    'company_purpose': str,
    'nace_21_code': str,
    'division':str, 
    'group':str, 
    'class':str,
    'nace_21_description_nb':str,
    'pred_subclass':str,
}

flat_qwen3=pd.read_csv(os.path.join(RES_LM, f"data_lm_preds_flat_qwen3_ap.csv"), dtype=dtype_map, keep_default_na=False, na_values=[]).fillna("").set_index('orgnr')
flat_qwen7=pd.read_csv(os.path.join(RES_LM, f"data_lm_preds_flat_qwen7_ap.csv"), dtype=dtype_map, keep_default_na=False, na_values=[]).fillna("").set_index('orgnr')
flat_qwen14=pd.read_csv(os.path.join(RES_LM, f"data_lm_preds_flat_qwen14_ap.csv"), dtype=dtype_map, keep_default_na=False, na_values=[]).fillna("").set_index('orgnr')

hier_qwen3=pd.read_csv(os.path.join(RES_LM, f"data_lm_preds_hier_qwen3_ap.csv"), dtype=dtype_map, keep_default_na=False, na_values=[]).fillna("").set_index('orgnr')
hier_qwen7=pd.read_csv(os.path.join(RES_LM, f"data_lm_preds_hier_qwen7_ap.csv"), dtype=dtype_map, keep_default_na=False, na_values=[]).fillna("").set_index('orgnr')
hier_qwen14=pd.read_csv(os.path.join(RES_LM, f"data_lm_preds_hier_qwen14_ap.csv"), dtype=dtype_map, keep_default_na=False, na_values=[]).fillna("").set_index('orgnr')

a_cols=pd.read_csv(os.path.join(RES_LM, f"data_lm_preds_flat_conf_a_qwen3.csv"), dtype=dtype_map, keep_default_na=False, na_values=[]).fillna("").set_index('orgnr')
an_cols=pd.read_csv(os.path.join(RES_LM, f"data_lm_preds_flat_conf_an_qwen3.csv"), dtype=dtype_map, keep_default_na=False, na_values=[]).fillna("").set_index('orgnr')
ap_cols=pd.read_csv(os.path.join(RES_LM, f"data_lm_preds_flat_conf_ap_qwen3.csv"), dtype=dtype_map, keep_default_na=False, na_values=[]).fillna("").set_index('orgnr')
anp_cols=pd.read_csv(os.path.join(RES_LM, f"data_lm_preds_flat_conf_anp_qwen3.csv"), dtype=dtype_map, keep_default_na=False, na_values=[]).fillna("").set_index('orgnr')


# ===============================
# metrics
# ===============================

df_list_cols=[
    a_cols,
    an_cols,
    ap_cols,
    anp_cols
]
df_name_list_cols = [
    'a_cols',
    'an_cols',
    'ap_cols',
    'anp_cols'
]

df_list = [
    flat_qwen3, 
    flat_qwen7, 
    flat_qwen14, 
    hier_qwen3, 
    hier_qwen7, 
    hier_qwen14
    ]



df_name_list = [
    'flat_qwen3_ap', 
    'flat_qwen7_ap', 
    'flat_qwen14_ap', 
    'hier_qwen3_ap', 
    'hier_qwen7_ap', 
    'hier_qwen14_ap'
    ]
    
for df, name in zip(df_list_cols,df_name_list_cols):
    test_labels=df['nace_21_code']
    pred_labels_test=df['pred_subclass']

    res_sub_test, res_cl_test, res_gro_test, res_div_test, res_sec_test = metrics_levels(target=test_labels, pred=pred_labels_test)
    with PdfPages(os.path.join(RES_LM,f"{name}_results.pdf")) as pdf:
        for res, title in [
            (res_sub_test, "Subclass Results"),
            (res_cl_test, "Class Results"),
            (res_gro_test, "Group Results"),
            (res_div_test, "Division Results"),
            (res_sec_test, "Section Results"),
        ]:
            fig = df_to_table(res, title)  # get figure
            pdf.savefig(fig)
            plt.close(fig) 

