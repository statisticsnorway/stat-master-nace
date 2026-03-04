# script for calculating the scores for the flat and hierarchical llm model

from src.config import HIERARCHY_DATA, RES_LM
from src.metrics import metrics_levels


# ===============================================
# reading the csv files of llm classified data
# ===============================================
     

# ===============================
# metrics
# ===============================
df_results_test = metrics(test_labels, pred_labels_test)
print('macro_f1 \n', df_results_test)
macro_f1 = df_results_test.loc['macro', 'f1']
weighted_f1 = df_results_test.loc['weighted', 'f1']
brier = df_results_test.loc['score', 'brier_score']
Hf1 = df_results_test.loc['score', 'HF1']

macro_f1_list.append(macro_f1)
weighted_f1_list.append(weighted_f1) 
brier_list.append(brier)
Hf1_list.append(Hf1)


res_sub_test, res_cl_test, res_gro_test, res_div_test, res_sec_test = metrics_levels(target=test_labels, pred=pred_labels_test)
with PdfPages(os.path.join(RES_HIER_FASTXT,f"test_hier_results.pdf")) as pdf:
    pdf.savefig(df_to_table(res_sub_test, "Subclass Results"))
    pdf.savefig(df_to_table(res_cl_test, "Class Results"))
    pdf.savefig(df_to_table(res_gro_test, "Group Results"))
    pdf.savefig(df_to_table(res_div_test, "Division Results"))
    pdf.savefig(df_to_table(res_sec_test, "Section Results"))

