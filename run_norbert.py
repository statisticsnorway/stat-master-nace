import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorWithPadding, TrainingArguments, EarlyStoppingCallback, get_scheduler, Trainer

from src.config import HIERARCHY_DATA, RANDOM_STATE,THREAD, DATA_LM_TR_TE, DATA_LM_TR_VAL_TE, RES_HIER_M, JSON_FILES, RES_LM
from src.utils.llm.dataset import NaceDataset 
from src.metrics import metrics_lm, metrics_levels

input_cols=["company_activity", "company_name"]
label_col='nace_21_code'

# Load the CSV file into a DataFrame

train=pd.read_csv(f'{DATA_LM_TR_TE}train.csv')
test=pd.read_csv(f'{DATA_LM_TR_TE}test.csv')

tokenizer = AutoTokenizer.from_pretrained("ltg/norbert3-base")

# datacollator from hugging face. This can be replaced with a custom collator function for the advanced method
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Dataloader that creates batches 
train_input = NaceDataset(df=train, input_cols=input_cols, label_col=label_col, tokenizer=tokenizer)

dataloader = DataLoader(train_input, batch_size=32, collate_fn=data_collator)

#initialising the model
model = AutoModelForMaskedLM.from_pretrained("ltg/norbert3-base", trust_remote_code=True)

"""
for batch in dataloader:
    outputs = model(**batch)
    loss = outputs.loss
output_text = torch.where(output_p.input_ids == mask_id, output_p.logits.argmax(-1), output_p.input_ids)
# should output: '[CLS] Nå ønsker de seg en ny bolig.[SEP]'
print(tokenizer.decode(output_text[0].tolist()))
"""


# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",           # Directory for saving model checkpoints
    evaluation_strategy="epoch",     # Evaluate at the end of each epoch
    learning_rate=5e-5,              # Start with a small learning rate
    per_device_train_batch_size=16,  # Batch size per GPU
    per_device_eval_batch_size=16,
    num_train_epochs=3,              # Number of epochs
    weight_decay=0.01,               # Regularization
    save_total_limit=2,              # Limit checkpoints to save space
    load_best_model_at_end=True,     # Automatically load the best checkpoint
    logging_dir="./logs",            # Directory for logs
    logging_steps=100,               # Log every 100 steps
    fp16=True,                        # Enable mixed precision for faster training
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

print(training_args)

#bringing all together
trainer = Trainer(
    model=model,                        # Pre-trained BERT model
    args=training_args,                 # Training arguments
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,        # Efficient batching
    compute_metrics=metrics_lm    # Custom metric
)

# Dynamic learning rate scheduler.
scheduler = get_scheduler( 
    "linear",
    optimizer=trainer.optimizer,
    num_warmup_steps=0,
    num_training_steps=len(trainer.train_dataset) * training_args.num_train_epochs
)

# adding early stopping to avoid overfitting
trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=2))

# Start training
trainer.train()

# predicting
test_input = NaceDataset(df=test, input_cols=input_cols, label_col=label_col, tokenizer=tokenizer)
predictions = trainer.predict(test_input)
predicted_labels = predictions.predictions.argmax(axis=-1)

# metrics
metrics_levels(target=test[label_col], pred=predicted_labels)
with PdfPages(f"{RES_LM}test_results_sub.pdf") as pdf:
    pdf.savefig(df_to_table(res_sub_te, "Subclass Results"))
    pdf.savefig(df_to_table(res_cl_te, "Class Results"))
    pdf.savefig(df_to_table(res_gro_te, "Group Results"))
    pdf.savefig(df_to_table(res_div_te, "Division Results"))


#TODO: use optuna for hyp param tuning?
#TODO: complete the coding
