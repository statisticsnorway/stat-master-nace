# imported libraries
import fasttext


# Skipgram model, finetuned:
model = fasttext.train_supervised(input=f"{SAVE_PATH}/train_fasttext.txt", autotuneValidationFile=f"{SAVE_PATH}/val_fasttext.txt") # Hyperparameter tuning by using "autotuneValidationFile" parameter

#Saving the model
model.save_model(f"{SAVE_PATH}/model_nace.bin")

# using saved model
model = fasttext.load_model(f"{SAVE_PATH}/model_nace.bin")

labels, probs = model.predict(test_text) 

# hierarchical model

