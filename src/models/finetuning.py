from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments, TrainerCallback, DataCollatorForLanguageModeling
import json
import os
import sys
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
import pickle

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
sys.path.append(project_root)
from src.data_processing.Formality_Transfer_Dataset import FormalityTransferDataset

# Load model directly
model_id = "gpt2-medium"
tokenizer = AutoTokenizer.from_pretrained(model_id)

tokenizer.add_special_tokens({
    'eos_token': '<|endoftext|>',  # Custom EOS token
    'bos_token': '<|startoftext|>',  # Custom BOS token
    'pad_token': '<|pad|>'  # Padding token
})
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
model.resize_token_embeddings(len(tokenizer))

tokenizer.add_tokens(['[Formal]', '[Informal]'])
model.resize_token_embeddings(len(tokenizer))
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
train_path = os.path.join(os.getcwd(), 'data/processed/entertainment_train_dataset.pkl')
tune_path = os.path.join(os.getcwd(), 'data/processed/entertainment_tune_dataset.pkl')

# Load datasets
with open(tune_path, 'rb') as f:
    tune: FormalityTransferDataset = pickle.load(f)

with open(train_path, 'rb') as f:
    train: FormalityTransferDataset = pickle.load(f)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results_formalitytransfer',  # Output directory
    num_train_epochs=4,                        # Number of training epochs
    per_device_train_batch_size=8,             # Batch size for training
    save_steps=2500,                          # Save checkpoint every 2500 steps
    save_total_limit=2,                         # Only keep the last 2 checkpoints
    logging_dir='./logs',                       # Directory for storing logs
    logging_steps=300,                          # Log every 300 steps
    eval_steps=300,                             # Evaluate every 300 steps
    evaluation_strategy="epochs",                # Evaluate at the end of each epoch
)

# Initialize lists to store loss values
training_loss = []
validation_loss = []

# Define a custom callback to track losses
class LossLoggerCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if 'loss' in logs:
                training_loss.append(logs['loss'])
            if 'eval_loss' in logs:
                validation_loss.append(logs['eval_loss'])

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train,
    eval_dataset=tune,
    callbacks=[LossLoggerCallback],
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Save the trained model
model.save_pretrained(os.path.join(os.getcwd(), 'src/models/fulldataset_FT_model'))
tokenizer.save_pretrained(os.path.join(os.getcwd(), 'src/models/fulldataset_FT_model'))

#Save loss data
loss_data = pd.DataFrame({
    'training_loss': training_loss,
    'validation_loss': validation_loss[:len(training_loss)]  # Ensure same length
})
loss_data.to_csv(os.path.join(os.getcwd(), 'utils/loss_data.png', index=False)

# Plot the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig(os.path.join(os.getcwd(), 'images/loss_plot.png'))
plt.show()
