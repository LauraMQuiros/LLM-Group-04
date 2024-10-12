from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Trainer, TrainingArguments, TrainerCallback
import json
import pandas as pd
import torch
import matplotlib.pyplot as plt

#Load in the base model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base", model_max_length=512)
model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")

#Load the cleaned data into a dataframe
df = pd.read_csv('data/squad_cleaned.csv')

#Split into train and validation set
train = pd.read_table('data/train.tsv')
val = pd.read_table('data/val.tsv')

# Convert context and questions to lists of strings
train_context = train['Context'].astype(str).tolist()
train_questions = train['Question'].astype(str).tolist()

val_context = val['Context'].astype(str).tolist()
val_questions = val['Question'].astype(str).tolist()

# Define a custom dataset class
class QuestionGenerationDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = tokenizer(labels, truncation=True, padding=True, max_length=max_length).input_ids

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create the training and validation datasets
train_dataset_questiongeneration = QuestionGenerationDataset(train_context, train_questions, tokenizer)
val_dataset_questiongeneration = QuestionGenerationDataset(val_context, val_questions, tokenizer)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results_questiongeneration',  # Output directory
    num_train_epochs=15,                        # Number of training epochs
    per_device_train_batch_size=16,             # Batch size for training
    save_steps=10_000,                          # Save checkpoint every 10,000 steps
    save_total_limit=2,                         # Only keep the last 2 checkpoints
    logging_dir='./logs',                       # Directory for storing logs
    logging_steps=300,                          # Log every 500 steps
    eval_steps=300,                             # Evaluate every 500 steps
    evaluation_strategy="steps",                # Evaluate at the end of each step
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
    train_dataset=train_dataset_questiongeneration,
    eval_dataset=val_dataset_questiongeneration,
    callbacks=[LossLoggerCallback],
)

# Train the model
trainer.train()

# Save the trained model
model.save_pretrained('./fine_tuned_t5_questiongeneration_valset')
tokenizer.save_pretrained('./fine_tuned_t5_questiongeneration_valset')

#Save loss data
loss_data = pd.DataFrame({
    'training_loss': training_loss,
    'validation_loss': validation_loss[:len(training_loss)]  # Ensure same length
})
loss_data.to_csv('loss_data.csv', index=False)

# Plot the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig('loss_plot.png')
plt.show()
