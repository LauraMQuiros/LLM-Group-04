import peft
import os
import sys
from peft import get_peft_model, LoraConfig, PeftModel, PeftConfig
import torch
import pickle
from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments, GPT2Tokenizer
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the project root to sys.path (assuming src is in the root directory)
project_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
sys.path.append(project_root)
from src.data_processing.FormalityTransferDataset import FormalityTransferDataset

MODEL_CHECKPOINT = 'gpt2-medium'
PREFIX_FORMAL = '[Formal]'
PREFIX_INFORMAL = '[Informal]'

# paths
test_path = os.path.join(os.getcwd(), 'data/processed/entertainment_test_dataset_processed.pkl')
train_path = os.path.join(os.getcwd(), 'data/processed/entertainment_train_dataset_processed.pkl')
tune_path = os.path.join(os.getcwd(), 'data/processed/entertainment_tune_dataset_processed.pkl')
tokeniser_path = os.path.join(os.getcwd(), 'src/models/tokenizer/tokenizer.pkl')
sys.path.append(os.path.join(os.getcwd(), 'src/data_processing'))
#print(sys.path)

# Load datasets
with open(test_path, 'rb') as f:
    test : FormalityTransferDataset = pickle.load(f)
with open(train_path, 'rb') as f:
    train : FormalityTransferDataset = pickle.load(f)
with open(tune_path, 'rb') as f:
    tune : FormalityTransferDataset = pickle.load(f)
with open(tokeniser_path, 'rb') as f:
    tokenizer : GPT2Tokenizer = pickle.load(f)

tokenized_dataset = {
    'train': train,
    'test': test,
    'tune': tune
}

small_dataset = {
    'train': train.get_slice(0, 100),
    'test': test.get_slice(0, 100),
    'tune': tune.get_slice(0, 100)
}

print(f"The lengths of each of the small dataset sections are: {len(small_dataset['train'])} for the train, "
      f"{len(small_dataset['test'])} for the test and {len(small_dataset['tune'])} for the tune.")

# Load and resize the model to accommodate new tokens
model = AutoModelForCausalLM.from_pretrained(MODEL_CHECKPOINT, device_map="auto")
model.resize_token_embeddings(len(tokenizer))

# Define LoRA Config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Add LoRA adaptor
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.to('cpu') # if GPU is available later on, change to 'cuda'

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer,
    mlm=False,
    #pad_to_multiple_of=8
)

# Training arguments NOTE: Add GPU later !!
training_args = TrainingArguments(
    output_dir="logs",
    per_device_train_batch_size=1,  # Lowered for memory
    per_device_eval_batch_size=1,   # Lowered for memory
    learning_rate=1e-5,
    num_train_epochs=3,
    logging_dir="logs/training",
    gradient_accumulation_steps=8,  # Adjust based on your needs
    logging_steps=100,
    save_strategy="epoch",
    eval_strategy="steps",
    eval_steps=500,
    report_to="tensorboard",
    fp16=False  # Keep as False on MPS
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=small_dataset["train"],
    eval_dataset=small_dataset["tune"]
)

# Disable cache for training
model.config.use_cache = False

# Train the model
trainer.train()

# Save our LoRA model & tokenizer results
trainer_model_id="models/trainer"
tokenizer_model_id="models/tokenizer"
trainer.model.save_pretrained(trainer_model_id)
tokenizer.save_pretrained(tokenizer_model_id)
# if you want to save the base model to call
trainer.model.base_model.save_pretrained(trainer_model_id)

# Load peft config for pre-trained checkpoint etc.
config = PeftConfig.from_pretrained(trainer_model_id)

# Load base LLM model and tokenizer (on CPU)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map={"": "cpu"})
tokenizer = GPT2Tokenizer.from_pretrained(config.base_model_name_or_path)

# Load the LoRA model (on CPU)
model = PeftModel.from_pretrained(model, trainer_model_id, device_map={"": "cpu"})
model.eval()

print("Peft model loaded")

# Use the first sample of the test set
try:
    sample = small_dataset['test'][0]
except KeyError as e:
    print(f"KeyError: {e} - Ensure 'test' exists in the dataset.")
except IndexError as e:
    print(f"IndexError: {e} - Ensure there are samples in 'test'.")

# Move the input_ids to CPU (remove .cuda() since it's not needed)
input_ids = tokenizer(sample["dialogue"], return_tensors="pt", truncation=True).input_ids

# Generate outputs
outputs = model.generate(input_ids=input_ids, max_new_tokens=10, do_sample=True, top_p=0.9)

# Print the input sentence and the generated summary
print(f"input sentence: {sample['dialogue']}\n{'---'* 20}")
print(f"summary:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]}")
