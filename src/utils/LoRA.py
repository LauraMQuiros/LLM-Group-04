import peft
import os
import sys
from peft import get_peft_model, LoraConfig, PeftModel, PeftConfig
import torch
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the project root to sys.path (assuming src is in the root directory)
project_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
sys.path.append(project_root)
from src.data_processing.FormalityTransferDataset import FormalityTransferDataset

# paths
test_path = os.path.join(os.getcwd(), 'data/processed/test_dataset_processed.pkl')
train_path = os.path.join(os.getcwd(), 'data/processed/train_dataset_processed.pkl')
tune_path = os.path.join(os.getcwd(), 'data/processed/tune_dataset_processed.pkl')
sys.path.append(os.path.join(os.getcwd(), 'src/data_processing'))
#print(sys.path)

# Load datasets
with open(test_path, 'rb') as f:
    test : FormalityTransferDataset = pickle.load(f)
with open(train_path, 'rb') as f:
    train : FormalityTransferDataset = pickle.load(f)
with open(tune_path, 'rb') as f:
    tune : FormalityTransferDataset = pickle.load(f)


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

model_id = "gpt2-medium"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Add special tokens
tokenizer.add_special_tokens({
    'pad_token': '[PAD]',
    'bos_token': '[FORMAL]',
    'eos_token': '[INFORMAL]'
})

# Load and resize the model to accommodate new tokens
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
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
peft_model_id="path_to_trained_model"
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)
# if you want to save the base model to call
trainer.model.base_model.save_pretrained(peft_model_id)

# Load peft config for pre-trained checkpoint etc.
peft_model_id = "path_to_trained_model"
config = PeftConfig.from_pretrained(peft_model_id)

# load base LLM model and tokenizer
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map={"":0})
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id, device_map={"":0})
model.eval()

print("Peft model loaded")

# use the first sample of the test set
try:
    sample = small_dataset['test'][0]
except KeyError as e:
    print(f"KeyError: {e} - Ensure 'test' exists in the dataset.")
except IndexError as e:
    print(f"IndexError: {e} - Ensure there are samples in 'test'.")

input_ids = tokenizer(sample["dialogue"], return_tensors="pt", truncation=True).input_ids.cuda()
# with torch.inference_mode():
outputs = model.generate(input_ids=input_ids, max_new_tokens=10, do_sample=True, top_p=0.9)
print(f"input sentence: {sample['dialogue']}\n{'---'* 20}")

print(f"summary:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]}")