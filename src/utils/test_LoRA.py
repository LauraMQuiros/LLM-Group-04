import os
import sys
import pickle
from transformers import GPT2Tokenizer, AutoModelForCausalLM
from peft import PeftConfig, PeftModel


# Set paths to saved tokenizer and test dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
sys.path.append(project_root)
sys.path.append(os.path.join(os.getcwd(), 'src/data_processing'))
from src.data_processing.FormalityTransferDataset import FormalityTransferDataset

tokenizer_path = os.path.join(os.getcwd(), 'src/models/trained_tokenizer')
model_path = os.path.join(os.getcwd(), 'src/models/base_model')
test_path = os.path.join(os.getcwd(), 'data/processed/entertainment_test_dataset_processed.pkl')
lora_model_path = os.path.join(os.getcwd(), 'src/models/lora_trained')

# Load test dataset and tokenizer
with open(test_path, 'rb') as f:
    test: FormalityTransferDataset = pickle.load(f)


# Load tokenizer from its directory
tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)

# Load the saved LoRA configuration
config = PeftConfig.from_pretrained(lora_model_path)
base_model = AutoModelForCausalLM.from_pretrained(model_path, device_map={"": "cpu"})
base_model.resize_token_embeddings(len(tokenizer))

# Load the LoRA model on top of the base model
lora_model = PeftModel.from_pretrained(base_model, lora_model_path, device_map={"": "cpu"})
lora_model.eval()

print("LoRA model loaded")

# Use the first sample from the test set
sample = test[0]

# Move input_ids and attention_mask to CPU, handle unsqueeze for batch processing
input_ids = sample["input_ids"].unsqueeze(0) if sample["input_ids"].dim() == 1 else sample["input_ids"]
attention_mask = sample["attention_mask"].unsqueeze(0) if sample["attention_mask"].dim() == 1 else sample["attention_mask"]

# Generate outputs using the LoRA model
outputs = lora_model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=10, do_sample=True, top_p=0.9)

# Print the untokenized input sentence and generated output
print(f"Input sentence (untokenized): {tokenizer.decode(sample['input_ids'], skip_special_tokens=True)}")
print(f"Generated output:{tokenizer.decode(outputs[0], skip_special_tokens=True)}")