import peft
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

# paths
dataset_path = "data/processed/Entertainment_Music/train/formal"
tokenised_dataset_path = "data/processed/Entertainment_Music/train/tokenised_formal"
dataset = torch.load(dataset_path)
tokenized_dataset = torch.load(tokenised_dataset_path)
model_id = "chatgpt-2"

# model loading and training
# When tokenising, text is converted to machine readable format, which are the token IDs.
tokenizer = AutoTokenizer.from_pretrained(model_id)
# model_id being the pretrained model you want to use

model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map="auto")

# Define LoRA Config
lora_config = LoraConfig(
 r=16,
 lora_alpha=32,
 target_modules=["q", "v"],
 lora_dropout=0.05,
 bias="none",
 task_type=TaskType.SEQ_2_SEQ_LM
)
# prepare int-8 model for training when you use a quatizied model
# model = repare_model_for_kbit_training(model)

# add LoRA adaptor
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)

output_dir="tutorial"

# Define training args
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
	auto_find_batch_size=True,
    learning_rate=1e-3, # higher learning rate
    num_train_epochs=5,
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=500,
    save_strategy="no",
    report_to="tensorboard",
)

# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

trainer.train()

# Save our LoRA model & tokenizer results
peft_model_id="path_to_trained_model"
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)
# if you want to save the base model to call
# trainer.model.base_model.save_pretrained(peft_model_id)

# Load peft config for pre-trained checkpoint etc.
peft_model_id = "path_to_trained_model"
config = PeftConfig.from_pretrained(peft_model_id)

# load base LLM model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path, device_map={"":0})
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id, device_map={"":0})
model.eval()

print("Peft model loaded")

# use the first sample of the test set
sample = dataset['test'][0]

input_ids = tokenizer(sample["dialogue"], return_tensors="pt", truncation=True).input_ids.cuda()
# with torch.inference_mode():
outputs = model.generate(input_ids=input_ids, max_new_tokens=10, do_sample=True, top_p=0.9)
print(f"input sentence: {sample['dialogue']}\n{'---'* 20}")

print(f"summary:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]}")