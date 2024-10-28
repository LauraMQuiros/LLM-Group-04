import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer
from peft import PeftConfig, PeftModel
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
import json
import pandas as pd
import torch
import os
from fastapi import UploadFile, HTTPException

st.title('Formality Transfer')

# Load the tokenizer and model

my_model = AutoModelForCausalLM.from_pretrained('full_FT_model_changedtokens', use_safetensors=True)
my_tokenizer = GPT2Tokenizer.from_pretrained('full_FT_model_changedtokens')
#my_tokenizer.pad_token = '[PAD]'  # Assuming you have defined a padding token
#my_model.config.pad_token_id = my_tokenizer.pad_token_id

tokenizer_path = 'trained_tokenizer'
model_path = 'base_model'
test_path = 'data/processed/entertainment_test_dataset_processed.pkl'
lora_model_path = 'lora_trained'

# Load tokenizer from its directory
lora_tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)

# Load the saved LoRA configuration
config = PeftConfig.from_pretrained(lora_model_path)
base_model = AutoModelForCausalLM.from_pretrained(model_path, device_map={"": "cpu"})
base_model.resize_token_embeddings(len(lora_tokenizer))

# Load the LoRA model on top of the base model
lora_model = PeftModel.from_pretrained(base_model, lora_model_path, device_map={"": "cpu"})
lora_model.eval()

# User input prompt in the Streamlit chat input widget
prompt = st.chat_input("Please input a short informal text, and I will respond with the formal version!")
if prompt:
    try:
        # Preprocess the text, include attention_mask
        inputs = my_tokenizer(prompt, padding=True, truncation=True, return_tensors="pt")
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        inputs_lora = lora_tokenizer(prompt, padding=True, truncation=True, return_tensors="pt")
        input_ids_lora = inputs_lora['input_ids']
        attention_mask_lora = inputs_lora['attention_mask']

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in text preprocessing: {str(e)}")

    input_text = prompt
    # Generate output from the model using both input_ids and attention_mask
    outputs_base = my_model.generate(input_ids=input_ids, attention_mask=attention_mask, pad_token_id=my_tokenizer.pad_token_id, max_length = ((input_ids.shape[1]) * 2) + 1)
    # Decode the generated output
    # response_base = my_tokenizer.decode(outputs_base[0], skip_special_tokens=True)
    response_base = my_tokenizer.decode(outputs_base[0])

    # Generate output from the model using both input_ids and attention_mask
    outputs_lora = lora_model.generate(input_ids=input_ids_lora, attention_mask=attention_mask_lora, pad_token_id=lora_tokenizer.pad_token_id, max_length = ((input_ids_lora.shape[1]) * 2) + 1)
    # Decode the generated output
    # response_lora = lora_tokenizer.decode(outputs_lora[0], skip_special_tokens=True)
    response_lora = lora_tokenizer.decode(outputs_lora[0])

    st.write(f"Text input: {input_text}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Response from non-peft model")
        st.write(response_base)

    with col2:
        st.subheader("Response from LoRA (peft) model")
        st.write(response_lora)
