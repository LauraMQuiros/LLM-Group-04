import csv
import pandas as pd
import os
from transformers import GPT2Tokenizer
import torch
from src.data_processing.Formality_Transfer_Dataset import FormalityTransferDataset
import pickle

# Defining constants
# Parameters
MODEL_CHECKPOINT = 'gpt2-medium'
PREFIX_FORMAL = '[Formal]'
PREFIX_INFORMAL = '[Informal]'
MAX_INPUT_LEN = 2048
MAX_TARGET_LEN = 128


# Establishing the directory where the script is executed and constructing the path of the dataset
def establish_data_path():
    path = os.getcwd()

    # If you want to run the file specifically from this file, uncomment the two lines below
    path = os.path.abspath(os.path.join(path, os.pardir))
    path = os.path.abspath(os.path.join(path, os.pardir))
    path = os.path.join(path, 'data/raw/GYAFC_Corpus')
    print(path)
    return path


# Loading the splits for the formal/informal pairs based on theme:
# "Entertainment_Music" or "Family_Relationships"
def load_train_tune_test(theme, path):
    formal_train = pd.read_table(path + f'/{theme}/train/formal', names=['label'], header=None, quoting=csv.QUOTE_NONE)
    informal_train = pd.read_table(path + f'/{theme}/train/informal', names=['input_sentence'], header=None,
                                   quoting=csv.QUOTE_NONE)
    train = pd.DataFrame(PREFIX_INFORMAL + informal_train['input_sentence'] + PREFIX_FORMAL + formal_train['label'])

    formal_tune = pd.read_table(path + f'/{theme}/tune/formal', names=['label'], header=None, quoting=csv.QUOTE_NONE)
    informal_tune = pd.read_table(path + f'/{theme}/tune/informal.ref0', names=['input_sentence'], header=None,
                                  quoting=csv.QUOTE_NONE)
    tune = pd.DataFrame(PREFIX_INFORMAL + informal_tune['input_sentence'] + PREFIX_FORMAL + formal_tune['label'])

    formal_test = pd.read_table(path + f'/{theme}/test/formal', names=['label'], header=None, quoting=csv.QUOTE_NONE)
    informal_test = pd.read_table(path + f'/{theme}/test/informal.ref0', names=['input_sentence'], header=None,
                                  quoting=csv.QUOTE_NONE)
    test = pd.DataFrame(PREFIX_INFORMAL + informal_test['input_sentence'] + PREFIX_FORMAL + formal_test['label'])

    return train, tune, test


def check_gpu_availability():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)
    return device


# Adding prefix tokens for the formal and informal datasets
def tokenizing(tokenizer, data, split, theme):
    # Adding the tokens to the model, so it knows to treat them as special tokens
    tokenizer.add_tokens([PREFIX_FORMAL, PREFIX_INFORMAL])


    # tokenization of the conjoined input formal sequences and their formal targets
    formal_informal_conjoined = tokenizer(data.iloc[:, 0].tolist(), max_length=MAX_INPUT_LEN, truncation=True,
                                          padding=True,
                                          return_tensors='pt')

    # Converting the processed data into a dictionary
    data_processed = {
        'input_ids': formal_informal_conjoined['input_ids'],
        'attention_mask': formal_informal_conjoined['attention_mask']

    }

    # Converting the processed data into an instance of the Dataset class to prepare for training
    dataset_processed = FormalityTransferDataset(data_processed)

    # Getting the current directory and move up two levels to reach the project root
    project_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))

    # Appending the correct data path
    save_path = os.path.join(project_root, 'data', 'processed')

    # Checking if the directory exists, and if not, create it
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Pickling the tokenized dataset to avoid pre-processing multiple times
    with open(os.path.join(save_path, f'{theme}_{split}_dataset_processed.pkl'), 'wb') as f:
        pickle.dump(dataset_processed, f)

    return dataset_processed


def main():
    data_path = establish_data_path()
    device = check_gpu_availability()

    train, tune, test = load_train_tune_test('Entertainment_Music', data_path)

    # Initializiing the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_CHECKPOINT,
                                              bos_token='<|startoftext|>', eos_token='<|endoftext|>',
                                              pad_token='<|pad|>')

    # Tokenizing the train, tune and test splits
    tokenized_train_ent = tokenizing(tokenizer, train, "train", "entertainment")
    tokenized_tune_ent = tokenizing(tokenizer, tune, "tune", "entertainment")
    tokenized_test_ent = tokenizing(tokenizer, test, "test", "entertainment")




if __name__ == "__main__":
    main()
