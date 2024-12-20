import csv
import pandas as pd
import os
import sys
from transformers import GPT2Tokenizer
import torch
import pickle

# needed to import the FormalityTransferDataset class from the src.data_processing.Formality_Transfer_Dataset module
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
sys.path.append(project_root)

from src.data_processing.Formality_Transfer_Dataset import FormalityTransferDataset

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

    # If you want to run the file specifically from this file path instead of root, uncomment the two lines below
    # path = os.path.abspath(os.path.join(path, os.pardir))
    # path = os.path.abspath(os.path.join(path, os.pardir))
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

    # Leaving 2500 sentence pairs from each of the datasets for the two themes
    # These sentence pairs will then be used for the training of the classifier used to
    # assess the formality of the fine-tuned models
    train = train.iloc[:-2500]


    formal_tune = pd.read_table(path + f'/{theme}/tune/formal', names=['label'], header=None, quoting=csv.QUOTE_NONE)
    informal_tune = pd.read_table(path + f'/{theme}/tune/informal.ref0', names=['input_sentence'], header=None,
                                  quoting=csv.QUOTE_NONE)
    tune = pd.DataFrame(PREFIX_INFORMAL + informal_tune['input_sentence'] + PREFIX_FORMAL + formal_tune['label'])

    formal_test = pd.read_table(path + f'/{theme}/test/formal', names=['label'], header=None, quoting=csv.QUOTE_NONE)
    informal_test = pd.read_table(path + f'/{theme}/test/informal.ref0', names=['input_sentence'], header=None,
                                  quoting=csv.QUOTE_NONE)
    test = pd.DataFrame(PREFIX_INFORMAL + informal_test['input_sentence'] + PREFIX_FORMAL + formal_test['label'])

    return train, tune, test

def mix_music_relationship_datasets(data_music, data_relationships):
    data_conjoined= pd.concat([data_music, data_relationships], axis=0, ignore_index=True)

    data_conjoined = data_conjoined.sample(frac=1).reset_index(drop=True)
    return data_conjoined

def check_gpu_availability():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)
    return device


# Adding prefix tokens for the formal and informal datasets
def tokenizing(tokenizer, data, split):

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
    project_root = os.getcwd()
    # if you want to run it from this file path, uncomment the line below
    # project_root = os.path.abspath(os.path.join(project_root, os.pardir, os.pardir))
    # Appending the correct data path
    save_path = os.path.join(project_root, 'data', 'processed')

    # Checking if the directory exists, and if not, create it
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Pickling the tokenized dataset to avoid pre-processing multiple times
    with open(os.path.join(save_path, f'{split}.pkl'), 'wb') as f:
        pickle.dump(dataset_processed, f)

    return dataset_processed


def main():
    data_path = establish_data_path()
    device = check_gpu_availability()

    train_rel, tune_rel, test_rel = load_train_tune_test('Family_Relationships', data_path)
    train_ent, tune_ent, test_ent = load_train_tune_test('Entertainment_Music', data_path)


    train=mix_music_relationship_datasets(train_ent, train_rel)
    tune=mix_music_relationship_datasets(tune_ent, tune_rel)
    test=mix_music_relationship_datasets(test_ent, test_rel)

    # Initializing the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_CHECKPOINT,
                                              bos_token='<|startoftext|>', eos_token='<|endoftext|>',
                                              pad_token='<|pad|>')

    # Adding the tokens to the model, so it knows to treat them as special tokens
    tokenizer.add_tokens([PREFIX_FORMAL, PREFIX_INFORMAL])


    # save tokeniser to src/models/tokenizer
    project_root = os.getcwd()
    save_path = os.path.join(project_root, 'src', 'models', 'tokenizer')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # pickle tokenizer
    with open(os.path.join(save_path, 'tokenizer.pkl'), 'wb') as f:
        pickle.dump(tokenizer, f)

    # Tokenizing the train, tune and test splits
    tokenized_train = tokenizing(tokenizer, train, "train")
    tokenized_tune = tokenizing(tokenizer, tune, "tune")
    tokenized_test = tokenizing(tokenizer, test, "test")


if __name__ == "__main__":
    main()
