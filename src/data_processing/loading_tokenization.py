import csv
import pandas as pd
import os
from transformers import GPT2Tokenizer
import torch

# Defining constants
# Parameters
MODEL_CHECKPOINT = 'gpt2-medium'
PREFIX_FORMAL = '<|formal|>'
PREFIX_INFORMAL = '<|informal|>'
MAX_INPUT_LEN = 2048
MAX_TARGET_LEN = 128


# Establishing the directory where the script is executed and constructing the path of the dataset
def establish_data_path():
    path = os.getcwd()
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
    train = pd.concat([informal_train, formal_train], axis=1)

    formal_tune = pd.read_table(path + f'/{theme}/tune/formal', names=['label'], header=None, quoting=csv.QUOTE_NONE)
    informal_tune = pd.read_table(path + f'/{theme}/tune/informal.ref0', names=['input_sentence'], header=None,
                                  quoting=csv.QUOTE_NONE)
    tune = pd.concat([informal_tune, formal_tune], axis=1)

    formal_test = pd.read_table(path + f'/{theme}/test/formal', names=['label'], header=None, quoting=csv.QUOTE_NONE)
    informal_test = pd.read_table(path + f'/{theme}/test/informal.ref0', names=['input_sentence'], header=None,
                                  quoting=csv.QUOTE_NONE)
    test = pd.concat([informal_test, formal_test], axis=1)

    return train, tune, test


def check_gpu_availability():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)
    return device


def tokenizing(tokenizer, data, split):
    tokenizer.add_tokens(['<|formal|>', '<|informal|>'])
    # Add prefixes to all contexts and targets to indicate the style of the given sentence
    data['input_sentence'] = data['input_sentence'].apply(lambda x: PREFIX_INFORMAL + x)
    data['label'] = data['label'].apply(lambda x: PREFIX_FORMAL + x)

    model_inputs = tokenizer(data['input_sentence'].tolist(), max_length=MAX_INPUT_LEN, truncation=True, padding=True,
                             return_tensors='pt')
    targets = tokenizer(data['label'].tolist(), max_length=MAX_INPUT_LEN, truncation=True, padding=True,
                        return_tensors='pt')

    # Converting the processed data into a dictionary
    data_processed = {
        'input_ids': model_inputs['input_ids'],
        'attention_mask': model_inputs['attention_mask'],
        'labels': targets['input_ids'],
        'decoder_attention_mask': targets['attention_mask']
    }

    print(len(model_inputs['input_ids']), len(targets['input_ids']))

    # data_processed_df=pd.DataFrame(data_processed)
    # data_processed_df.save_to_disk(f'{type}_processed_dataset')

    # Returning the data in a dictionary format, as expected by the GPT2
    return data_processed


def main():
    data_path = establish_data_path()
    device = check_gpu_availability()

    train, tune, test = load_train_tune_test('Entertainment_Music', data_path)

    # Initializiing the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_CHECKPOINT,
                                              bos_token='<|startoftext|>', eos_token='<|endoftext|>',
                                              pad_token='<|pad|>')

    tokenized_train = tokenizing(tokenizer, train, "train")
    tokenized_tune = tokenizing(tokenizer, tune, "tune")
    tokenized_test = tokenizing(tokenizer, test, "test")

    tokenized_test.save_to_disk("processed_data")

    print(tokenized_tune)


if __name__ == "__main__":
    main()
