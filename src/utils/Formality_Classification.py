import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_hub as hub
import os
import pandas as pd
import csv
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score
import pickle

from tensorflow.python.keras.metrics import accuracy

# Loading version 5 of the Universal Sentence Encoder module
url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
universal_encoder = hub.load(url)


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
def load_reshape(theme, path):

    # Loading the formal-informal pairs from the GYAFC test split, to avoid data leakage
    formal_test = pd.read_table(path + f'/{theme}/test/formal', names=['sentence'], header=None, quoting=csv.QUOTE_NONE)
    informal_test = pd.read_table(path + f'/{theme}/test/informal.ref0', names=['sentence'], header=None,
                                  quoting=csv.QUOTE_NONE)

    # Creating a new DataFrame for the combined data (sentences and their respective labels: 1 [formal] or -1 [informal] )
    informal_labeled = pd.DataFrame({'sentence': informal_test['sentence'], 'label': -1})  # informal
    formal_labeled = pd.DataFrame({'sentence': formal_test['sentence'], 'label': 1})  # formal
    test_conjoined = pd.concat([informal_labeled, formal_labeled], axis=0, ignore_index=True)

    # Shuffling the resulting conjoined dataset, as to introduce randomness in the order of tha sentence-label pairs
    test_conjoined = test_conjoined.sample(frac=1).reset_index(drop=True)

    print(test_conjoined)
    return test_conjoined


def generate_use_embeddings(data):
    encoded_sentences= universal_encoder(data['sentence'])

    return encoded_sentences

def split_train_val_test(data):
    # Splitting into training and testing sets
    train_set, test_set = train_test_split(data, test_size=0.2, random_state=42,
                                           stratify=data['label'])

    return train_set, test_set

# Building a model with three dense layers, where the last one is the classification output layer
def build_classifier():
    classifier= tf.keras.Sequential([layers.Dense(units=128, input_shape=(512,), activation='relu'),
    layers.Dense(units=64, activation='relu'),
    layers.Dense(units=1, activation='tanh') ]
     )
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier



def main():
    data_path = establish_data_path()
    conj_test = load_reshape(theme='Entertainment_Music', path=data_path)

    train, test = split_train_val_test(conj_test)

    embedded_train=generate_use_embeddings(train)
    embedded_test=generate_use_embeddings(test)

    print(embedded_train)
    print(embedded_test)

    #conj_test.to_csv("classifier_dataset.csv")


    formality_score_classifier = build_classifier()
    scores = cross_validate(formality_score_classifier, embedded_train, train['label'], return_train_score=True, cv=5, n_jobs=-1)
    print(scores)
    print(accuracy_score(train['label'], scores))

    formality_score_classifier.fit(embedded_train, train['label'])



if __name__ == "__main__":
    main()
