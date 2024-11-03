import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_hub as hub
import os
import pandas as pd
import csv
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import itertools
from tensorflow.keras.callbacks import EarlyStopping

# Loading version 5 of the Universal Sentence Encoder module
url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
universal_encoder = hub.load(url)


def establish_data_path():
    path = os.getcwd()

    # If you want to run the file specifically from this file, uncomment the two lines below
    # path = os.path.abspath(os.path.join(path, os.pardir))
    # path = os.path.abspath(os.path.join(path, os.pardir))
    path = os.path.join(path, 'data/raw/GYAFC_Corpus')
    print(path)
    return path


# Loading the splits for the formal/informal pairs based on theme:
# "Entertainment_Music" or "Family_Relationships"
def load_reshape(theme, path):
    # Loading the formal-informal pairs from the GYAFC test split, to avoid data leakage
    formal_train = pd.read_table(path + f'/{theme}/train/formal', names=['sentence'], header=None,
                                 quoting=csv.QUOTE_NONE)
    informal_train = pd.read_table(path + f'/{theme}/train/informal', names=['sentence'], header=None,
                                   quoting=csv.QUOTE_NONE)

    # Taking the last 2500 entries from the train split, that were left unused during training
    formal_train = formal_train.iloc[-2500:]
    informal_train = informal_train.iloc[-2500:]

    # Creating a new DataFrame for the combined data (sentences and their respective labels: 1 [formal] or 0 [informal] )
    informal_labeled = pd.DataFrame({'sentence': informal_train['sentence'], 'label': 0})  # informal
    formal_labeled = pd.DataFrame({'sentence': formal_train['sentence'], 'label': 1})  # formal
    train_conjoined = pd.concat([informal_labeled, formal_labeled], axis=0, ignore_index=True)

    print(train_conjoined)
    return train_conjoined


def mix_music_relationship_datasets(data_music, data_relationships):
    data_conjoined = pd.concat([data_music, data_relationships], axis=0, ignore_index=True)

    data_conjoined = data_conjoined.sample(frac=1).reset_index(drop=True)
    return data_conjoined


def generate_use_embeddings(data):
    encoded_sentences = universal_encoder(data['sentence']).numpy()

    return encoded_sentences


def split_train_val_test(data):
    # Splitting into training and testing sets
    train_set, test_set = train_test_split(data, test_size=0.2, random_state=42,
                                           stratify=data['label'])

    return train_set, test_set


# Building a model with three dense layers, where the last one is the classification output layer
def build_classifier(units_layer1=128, units_layer2=64, activation="relu", learning_rate=0.001):
    # Initializing the structure of the architecture
    classifier = tf.keras.Sequential([layers.Dense(units=units_layer1, input_shape=(512,), activation=activation),
                                      #layers.Dropout(0.2),
                                      layers.Dense(units=units_layer2, activation=activation),
                                      layers.Dropout(0.2),
                                      layers.Dense(units=1, activation='sigmoid')]
                                     )

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Compiling the model with the Adam optimizer and the chosen learning rate from the GS
    classifier.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy', 'mse'])
    return classifier


def cross_validation_with_grid_search(encodings, labels, num_k_folds, param_types, param_combs):
    k_folds = KFold(n_splits=num_k_folds, random_state=42, shuffle=True)

    # Initializing the best score and the parameters with which it is obtained
    # The best score is initialized to - infinity to set the baseline to be lower than
    # any possible obtained best scores during grid search
    best_params = None
    best_score = -np.inf

    for param_comb in param_combs:

        print(f"Testing combination: {dict(zip(param_types, param_comb))}")

        fold_scores = []
        histories=[]

        # Extract hyperparameters for this combination
        current_parameters = dict(zip(param_types, param_comb))
        print(current_parameters)

        for train_id, test_id in k_folds.split(encodings):
            # Use tf.gather to get the train and validation sets
            train_encodings = tf.gather(encodings, train_id)
            val_encodings = tf.gather(encodings, test_id)

            # Similarly, gather the labels
            train_labels = tf.gather(labels, train_id)
            val_labels = tf.gather(labels, test_id)
            # Initializing the classifier model
            formality_score_classifier = build_classifier(current_parameters['units_layer1'],
                                                          current_parameters['units_layer2'],
                                                          learning_rate=current_parameters['learning_rate'])

            # Initializing early stopping
            early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)


            # Training and evaluating
            training_history=formality_score_classifier.fit(train_encodings, train_labels, batch_size=16, epochs=current_parameters['epochs'],
                                           validation_data=(val_encodings, val_labels),
                                           callbacks=[early_stopping])

            #print(formality_score_classifier.metrics_names)

            val_loss, val_accuracy, val_mse = formality_score_classifier.evaluate(val_encodings, val_labels, verbose=0)
            fold_scores.append(val_accuracy)
            histories.append(training_history.history)

        avg_score = np.mean(fold_scores)
        print(f"Avg validation accuracy for this combination: {avg_score:.4f}")

        # Updating the best parameters if the current combination is better
        if avg_score > best_score:
            best_score = avg_score
            best_params = current_parameters
            best_history = training_history

    print(f"Best Score: {best_score:.4f}")
    print(f"Best Parameters: {best_params}")

    best_params_df = pd.DataFrame([best_params])

    # Save the DataFrame to a CSV file
    best_params_df.to_csv('best_params.csv', index=False)

    return best_params, best_history


import matplotlib.pyplot as plt


def plot_training_history(history):
    """
    Plots the training and validation loss from the model's training history.

    Parameters:
    history: The history object returned by the model's fit method.
    """
    # Extract loss values from history
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)  # Number of epochs

    # Create a figure for the plot
    plt.figure(figsize=(12, 6))

    # Plot training loss
    plt.plot(epochs, loss, label='Training Loss', color='blue', linewidth=2)
    # Plot validation loss
    plt.plot(epochs, val_loss, label='Validation Loss', color='orange', linewidth=2)

    # Add labels and title
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    # Show the plot
    plt.show()


def main():
    data_path = establish_data_path()
    train_ent = load_reshape(theme='Entertainment_Music', path=data_path)
    train_rel = load_reshape(theme='Family_Relationships', path=data_path)
    conj_train = mix_music_relationship_datasets(train_ent, train_rel)
    conj_train.to_csv("why_so_low.csv")
    #conj_train = conj_train.iloc[:9000]

    train, test = split_train_val_test(conj_train)
    print(train.shape, test.shape)

    embedded_train = generate_use_embeddings(train)
    embedded_test = generate_use_embeddings(test)

    print(embedded_train)
    print(embedded_test)

    conj_train.to_csv("classifier_dataset.csv")

    # Defining possible parameters for the grid search
    param_grid = {
        'units_layer1': [ 128, 256, 512],
        'units_layer2': [16, 32, 64],
        'epochs': [10, 20, 50, 80],
        'learning_rate': [ 0.00001, 0.00005, 0.0001]
    }

    param_combinations = list(itertools.product(*param_grid.values()))
    param_names = list(param_grid.keys())

    # Obtain the best parameters
    best_params, best_history = cross_validation_with_grid_search(embedded_train, train['label'], 7, param_names, param_combinations)

    # Plot the validation loss of the best performing model from the cross validation wth grid search
    plot_training_history(best_history)
    print(best_params)


if __name__ == "__main__":
    main()
