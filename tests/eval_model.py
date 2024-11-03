import pickle
import torch
import nltk
import os
import sys
from tqdm import tqdm
from nltk.lm import KneserNeyInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.util import ngrams
from transformers import AutoModelForCausalLM, GPT2Tokenizer
from peft import PeftModel
from bert_score import score as BERT_score

def get_model_predictions(model, tokenizer, data):
    """
    Generates predictions using the model for each sentence in the provided data.
    :param model: GPT2 FT or PEFT model to evaluate
    :param tokenizer: Tokenizer for the model
    :param data: List of tokenised sentences for evaluation
    :return: List of generated sentences
    """
    generated_sentences = []

    # Step 1: Generate predictions using the model for each sentence
    with torch.no_grad():
        for tokens in tqdm(data, desc="Generating sentences", unit="sentence"):
            input_sentence = tokenizer.convert_tokens_to_string(tokens)
            inputs = tokenizer(input_sentence, return_tensors="pt")
            input_ids = inputs["input_ids"]

            # Generate predictions using model
            output = model.generate(input_ids, max_length=len(input_ids[0]) + 20, num_return_sequences=1)
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            generated_sentences.append(generated_text)

    return generated_sentences

def get_perplexity_kneser_ney(generated_sentences, n=3):
    """
    Calculates the perplexity of model predictions on the given data using Kneser-Ney smoothing.
    :param generated_sentences: List of generated sentences
    :param n: Order of the n-gram model (3 for trigram)
    :return average_perplexity: Average perplexity score for the data
    """
    # Step 1: Prepare n-grams for Kneser-Ney
    # Pad trigrams and prepare vocabulary for Kneser-Ney smoothing
    train_data, vocab = padded_everygram_pipeline(n, generated_sentences)

    # Initialize Kneser-Ney language model
    kn_model = KneserNeyInterpolated(n)
    kn_model.fit(train_data, vocab)

    # Step 2: Calculate perplexity of the generated sentences
    total_perplexity = 0
    for sentence_tokens in tqdm(generated_sentences, desc="Calculating perplexity", unit="sentence"):
        sentence_ngrams = list(ngrams(sentence_tokens, n))
        perplexity = kn_model.perplexity(sentence_ngrams)
        total_perplexity += perplexity

    # Calculate average perplexity across all generated sentences
    average_perplexity = total_perplexity / len(generated_sentences) if len(generated_sentences) > 0 else float('inf')
    return average_perplexity

def get_word_overlap(generated_sentences, data):
    """
    This function calculates the word overlap between the original and generated sentences.
    :param generated_sentences: the generated sentences from the GPT2 model
    :param data: the test data from which the generated sentences were generated
    :return overlap: the word overlap between the original and generated sentences
    """
    overlap_counts = []

    for original, generated in zip(data, generated_sentences):
        # Convert the original and generated sentences to sets of words
        original_words = set(original)  # Assuming 'original' is a list of tokens
        generated_words = set(generated.split())  # Split generated sentence into words

        # Calculate the overlap between the two sets
        overlap = original_words.intersection(generated_words)

        # Store the count of overlapping words
        overlap_counts.append(len(overlap))

    # Calculate average overlap across all sentence pairs
    average_overlap = sum(overlap_counts) / len(overlap_counts) if overlap_counts else 0
    return average_overlap

def get_BERT_score(generated_sentences, data):
    """
        This function calculates the BERT score between the original and generated sentences.
        :param generated_sentences: The generated sentences from the GPT2 model
        :param data: The original sentences (in their natural form, not tokenized) for evaluation
        :return: BERT_score: The BERT score between the original and generated sentences
    """
    # Compute BERT scores
    P, R, F1 = BERT_score(generated_sentences, data, lang='en', verbose=True)

    # Return the F1 score as a representative BERT score
    return F1.mean().item()  # Convert tensor to a standard float

def main():
    # Load the model, tokeniser and data from pkl
    path = os.getcwd()
    sys.path.append(path)
    tokenizer_path = 'src/models/tokenizer'
    model_path = 'src/models/model'
    lora_model_path = 'src/models/lora_model'
    # data path is path + 'src/data/processed/data.pkl'
    data_path = os.path.join(path, 'data/processed/test.pkl')

    lora_tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    print(len(lora_tokenizer))
    base_model = AutoModelForCausalLM.from_pretrained(model_path, device_map={"": "cpu"})
    base_model.resize_token_embeddings(len(lora_tokenizer))

    # Load the LoRA model on top of the base model
    lora_model = PeftModel.from_pretrained(base_model, lora_model_path, device_map={"": "cpu"})
    lora_model.eval()

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    model_predictions = get_model_predictions(lora_model, lora_tokenizer, data)

    # Give the fluency of the model (perplexity score with Kneser-Ney smoothing)
    PPL_score = get_perplexity_kneser_ney(model_predictions)
    print(f'The perplexity of the model is: {PPL_score}')

    # Give the content-preservation score of the model (BERT score and word overlap)
    # BERT_score = get_BERT_score(model_predictions, data)
    # print(f'The BERT score of the model is: {BERT_score}')
    # average_word_overlap = get_word_overlap(model_predictions, data)
    # print(f'The word overlap of the model is: {average_word_overlap}')

    # Give the formality of the model (formality score given by the classifier)

if __name__ == "__main__":
    main()