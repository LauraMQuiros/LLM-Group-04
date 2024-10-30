import pickle


def get_perplexity(model, tokenizer, data):
    """
    This function calculates the perplexity of the model on the given data using the trigram Kneser-Ney smoothing method.
    :param model:
    :param tokenizer:
    :param data:
    :return:
    """
    # Get the perplexity of the model
    perplexity = model.perplexity(data, tokenizer=tokenizer)
    print(f'The perplexity of the model is: {perplexity}')
    return perplexity

def get_word_overlap(model, tokenizer, data):
    """
    This function calculates the word overlap between the original and generated sentences.
    :param model:
    :param tokenizer:
    :param data:
    :return:
    """
    # Get the word overlap between the original and generated sentences
    word_overlap = model.word_overlap(data, tokenizer=tokenizer)
    print(f'The word overlap between the original and generated sentences is: {word_overlap}')
    return word_overlap

def get_BERT_score(model, tokenizer, data):
    """
    This function calculates the BERT score between the original and generated sentences.
    :param model:
    :param tokenizer:
    :param data:
    :return:
    """
    # Get the BERT score between the original and generated sentences
    BERT_score = model.BERT_score(data, tokenizer=tokenizer)
    print(f'The BERT score between the original and generated sentences is: {BERT_score}')
    return BERT_score

def main():
    # Load the model, tokeniser and data from pkl
    with open('src/models/tokenizer/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    with open('src/models/model/model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('src/data/processed/test.pkl', 'rb') as f:
        data = pickle.load(f)

    # Give the perplexity of the model (fluency)
    get_perplexity(model, tokenizer, data)

    # Give the content-preservation score of the model
    get_word_overlap(model, tokenizer, data)