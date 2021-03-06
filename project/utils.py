import gzip
import nltk
import pickle
import re
import numpy as np

nltk.download('stopwords')
from nltk.corpus import stopwords

# Paths for all resources for the bot.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': 'model/intent_recognizer.pkl',
    'TAG_CLASSIFIER': 'model/tag_classifier.pkl',
    'TFIDF_VECTORIZER': 'model/tfidf_vectorizer.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'model/thread_embeddings_by_tags',
    'WORD_EMBEDDINGS': 'model/word_embeddings.tsv',
}


def text_prepare(text):
    """Performs tokenization and simple preprocessing."""

    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def load_embeddings(embeddings_path):
    """Loads pre-trained word embeddings from tsv file.

    Args:
      embeddings_path - path to the embeddings file.

    Returns:
      embeddings - dict mapping words to vectors;
      embeddings_dim - dimension of the vectors.
    """

    embeddings = dict()
    embeddings_dim = -1
    for line in open(embeddings_path):
        word, *embedding = line.split('\t')
        embeddings[word] = np.array([float(x) for x in embedding], dtype=np.float32)
        if embeddings_dim < 0:
            embeddings_dim = len(embedding)
        elif embeddings_dim != len(embedding):
            raise ValueError('Embeddings have different dimensions. Current embeddings dim is set to {}, while last loaded is of dim {}.'.format(embeddings_dim, len(embedding)))
    
    return embeddings, embeddings_dim


def question_to_vec(question, embeddings, dim):
    """
        question: a string
        embeddings: dict where the key is a word and a value is its' embedding
        dim: size of the representation

        result: vector representation for the question, as an averate of word embeddings
    """
    wv_sum, count = np.zeros(shape=dim), 0
    for word in question.split():
      if word in embeddings:
        wv_sum += embeddings[word]
        count += 1
    return wv_sum / count if count > 0 else wv_sum


def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    try:
        with gzip.open(filename, 'rb') as f:
            return pickle.load(f)
    except OSError:
        with open(filename, 'rb') as f:
            return pickle.load(f)
