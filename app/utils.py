import pickle
import re
import os 

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.tokenize import word_tokenize as tokenizer 
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import CountVectorizer


BASE_DIR = os.path.dirname(__file__)

stemmer = SnowballStemmer(language = 'english')
# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def normalize(sentence):
    """this function lowercase the sentence, and remove non numbers/characters from the sentence
    # normalize(Hello world#98@) returns Hello world98
    
    """

    normalized_sentence = sentence.lower()
    normalized_sentence = re.compile(r"[^a-zA-Z0-9 ]").sub("", normalized_sentence)
    return normalized_sentence

def tokenize(sentence):
    tokenized_sentence = sentence.split(" ")
    return tokenized_sentence

def stem(word):
    return stemmer.stem(word)

# def in_stop_words(word):
#     return True if word in stop_words else False

def bag_of_words(token_list, all_tokens):
    # bag_words = np.zeros(len(all_words))
    # for i, token in enumerate(word_tokens):
    #     if token in all_words:
    #         bag_words[i] = 1
    bag_words = dict.fromkeys(all_tokens, 0)
    for token in token_list:
        if token in bag_words:
            
            bag_words[token] = 1 
    return bag_words

# cv = CountVectorizer(max_features = 1500)
# feature_names = cv.get_feature_names()
# X = cv.fit_trannsforms(sentences).toarray()

def pickle_save(data, file_name = "data.pickle"):
    with open(file_name, mode = "wb") as f:
        pickle.dump(data, f)
    

def pickle_read(file_name = "data.pickle"):
    if os.path.isfile(os.path.join(BASE_DIR, file_name)):
        with open(file_name, mode="rb") as f:
            data = pickle.load(f)
        return data
    else:
        raise FileExistsError 

class Dataset:
    def __init__(self, X, y):
        self.sample_size = len(y)
        self.X = torch.tensor(X, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)

    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return self.sample_size

class DataLoader:
    def __init__(self, dataset, batch_size = None, shuffle=True):
        self.do_shuffle = shuffle
        self.dataset = dataset
        self.batch_size = batch_size if batch_size is not None else len(dataset)

    def get_indices(self, shuffle=True):
        np.random.seed(42)
        indices = np.arange(len(self.dataset))
        if shuffle:
            np.random.shuffle(indices)
        return indices

    def __iter__(self):
        indices = self.get_indices(self.do_shuffle)
        for i in range(0, len(self.dataset), self.batch_size):
            if i < len(self.dataset) - self.batch_size:
                yield self.dataset[indices[i : i + self.batch_size]]
            else:
                yield self.dataset[indices[i: ]]
