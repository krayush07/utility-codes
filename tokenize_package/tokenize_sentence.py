# -*- coding: utf-8 -*-
import nltk
from nltk.stem.wordnet import WordNetLemmatizer

class SentenceTokenizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def tokenize(self, sentence):
        try:
            return ' '.join(nltk.word_tokenize(sentence.decode('utf-8'))).strip()
        except:
            print sentence

    def lemmatize(self, sentence):
        words = sentence.split()
        lemmatized_str = ''
        for each_word in words:
            lemmatized_str += self.lemmatizer.lemmatize(each_word, pos='v') + ' '
        return lemmatized_str.strip()