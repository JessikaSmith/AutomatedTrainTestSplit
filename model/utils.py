import os.path
import sys
import keras

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from gensim.models import KeyedVectors
from gensim.models import FastText
import pickle

import requests
import re

import pymorphy2
import pymystem3

morph = pymorphy2.MorphAnalyzer()

path_to_w2v = '/data/GAforAutomatedTrainTestSplit/model/produced_data/ruwikiruscorpora_upos_skipgram_300_2_2018.vec'
path_to_fasttext_emb = '/tmp/wiki.ru.bin'
path_to_fasttext_emb_2 = '/data/GAforAutomatedTrainTestSplit/model/produced_data/ft_native_300_ru_wiki_lenta_lemmatize.bin'
path_to_fasttext_unlem = '/tmp/ft_native_300_ru_wiki_lenta_lower_case.bin'

upt_url = 'https://raw.githubusercontent.com/akutuzov/universal-pos-tags/4653e8a9154e93fe2f417c7fdb7a357b7d6ce333/ru-rnc.map'
m = pymystem3.mystem.Mystem(mystem_bin='/home/gmaster/mystem')

mapping = {}
r = requests.get(upt_url, stream=True)
for pair in r.text.split('\n'):
    pair = re.sub('\s+', ' ', pair, flags=re.U).split(' ')
    if len(pair) > 1:
        mapping[pair[0]] = pair[1]


def embedding(emb_type):
    if emb_type == 'w2v':
        model = KeyedVectors.load_word2vec_format(path_to_w2v)
    if emb_type == 'fasttext':
        model = FastText.load_fasttext_format(path_to_fasttext_emb)
    if emb_type == 'fasttext_2':
        print('loading fasttext embedding...')
        model = FastText.load_fasttext_format(path_to_fasttext_emb_2)
        print('Done!')
    if emb_type == 'fasttext_unlem':
        model = FastText.load_fasttext_format(path_to_fasttext_unlem)
    return model


def add_universal_tag(word):
    processed = m.analyze(word)
    tagged = []
    for w in processed:
        try:
            lemma = w["analysis"][0]["lex"].lower().strip()
            pos = w["analysis"][0]["gr"].split(',')[0]
            pos = pos.split('=')[0].strip()
            if pos in mapping:
                tagged.append(lemma + '_' + mapping[pos])  # tags conversion
            else:
                tagged.append(lemma + '_X')
        except KeyError:
            continue
    return tagged


class Processor:

    def __init__(self, max_features, emb_type, max_len, emb_dim=300):
        self.tokenizer = None
        self.max_features = max_features
        self.emb_type = emb_type
        self.model = None
        self.emb_dim = emb_dim
        self.embedding_matrix = None
        self.x_train_name = None
        self.max_len = max_len

    def prepare_embedding_matrix(self, word_index, x_train_name):
        print('Starting embedding matrix preparation...')
        embedding_matrix = np.zeros((self.max_features, self.emb_dim))
        if self.emb_type == 'w2v':
            for word, i in word_index.items():
                try:
                    print(word)
                    emb_vect = self.model.wv[add_universal_tag(word)].astype(np.float32)
                    embedding_matrix[i] = emb_vect
                # out of vocabulary exception
                except:
                    print(word)
        else:
            for word, i in word_index.items():
                try:
                    emb_vect = self.model.wv[word]
                    embedding_matrix[i] = emb_vect.astype(np.float32)
                # out of vocabulary exception
                except:
                    print(word)
        np.save('produced_data/%s_%s_%s.npy' % (
            self.emb_type, x_train_name, self.max_features), embedding_matrix)

        return embedding_matrix

    def fit_processor(self, x_train, x_train_name, other=None):
        self.x_train_name = x_train_name
        try:
            self.embedding_matrix = np.load(
                'produced_data/%s_%s_%s.npy' % (
                    self.emb_type, x_train_name, self.max_features))
            with open('produced_data/tokenizer_%s_%s_%s.pickle' % (
                    self.emb_type, x_train_name, self.max_features), 'rb') as handle:
                self.tokenizer = pickle.load(handle)

        # not found exception
        except:  # to check
            print('No model found...initialization...')
            #x_train = [sent[0] for sent in x_train]
            self.tokenizer = Tokenizer(num_words=self.max_features + 1, oov_token='oov')
            if not other:
                self.tokenizer.fit_on_texts(x_train)
            else:
                if isinstance(other[0], list):
                    other = [sent[0] for sent in other]
                self.tokenizer.fit_on_texts(x_train)
            # hopefully this staff helps to avoid issues with oov (NOT SURE needs to be checked)
            self.tokenizer.word_index = {e: i for e, i in self.tokenizer.word_index.items() if i <= self.max_features}
            self.tokenizer.word_index[self.tokenizer.oov_token] = self.max_features + 1
            word_index = self.tokenizer.word_index
            with open('produced_data/tokenizer_%s_%s_%s.pickle' % (
                    self.emb_type, x_train_name, self.max_features), 'wb') as handle:
                pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # ======================== write tokenizer to file ===================================

            print('Amount of unique tokens %s' % len(word_index))

            self.model = embedding(self.emb_type)
            self.embedding_matrix = self.prepare_embedding_matrix(word_index, x_train_name)

    def prepare_input(self, x, y=None):
        # prepare x data
        if isinstance(x[0], list):
            x = [sent[0] for sent in x]
        sequences_x = self.tokenizer.texts_to_sequences(x)
        x = pad_sequences(sequences_x, maxlen=self.max_len)
        # prepare labels
        if y:
            if isinstance(y[0], list):
                y = [y[0] for y in y]
            y = np.asarray(y)
            return x, y
        return x

    def prepare_sequence(self, text):
        text = [text]
        sequences = self.tokenizer.texts_to_sequences(text)
        x = pad_sequences(sequences, maxlen=self.max_len)
        return x

    def prepare_custom_embedding(self,  vocabulary, x_train_name='custom'):
        self.max_features = len(vocabulary)
        try:
            self.embedding_matrix = np.load('produced_data/%s_%s_%s.npy' % (
                    self.emb_type, x_train_name, self.max_features))
        except:
            print('Starting embedding matrix preparation...')
            self.model = embedding(self.emb_type)
            embedding_matrix = np.zeros((len(vocabulary), self.emb_dim))
            if self.emb_type == 'w2v':
                for i, word in enumerate(vocabulary):
                    try:
                        emb_vect = self.model.wv[add_universal_tag(word)].astype(np.float32)
                        embedding_matrix[i] = emb_vect
                    # out of vocabulary exception
                    except:
                        print(word)
            else:
                for i, word in enumerate(vocabulary):
                    try:
                        emb_vect = self.model.wv[word]
                        embedding_matrix[i] = emb_vect.astype(np.float32)
                    # out of vocabulary exception
                    except:
                        print(word)

            self.embedding_matrix = embedding_matrix
            np.save('produced_data/%s_%s_%s.npy' % (
                self.emb_type, x_train_name, self.max_features), embedding_matrix)
