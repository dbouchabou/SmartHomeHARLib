# coding: utf-8
# !/usr/bin/env python3


class BaseEmbedding(object):

    def __init__(self, sentences, embedding_size = 64, window_size = 20):

        self.sentences = sentences
        self.embedding_size = embedding_size
        self.window_size = window_size

        self.model = None
        self.vocabulary = None
        self.vectors = None

    def train(self):
        
        raise NotImplementedError

    def get_vector_word(self, word):
        
        raise NotImplementedError


    def get_vectors_for_vocabulary(self, vocabulary):

        raise NotImplementedError


    def save_model(self, model_name):

        raise NotImplementedError


    def load_model(self, model_path):
        
        raise NotImplementedError