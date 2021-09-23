# coding: utf-8
# !/usr/bin/env python3

import numpy as np
from gensim.models import Word2Vec


from .base_embedding import BaseEmbedding


class Word2VecEventEmbedder(BaseEmbedding):

    def __init__(self, sentences = [], embedding_size = 64, window_size = 20, workers_number = 1, nb_epoch = 100, method = "sg", hierarchic_softmax = False, negative_sample = 10):
        self.sentences = sentences
        self.embedding_size = embedding_size
        self.window_size = window_size
        self.workers_number = workers_number
        self.nb_epoch = nb_epoch
        self.method = method
        self.hierarchic_softmax = hierarchic_softmax
        self.negative_sample = negative_sample

        self.model = None
        self.vocabulary = None
        self.vectors = None
    

    def train(self):

        if self.method == "sg":
            method = 1
        else:
            method = 0

        self.model = Word2Vec(self.sentences, vector_size=self.embedding_size, window=self.window_size, min_count=1, workers=self.workers_number, epochs = self.nb_epoch, sg = method, hs = self.hierarchic_softmax, negative = self.negative_sample)

        # Get word vocabulary
        self.vocabulary = self.model.wv.key_to_index

        # Get vectors for all word in the vocabulary 
        self.vectors = self.model.wv[self.vocabulary]
        #self.vectors = []
        #for word in self.vocabulary:
        #    self.vectors.append(self.model.wv[word])


    def get_vector_word(self, word):
        
        return self.model.wv[word]
    

    def _get_vector_event_mode_sum(self, event):
        
        vector_event = np.zeros(self.embedding_size)
        
        for event_part  in event:
            vector_event += np.sum((vector_event, self.get_vector_word(event_part)), axis=0)
        
        return np.array(vector_event)

    
    def _get_vector_event_mode_concat(self, event):
        
        vector_event = []
        for event_part  in event:
            vector_event = np.concatenate((vector_event, self.get_vector_word(event_part)), axis=0)
        
        return np.array(vector_event)


    def get_vector_event(self, event, mode = "sum"):

        #if not isinstance(event, list):
        #    event = [event]

        if mode == "sum":
            return self._get_vector_event_mode_sum(event)
        elif mode == "concat":
            return self._get_vector_event_mode_concat(event)
        else :
            raise ValueError("Incorect mode please select 'sum' or 'concat' mode")


    def get_vector_sequence(self, sequence_of_events, event_mode = "sum", sequence_mode = "mean"):

        vector_sequence = []

        for event in sequence_of_events:
            vector_sequence.append(self.get_vector_event(event, mode = event_mode))
        
        if sequence_mode == "mean":
            vector_sequence = np.mean(vector_sequence, axis = 0)

        elif sequence_mode == "sum":
            vector_sequence = np.sum(vector_sequence, axis = 0)
        
        else:
           raise ValueError("Incorect 'sequence_mode' please select 'mean' or 'sum' mode") 
        
        return vector_sequence

    
    def get_vectors_for_vocabulary(self, vocabulary):

        vectors = []

        cpt_oov = 0

        for word in vocabulary:
            
            if word in self.vocabulary:
                vectors.append(np.array(self.get_vector_word(word), dtype='float32'))
            
            # If word does not exist use a zeros vector
            else:
                vectors.append(np.array(np.zeros(self.embedding_size), dtype='float32'))
                cpt_oov += 1

        #print("NUMBER OF OOV: {}".format(cpt_oov))
        #input("Press Enter to continue...")

        return np.array(vectors, dtype='float32')


    def most_similar(self, word, top_n = 10):

        return self.model.wv.most_similar(word, topn = top_n)


    def save_model(self, model_name):

        self.model.save(model_name)


    def load_model(self, model_path):

        self.model = Word2Vec.load(model_path)

        # Get word vocabulary
        self.vocabulary = self.model.wv.key_to_index

        # Get vectors for all word in the vocabulary
        self.vectors = self.model.wv[self.vocabulary] 
        #self.vectors = []
        #for word in self.vocabulary:
        #    self.vectors.append(self.model.wv[word])

        # get vectors lenght
        self.embedding_size = len(self.vectors[0])
