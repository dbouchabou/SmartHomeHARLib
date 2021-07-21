# coding: utf-8
# !/usr/bin/env python3


import os
import json
import numpy as np
from tqdm import tqdm


import tensorflow as tf

from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.activations import *
from tensorflow.keras.utils import *
from tensorflow.keras.callbacks import *

from tensorflow import keras
from tensorflow.keras import backend as K

# delette some warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from SmartHomeHARLib.utils.data_segmentation import *

from .base_embedding import BaseEmbedding


def perplexity(y_true, y_pred):
    """
    Popular metric for evaluating language modelling architectures.
    More info: http://cs224d.stanford.edu/lecture_notes/LectureNotes4.pdf
    """
    cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
    return K.mean(K.exp(K.mean(cross_entropy, axis=-1)))


class ELMoEventEmbedder(BaseEmbedding):
    """
    ELMo class to create a contextualized language model

    Attributes
    ----------
    sentences : array, optional
        raw sentences array (default is [])
    
    embedding_size : int, optional
        size of vectors (default is 64)

    window_size : int, optional
        size for the past and future context (default is 20)
    
    step : int, optional
        shift number for windows context (default is 1)
    
    nb_epoch : int, optional
        maximum number of epochs for the training phase (default is 100)

    batch_size : int, optional
        size of the training batch (default is 64)
    
    verbose : boolean, optional
        verbose the training phase (default is True)

    residual : boolean, optional
        activate the residual connection in the model (default is True)

    Methods
    -------
    tokenize(filter = '', lower = False)
        Process and tokenize sentences
    """

    def __init__(self, sentences = [], embedding_size = 64, window_size = 20, step = 1, nb_epoch = 100, batch_size = 64, verbose = True, residual = True ):
        
        self.sentences = sentences
        self.embedding_size = embedding_size
        self.window_size = window_size
        self.step = step
        self.residual = residual
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.verbose = verbose

        self.model = None
        self.vocabulary = []
        self.vectors = []

        self.tokenizer = None
        
        self.forward_inputs = []
        self.forward_outputs = []

        self.backward_inputs = []
        self.backward_outputs = []

        self.best_model_path = "elmo_model.h5"
        self.elmo_model = None

    
    def tokenize(self, filter = '', lower = False ):
        """
        Process and tokenize sentences

        Parameters
        ----------
        filter : str
            filters caracters (default is '')
        lower : boolean
            lower case words before tokenize them (default is False)
        
        """

        # add special tokens to the begining and the end of sentences
        for i, sentence in enumerate(self.sentences):
            self.sentences[i] = "<start> " + sentence + " <end>"


        # concatenate all sentences into one big sequence
        #self.sentences = [" ".join(self.sentences)]

        self.tokenizer = Tokenizer(filters = filter, lower = lower)
        self.tokenizer.fit_on_texts(self.sentences)

        # replace words into sentences by their index token
        self.sentences = np.array(self.tokenizer.texts_to_sequences(self.sentences))

        self.vocabulary = self.tokenizer.word_index
    

    # TODO: Delete this function, better to use the one in utils
    def __slidingWindow(self, X, Y, winSize, step=1):

        chunksX = []
        chunksY = []

        numOfChunks = int(((X.shape[0] - winSize) / step) + 1)

        for i in range(0, numOfChunks * step, step):
            # sew = self.df[i:i+winSize]

            chunksX.append(X[i:i + winSize])
            chunksY.append(Y[i:i + winSize])

        return chunksX, chunksY


    def prepare(self):

        self.forward_inputs = []
        self.forward_outputs = []

        self.backward_inputs = []
        self.backward_outputs = []

        f_inputs = []
        b_inputs = []

        for i, sentence in enumerate(self.sentences):
            # remove end token
            f_inputs.append(sentence[:-1])
            # remove start token
            b_inputs.append(sentence[1:])


        #print(f_inputs)
        #print(b_inputs)
        
        for i in range(len(f_inputs)):
            fi, fo = sliding_window(np.array(f_inputs[i]), np.array(b_inputs[i]), self.window_size, self.step)


            self.forward_inputs.append(fi)
            self.forward_outputs.append(fo)

            #
            #self.forward_inputs=np.concatenate((self.forward_inputs, fi), axis=0)
            #self.forward_outputs=np.concatenate((self.forward_outputs, fo), axis=0)
            #

            # reverse for backward
            b_inputs_revesed = b_inputs[i][::-1]
            f_inputs_reversed = f_inputs[i][::-1]
        
            bi, bo = sliding_window(np.array(b_inputs_revesed), np.array(f_inputs_reversed), self.window_size, self.step)

            self.backward_inputs.append(bi)
            self.backward_outputs.append(bo)

            #self.backward_inputs=np.concatenate((self.backward_inputs, bi), axis=0)
            #self.backward_outputs=np.concatenate((self.backward_outputs, bo), axis=0)



            self.forward_inputs = np.array(self.forward_inputs[0])
            self.forward_outputs = np.array(self.forward_outputs[0])

            self.backward_inputs = np.array(self.backward_inputs[0])
            self.backward_outputs = np.array(self.backward_outputs[0])


    def prepare_2(self):

        self.forward_inputs = []
        self.forward_outputs = []

        self.backward_inputs = []
        self.backward_outputs = []
 
        self.sentences = self.sentences[0]
        
        for i in range(1,len(self.sentences)-1, self.step): 

            if i-self.window_size < 0:
                f_inputs = self.sentences[:i]
            else:
                f_inputs = self.sentences[i-self.window_size:i]

            word_to_predict_forward = self.sentences[i:i+1]

            self.forward_inputs.append(f_inputs)
            self.forward_outputs.append(word_to_predict_forward)


        # reverse for backward
        sentence_revesed = self.sentences[::-1]

        for i in range(1,len(sentence_revesed)-1, self.step): 

            if i-self.window_size < 0:
                b_inputs = sentence_revesed[:i]
            else:
                b_inputs = sentence_revesed[i-self.window_size:i]

            word_to_predict_backward = sentence_revesed[i:i+1]

            self.backward_inputs.append(b_inputs)
            self.backward_outputs.append(word_to_predict_backward)

        
        self.forward_inputs = pad_sequences(self.forward_inputs, padding = 'post')
        self.backward_inputs = pad_sequences(self.backward_inputs, padding = 'post')

        self.forward_inputs = np.array(self.forward_inputs)
        self.forward_outputs = np.array(self.forward_outputs)

        self.backward_inputs = np.array(self.backward_inputs)
        self.backward_outputs = np.array(self.backward_outputs)
    

    def prepare_3(self):

        self.forward_inputs = []
        self.forward_outputs = []

        self.backward_inputs = []
        self.backward_outputs = []
 
        self.sentences = self.sentences[0]
        
        self.forward_inputs, self.forward_outputs = progressive_sliding_window(self.sentences,self.sentences,self.window_size,self.step)

        # reverse for backward
        sentences_revesed = self.sentences[::-1]

        self.backward_inputs, self.backward_outputs = progressive_sliding_window(sentences_revesed,sentences_revesed,self.window_size,self.step)
        
        self.forward_inputs = pad_sequences(self.forward_inputs, padding = 'post')
        self.backward_inputs = pad_sequences(self.backward_inputs, padding = 'post')

        self.forward_inputs = np.array(self.forward_inputs)
        self.forward_outputs = np.array(self.forward_outputs)

        self.backward_inputs = np.array(self.backward_inputs)
        self.backward_outputs = np.array(self.backward_outputs)


    def prepare_4(self):

        self.forward_inputs = []
        self.forward_outputs = []

        self.backward_inputs = []
 
        self.sentences = self.sentences
        
        for sentence in self.sentences:
        
            for i in range(self.step,len(sentence)-1,self.step): 

                if i-self.window_size < 0:
                    f_inputs = sentence[:i]
                else:
                    f_inputs = sentence[i-self.window_size:i]
                    
                if i+self.step+self.window_size > len(sentence):
                    b_inputs = sentence[i+self.step:]
                else:
                    b_inputs = sentence[i+self.step:i+self.step+self.window_size]

                word_to_predict_forward = sentence[i:i+self.step]

                self.forward_inputs.append(f_inputs)
                # return the sequence
                self.backward_inputs.append(b_inputs[::-1])
                self.forward_outputs.append(word_to_predict_forward)

        
        self.forward_inputs = pad_sequences(self.forward_inputs, padding = 'post')
        self.backward_inputs = pad_sequences(self.backward_inputs, padding = 'post')

        self.forward_inputs = np.array(self.forward_inputs)
        self.forward_outputs = np.array(self.forward_outputs)

        self.backward_inputs = np.array(self.backward_inputs)
        self.backward_outputs = np.array(self.backward_outputs)


    def __model(self):

        vocab_size = len(self.vocabulary)

        forward_inputs = Input(shape=((self.window_size,)))
        backward_inputs = Input(shape=((self.window_size,)))

        embedding = Embedding(input_dim = vocab_size+1, output_dim = self.embedding_size, input_length = self.window_size, mask_zero=False, name="embedding_layer")

        forward_inputs_embedded = embedding(forward_inputs)
        backward_inputs_embedded = embedding(backward_inputs)

        # forward
        forward_l1 = LSTM(self.embedding_size, return_sequences=True, name="forward_lstm_layer_1") (forward_inputs_embedded)

        if self.residual:
            forward_l1 = forward_l1 + left_context_embedded

        forward_l1 = BatchNormalization()(forward_l1)
        forward_l2 = LSTM(self.embedding_size, return_sequences=True, name="forward_lstm_layer_2") (forward_l1)


        # backward
        backward_l1 = LSTM(self.embedding_size, return_sequences=True, name="backward_lstm_layer_1") (backward_inputs_embedded)

        if self.residual:
            backward_l1 = backward_l1 + right_context_embedded

        backward_l1 = BatchNormalization()(backward_l1)
        backward_l2 = LSTM(self.embedding_size, return_sequences=True, name="backward_lstm_layer_2") (backward_l1)


        softmax_layer = TimeDistributed(Dense(vocab_size, activation='softmax'))
        
        forward_outputs = softmax_layer(forward_l2)
        backward_outputs = softmax_layer(backward_l2)

        model = Model(inputs=[forward_inputs,backward_inputs], outputs=[forward_outputs,backward_outputs], name="ELMoLike")


        return model


    def __model_2(self):

        vocab_size = len(self.vocabulary)

        forward_inputs = Input(shape=((self.window_size,)))

        embedding = Embedding(input_dim = vocab_size+1, output_dim = self.embedding_size, input_length = self.window_size, mask_zero=True, name="embedding_layer")

        forward_inputs_embedded = embedding(forward_inputs)

        # forward
        forward_l1 = LSTM(self.embedding_size, return_sequences=True, name="forward_lstm_layer_1") (forward_inputs_embedded)

        if self.residual:
            forward_l1 = forward_l1 + left_context_embedded

        forward_l1 = BatchNormalization()(forward_l1)
        forward_l2 = LSTM(self.embedding_size, return_sequences=True, name="forward_lstm_layer_2") (forward_l1)




        softmax_layer = TimeDistributed(Dense(vocab_size, activation='softmax'))
        
        forward_outputs = softmax_layer(forward_l2)

        model = Model(inputs=forward_inputs, outputs=forward_outputs, name="ELMoLike")


        return model


    def __model_3(self):

        vocab_size = len(self.vocabulary)

        forward_inputs = Input(shape=((None,)))
        backward_inputs = Input(shape=((None,)))

        embedding = Embedding(input_dim = vocab_size+1, output_dim = self.embedding_size, mask_zero=True, name="embedding_layer")

        forward_inputs_embedded = embedding(forward_inputs)
        backward_inputs_embedded = embedding(backward_inputs)

        # forward
        forward_l1 = LSTM(self.embedding_size, return_sequences=True, name="forward_lstm_layer_1") (forward_inputs_embedded)

        if self.residual:
            forward_l1 = Add(name="forward_residual")([forward_l1, forward_inputs_embedded])

        #forward_l1 = BatchNormalization()(forward_l1)
        forward_l2 = LSTM(self.embedding_size, return_sequences=False, name="forward_lstm_layer_2") (forward_l1)


        # backward
        backward_l1 = LSTM(self.embedding_size, return_sequences=True, name="backward_lstm_layer_1") (backward_inputs_embedded)

        if self.residual:
            backward_l1 = Add(name="backward_residual")([backward_l1, backward_inputs_embedded])

        #backward_l1 = BatchNormalization()(backward_l1)
        backward_l2 = LSTM(self.embedding_size, return_sequences=False, name="backward_lstm_layer_2") (backward_l1)


        softmax_layer = Dense(vocab_size, activation='softmax')
        
        forward_outputs = softmax_layer(forward_l2)
        backward_outputs = softmax_layer(backward_l2)

        model = Model(inputs=[forward_inputs,backward_inputs], outputs=[forward_outputs,backward_outputs], name="ELMoLike")


        return model

    def __model_4(self):

        vocab_size = len(self.vocabulary)

        forward_inputs = Input(shape=((None,)))
        backward_inputs = Input(shape=((None,)))

        embedding = Embedding(input_dim = vocab_size+1, output_dim = self.embedding_size, mask_zero=True, name="embedding_layer")

        forward_inputs_embedded = embedding(forward_inputs)
        backward_inputs_embedded = embedding(backward_inputs)

        # forward
        forward_l1 = LSTM(self.embedding_size, return_sequences=True, name="forward_lstm_layer_1") (forward_inputs_embedded)

        if self.residual:
            forward_l1 = Add(name="forward_residual")([forward_l1, forward_inputs_embedded])

        #forward_l1 = BatchNormalization()(forward_l1)
        forward_l2 = LSTM(self.embedding_size, return_sequences=False, name="forward_lstm_layer_2") (forward_l1)


        # backward
        backward_l1 = LSTM(self.embedding_size, 
                            return_sequences=True, 
                            #go_backwards=True, 
                            name="backward_lstm_layer_1"
        ) (backward_inputs_embedded)

        if self.residual:
            backward_l1 = Add(name="backward_residual")([backward_l1, backward_inputs_embedded])

        #backward_l1 = BatchNormalization()(backward_l1)
        backward_l2 = LSTM(self.embedding_size, 
                            return_sequences=False, 
                            #go_backwards=True, 
                            name="backward_lstm_layer_2"
        ) (backward_l1)

        x = concatenate([forward_l2,backward_l2])

        output = Dense(vocab_size, activation='softmax') (x)

        model = Model(inputs=[forward_inputs,backward_inputs], outputs=output, name="ELMoLike")


        return model
    
    
    def __model_5(self):

        vocab_size = len(self.vocabulary)

        forward_inputs = Input(shape=((None,)))
        backward_inputs = Input(shape=((None,)))

        embedding = Embedding(input_dim = vocab_size+1, output_dim = self.embedding_size, mask_zero=True, name="embedding_layer")

        forward_inputs_embedded = embedding(forward_inputs)
        backward_inputs_embedded = embedding(backward_inputs)

        # forward
        forward_l1 = LSTM(self.embedding_size, return_sequences=True, name="forward_lstm_layer_1") (forward_inputs_embedded)

        if self.residual:
            forward_l1 = Add(name="forward_residual")([forward_l1, forward_inputs_embedded])

        #forward_l1 = BatchNormalization()(forward_l1)
        forward_l2 = LSTM(self.embedding_size, return_sequences=False, name="forward_lstm_layer_2") (forward_l1)


        # backward
        backward_l1 = LSTM(self.embedding_size, 
                            return_sequences=True, 
                            #go_backwards=True, 
                            name="backward_lstm_layer_1"
        ) (backward_inputs_embedded)

        if self.residual:
            backward_l1 = Add(name="backward_residual")([backward_l1, backward_inputs_embedded])

        #backward_l1 = BatchNormalization()(backward_l1)
        backward_l2 = LSTM(self.embedding_size, 
                            return_sequences=False, 
                            #go_backwards=True, 
                            name="backward_lstm_layer_2"
        ) (backward_l1)

        x = concatenate([forward_l2,backward_l2])

        output = Dense(vocab_size, activation='softmax') (x)

        model = Model(inputs=[forward_inputs,backward_inputs], outputs=output, name="ELMoLike")


        return model

    def compile(self):
        self.model = self.__model_4()


        picture_name = "ELMo_4.png"
        picture_path = os.path.join("", picture_name)

        plot_model(self.model, show_shapes = True, to_file = picture_path)


        self.model.compile(loss = 'sparse_categorical_crossentropy', 
                                        optimizer = tf.keras.optimizers.Adam(),
                                        #metrics = ['sparse_categorical_accuracy']
                                        metrics = [perplexity]
        )

        # print summary
        print(self.model.summary())

    def train(self, best_model_path = None, patience = 20):

        if best_model_path != None:
            self.best_model_path = best_model_path

        # simple early stopping
        #es = EarlyStopping(monitor = 'val_time_distributed_1_loss', mode = 'min', verbose = 1, patience = 20)
        #mc = ModelCheckpoint(best_model_path, monitor = 'val_time_distributed_1_sparse_categorical_accuracy', mode = 'max', verbose = 1, save_best_only = True)

        es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = patience)
        #mc = ModelCheckpoint(self.best_model_path, monitor = 'val_dense_sparse_categorical_accuracy', mode = 'max', verbose = 1, save_best_only = True)
        mc = ModelCheckpoint(self.best_model_path, monitor = 'val_perplexity', mode = 'min', verbose = 1, save_best_only = True)
        
        cbs = [mc,es]

        #self.model.fit([self.forward_inputs, self.backward_inputs], 
        #                [self.forward_outputs, self.backward_outputs], 
        #                epochs = self.nb_epoch,
        #                batch_size=self.batch_size, 
        #                verbose=self.verbose,
        #                callbacks=cbs, 
        #                validation_split=0.2, 
        #                shuffle=True
        #)

        #self.model.fit(self.forward_inputs, 
        #                self.forward_outputs, 
        #                epochs = self.nb_epoch,
        #                batch_size=self.batch_size, 
        #                verbose=self.verbose,
        #                callbacks=cbs, 
        #                validation_split=0.2, 
        #                shuffle=True
        #)


        self.model.fit([self.forward_inputs, self.backward_inputs], 
                        self.forward_outputs, 
                        epochs = self.nb_epoch,
                        batch_size=self.batch_size, 
                        verbose=self.verbose,
                        callbacks=cbs, 
                        validation_split=0.2, 
                        shuffle=True
        )

    def __elmo_model(self, model):

        vocab_size = len(self.vocabulary)

        sentence_inputs = Input(shape=((None,)))

        embedding_weight = model.get_layer("embedding_layer").get_weights()[0]
        forward_lstm_layer_1_weight = model.get_layer("forward_lstm_layer_1").get_weights()
        backward_lstm_layer_1_weight = model.get_layer("backward_lstm_layer_1").get_weights()
        forward_lstm_layer_2_weight = model.get_layer("forward_lstm_layer_2").get_weights()
        backward_lstm_layer_2_weight = model.get_layer("backward_lstm_layer_2").get_weights()

        emb_dim = self.embedding_size

        embedding_layer = Embedding(input_dim = vocab_size+1, output_dim = emb_dim, mask_zero = True, weights = [embedding_weight], trainable = False) (sentence_inputs)

        # forward
        forward_l1_lstm = LSTM(emb_dim, return_sequences=True, weights = forward_lstm_layer_1_weight, trainable = False, name="forward_lstm_layer_1") (embedding_layer)

        forward_l1 = Add(name="forward_residual")([forward_l1_lstm, embedding_layer])

        #forward_l1 = BatchNormalization()(forward_l1)
        forward_l2 = LSTM(emb_dim, return_sequences=True, weights = forward_lstm_layer_2_weight, trainable = False, name="forward_lstm_layer_2") (forward_l1)


        # backward
        backward_l1_lstm = LSTM(emb_dim, 
        return_sequences=True, 
        weights = backward_lstm_layer_1_weight, 
        #go_backwards=True, 
        trainable = False, 
        name="backward_lstm_layer_1"
        ) (embedding_layer)

        backward_l1 = Add(name="backward_residual")([backward_l1_lstm, embedding_layer])

        #backward_l1 = BatchNormalization()(backward_l1)
        backward_l2 = LSTM(emb_dim, 
        return_sequences=True, 
        weights = backward_lstm_layer_2_weight, 
        #go_backwards=True, 
        trainable = False, 
        name="backward_lstm_layer_2"
        ) (backward_l1)

        x2 = concatenate([forward_l2,backward_l2])

        gap_layer = GlobalAveragePooling1D()(x2)
        #gap_layer = GlobalMaxPooling1D()(x2)

        self.elmo_model = tf.keras.models.Model(inputs=sentence_inputs,outputs=[gap_layer, embedding_layer, forward_l1_lstm, backward_l1_lstm, forward_l2, backward_l2], name="ELMO_model")


    def get_elmo_embedding_vectors(self):
    
        vocabulary_index = list(self.vocabulary.values())

        if self.elmo_model == None:
            self.__elmo_model(self.model)

        #Calling the progress bar in the jupyter noteboook
        with tqdm(total=len(vocabulary_index), desc="Get ELMo embedding vectors") as pbar:

            for voc_index in vocabulary_index:
                #print(voc_i)
                input_voc = np.array([[voc_index]])
                
                pred = self.elmo_model.predict([input_voc])
                
                # get outpus from different layers
                v1 = pred[1][0][0]
                v2 = pred[2][0][0]
                v3 = pred[3][0][0]
                v4 = pred[4][0][0]
                v5 = pred[5][0][0]
                
                vect = np.concatenate((v1, v2, v3, v4, v5), axis=0)
                
                self.vectors.append(vect)
                pbar.update(1)

            self.vectors = dict(zip(list(self.vocabulary.keys()), np.array(self.vectors)))

    
    def get_elmo_sentence_embedding(self, sentence, use_multiprocessing=True, workers = 32, batch_size = 2048):
        
        """
        Embed an integer sentence into a vector

        Parameters:
        -----------
        sentence (int array): integer sentence

        Returns:
        --------
        array: Embedded sentence

        """
        if self.elmo_model == None:
            self.__elmo_model(self.model)

        pred = self.elmo_model.predict(sentence, use_multiprocessing=use_multiprocessing, workers = workers, batch_size = batch_size)

        return pred[0][0]


    def get_elmo_embedding_layer(self, embedding_type = "concat", trainable = False, mask_zero = True):

        vocab_size = len(self.vocabulary)
        emb_dim = self.embedding_size

        sentence_inputs = Input(shape=((None,)))

        embedding_weight = self.model.get_layer("embedding_layer").get_weights()[0]
        forward_lstm_layer_1_weight = self.model.get_layer("forward_lstm_layer_1").get_weights()
        backward_lstm_layer_1_weight = self.model.get_layer("backward_lstm_layer_1").get_weights()
        forward_lstm_layer_2_weight = self.model.get_layer("forward_lstm_layer_2").get_weights()
        backward_lstm_layer_2_weight = self.model.get_layer("backward_lstm_layer_2").get_weights()

        embedding_layer = Embedding(input_dim = vocab_size+1, output_dim = emb_dim, mask_zero = mask_zero, weights = [embedding_weight], trainable = trainable) (sentence_inputs)

        # forward
        forward_l1_lstm = LSTM(emb_dim, 
                                return_sequences=True, 
                                weights = forward_lstm_layer_1_weight, 
                                trainable = trainable, 
                                name="forward_lstm_layer_1"
        ) (embedding_layer)

        forward_l1 = Add(name="forward_residual")([forward_l1_lstm, embedding_layer])

        #forward_l1 = BatchNormalization()(forward_l1)
        forward_l2 = LSTM(emb_dim, 
                            return_sequences=True, 
                            weights = forward_lstm_layer_2_weight, 
                            trainable = trainable, 
                            name="forward_lstm_layer_2"
        ) (forward_l1)


        # backward
        backward_l1_lstm = LSTM(emb_dim, 
                                return_sequences=True, 
                                weights = backward_lstm_layer_1_weight, 
                                #go_backwards=True, 
                                trainable = trainable, 
                                name="backward_lstm_layer_1"
        ) (embedding_layer)

        backward_l1 = Add(name="backward_residual")([backward_l1_lstm, embedding_layer])

        #backward_l1 = BatchNormalization()(backward_l1)
        backward_l2 = LSTM(emb_dim, 
                            return_sequences=True, 
                            weights = backward_lstm_layer_2_weight, 
                            #go_backwards=True, 
                            trainable = trainable, 
                            name="backward_lstm_layer_2"
        ) (backward_l1)


        # Select output type
        if embedding_type == "concat":

            output = concatenate([embedding_layer, forward_l1_lstm, backward_l1_lstm, forward_l2, backward_l2])
        
        elif embedding_type == "sum":

            l0 = concatenate([embedding_layer, embedding_layer])
            l1 = concatenate([forward_l1_lstm, backward_l1_lstm])
            l2 = concatenate([forward_l2, backward_l2])

            output = Add()([l0, l1, l2])

        elif embedding_type == "global":

            l2 = concatenate([forward_l2,backward_l2])

            output = GlobalAveragePooling1D()(l2)
            #output = GlobalMaxPooling1D()(22)
        
        elif embedding_type == "last":

            output = concatenate([forward_l2,backward_l2])
        
        elif embedding_type == "multi":

            l0 = concatenate([embedding_layer, embedding_layer])
            l1 = concatenate([forward_l1_lstm, backward_l1_lstm])
            l2 = concatenate([forward_l2, backward_l2])

            output = [l0, l1, l2]


        elmo_embedding = tf.keras.models.Model(inputs=sentence_inputs, outputs=output, name="ELMO_embedding")

        #elmo_embedding = output

        return elmo_embedding


    def __load_vocabulay_file(self, vocab_filename):
        with open(vocab_filename) as json_file:
            self.vocabulary = json.load(json_file)


    def __save_vocabulay_file(self, vocab_filename):
        with open(path,"w") as save_vocab_file:
            json.dump(self.vocabulary, save_vocab_file, indent = 4)


    def get_vector_word(self, word):
        
        return self.vectors[word]


    def get_vectors_for_vocabulary(self, vocabulary, return_numb_oov = False):

        vectors = []

        cpt_oov = 0

        for word in vocabulary:
            
            if word in self.vocabulary:
                vectors.append(np.array(self.vectors[word], dtype='float32'))
            
            # If word does not exist use a zeros vector
            else:
                vectors.append(np.array(np.zeros(self.embedding_size), dtype='float32'))
                cpt_oov += 1

        #print("NUMBER OF OOV: {}".format(cpt_oov))
        #input("Press Enter to continue...")

        if return_numb_oov:
            return cpt_oov, np.array(vectors, dtype='float32') 
        else:
            return np.array(vectors, dtype='float32')

    def save_model(self, model_path):

        tf.keras.models.save_model(model_path+"_elmo_model.h5")
        self.__save_vocabulay_file(model_path+"_elmo_dict_vocabulary.json")


    def load_model(self, model_path):
        
        # load the best model
        self.model = tf.keras.models.load_model(model_path+"_elmo_model.h5", custom_objects={"perplexity": perplexity})

        # load vocab dictionary
        self.__load_vocabulay_file(model_path+"_elmo_dict_vocabulary.json")

        # get vector size
        embedding_weight = self.model.get_layer("embedding_layer").get_weights()[0]
        self.embedding_size = len(embedding_weight[1])

        
        self.__elmo_model(self.model)




