# !/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Merge
from tensorflow.keras.embeddings import Embedding
from tensorflow.keras.models import Sequential


def LSTM(vocab_size, nb_classes, output_dim=64, max_lenght=2000, mask_zero=True):

	model = Sequential(name='Liciotti_LSTM')
	model.add(Embedding(input_dim = vocab_size+1, output_dim = output_dim, input_length=max_lenght, mask_zero=mask_zero))
	model.add(LSTM(output_dim))
	model.add(Dense(nb_classes, activation='softmax'))

	return model

def biLSTM(vocab_size, nb_classes, output_dim=64, max_lenght=2000, mask_zero=True):

	model = Sequential(name='Liciotti_BiLSTM')
	model.add(Embedding(input_dim = vocab_size+1, output_dim = output_dim, input_length=max_lenght, mask_zero=mask_zero))
	model.add(Bidirectional(LSTM(output_dim)))
	model.add(Dense(nb_classes, activation='softmax'))

	return model


def ensemble2LSTM(vocab_size, nb_classes, output_dim=64, max_lenght=2000, mask_zero=True):

	model1 = Sequential()
	model1.add(Embedding(input_dim = vocab_size+1, output_dim = output_dim, input_length=max_lenght, mask_zero=mask_zero))
	model1.add(Bidirectional(LSTM(output_dim)))

	model2 = Sequential()
	model2.add(Embedding(input_dim = vocab_size+1, output_dim = output_dim, input_length=max_lenght, mask_zero=mask_zero))
	model2.add(LSTM(output_dim))

	model = Sequential(name='Liciotti_Ensemble2LSTM')
	model.add(Merge([model1, model2], mode='concat'))
	model.add(Dense(nb_classes, activation='softmax'))

	return model


def cascadeEnsembleLSTM(vocab_size, nb_classes, output_dim=64, max_lenght=2000, mask_zero=True):

	model1 = Sequential()
	model1.add(Embedding(input_dim = vocab_size+1, output_dim = output_dim, input_length=max_lenght, mask_zero=True))
	model1.add(Bidirectional(LSTM(output_dim, return_sequences=True)))

	model2 = Sequential()
	model2.add(Embedding(input_dim = vocab_size+1, output_dim = output_dim, input_length=max_lenght, mask_zero=True))
	model2.add(LSTM(output_dim, return_sequences=True))

	model = Sequential(name='Liciotti_CascadeEnsembleLSTM')
	model.add(Merge([model1, model2], mode='concat'))
	model.add(LSTM(output_dim))
	model.add(Dense(nb_classes, activation='softmax'))

	return model


def cascadeLSTM(vocab_size, nb_classes, output_dim=64, max_lenght=2000, mask_zero=True):

	model = Sequential(name='Liciotti_CascadeLSTM')
	model.add(Embedding(input_dim = vocab_size+1, output_dim = output_dim, input_length=max_lenght, mask_zero=mask_zero))
	model.add(Bidirectional(LSTM(output_dim, return_sequences=True)))
	model.add(LSTM(output_dim))
	model.add(Dense(nb_classes, activation='softmax'))

	return model


def compile_model(model):
	model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model