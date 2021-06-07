# !/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Concatenate, Embedding, Input
from tensorflow.keras.models import Sequential, Model


def uniLSTM(vocab_size, nb_classes, output_dim=[64,64], max_lenght=2000, mask_zero=True):

	model = Sequential(name='Liciotti_LSTM')
	model.add(Embedding(input_dim = vocab_size+1, output_dim = output_dim[0], input_length=max_lenght, mask_zero=mask_zero))
	model.add(LSTM(output_dim[1]))
	model.add(Dense(nb_classes, activation='softmax'))

	return model

def biLSTM(vocab_size, nb_classes, output_dim=[64,64], max_lenght=2000, mask_zero=True):

	model = Sequential(name='Liciotti_BiLSTM')
	model.add(Embedding(input_dim = vocab_size+1, output_dim = output_dim[0], input_length=max_lenght, mask_zero=mask_zero))
	model.add(Bidirectional(LSTM(output_dim[1])))
	model.add(Dense(nb_classes, activation='softmax'))

	return model


def ensemble2LSTM(vocab_size, nb_classes, output_dim=[64,64,64,64], max_lenght=2000, mask_zero=True):

	input_layer_1 = Input(shape=((max_lenght,)))
	m1 = Embedding(input_dim = vocab_size+1, output_dim = output_dim[0], input_length=max_lenght, mask_zero=True) (input_layer_1)
	m1 = Bidirectional(LSTM(output_dim[1])) (m1)


	input_layer_2 = Input(shape=((max_lenght,)))
	m2 = Embedding(input_dim = vocab_size+1, output_dim = output_dim[2], input_length=max_lenght, mask_zero=True)(input_layer_2)
	m2 = LSTM(output_dim[3])(m2)

	
	m3 = Concatenate()([m1, m2])
	output_layer = Dense(nb_classes, activation='softmax') (m3)

	model = Model(inputs=[input_layer_1,input_layer_2], outputs=output_layer, name="Liciotti_Ensemble2LSTM")

	return model


def cascadeEnsembleLSTM(vocab_size, nb_classes, output_dim=[64,64,64,64,64], max_lenght=2000, mask_zero=True):

	
	input_layer_1 = Input(shape=((max_lenght,)))
	m1 = Embedding(input_dim = vocab_size+1, output_dim = output_dim[0], input_length=max_lenght, mask_zero=True) (input_layer_1)
	m1 = Bidirectional(LSTM(output_dim[1], return_sequences=True)) (m1)


	input_layer_2 = Input(shape=((max_lenght,)))
	m2 = Embedding(input_dim = vocab_size+1, output_dim = output_dim[2], input_length=max_lenght, mask_zero=True)(input_layer_2)
	m2 = LSTM(output_dim[3], return_sequences=True)(m2)

	
	m3 = Concatenate()([m1, m2])
	m3 = LSTM(output_dim[4]) (m3)
	output_layer = Dense(nb_classes, activation='softmax') (m3)

	model = Model(inputs=[input_layer_1,input_layer_2], outputs=output_layer, name="Liciotti_CascadeEnsembleLSTM")

	return model


def cascadeLSTM(vocab_size, nb_classes, output_dim=[64,64,64], max_lenght=2000, mask_zero=True):

	model = Sequential(name='Liciotti_CascadeLSTM')
	model.add(Embedding(input_dim = vocab_size+1, output_dim = output_dim[0], input_length=max_lenght, mask_zero=mask_zero))
	model.add(Bidirectional(LSTM(output_dim[1], return_sequences=True)))
	model.add(LSTM(output_dim[2]))
	model.add(Dense(nb_classes, activation='softmax'))

	return model


def compile_model(model):
	model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model