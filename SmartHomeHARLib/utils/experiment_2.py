# coding: utf-8
# !/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import csv
import pickle
import time


class Experiment(object):

	def __init__(self, dataset, experimentParameters=None, modelParameters=None):
		
		self.DEBUG = False

		self.dataset = dataset
		
		self.data_X_train = None
		self.data_Y_train = None
		self.data_X_val = None
		self.data_Y_val = None
		self.data_X_test = None
		self.data_Y_test = None

		self.modelParameters = modelParameters
		self.experimentParameters = experimentParameters
		
		self.model = None

		self.experimentPath = None

		self.currenttime  = None
		
		self.datasetEncoder = None
		
		self.datasetSegmentator = None

	def setDebugMode(self, activate):

		self.DEBUG = activate

		if self.DEBUG:
			self.experimentParameters["nb_epochs"] = 1

	def checkInputModel(self, runNumber=0):

		X_val_input = None
		Y_val_input = None

		if self.DEBUG:
			print(self.data_X_train.shape)
			print(self.data_X_test.shape)
			if self.data_X_val != None:
				print(self.data_X_val.shape)
			else:
				print("None")
			input("Press Enter to continue...")
		
		# Check number size of exemples
		if len(self.data_X_train) < 2:
			data_X_train = self.data_X_train[0]
			data_Y_train = self.data_Y_train[0]
		else:
			data_X_train = self.data_X_train[runNumber]
			data_Y_train = self.data_Y_train[runNumber]


		if len(self.data_X_test) < 2:
			data_X_test = self.data_X_test[0]
			data_Y_test = self.data_Y_test[0]
		else:
			data_X_test = self.data_X_test[runNumber]
			data_Y_test = self.data_Y_test[runNumber]


		if self.data_X_val != None:
			if len(self.data_X_val) < 2:
				data_X_val = self.data_X_val[0]
				data_Y_val = self.data_Y_val[0]
			else:
				data_X_val = self.data_X_val[runNumber]
				data_Y_val = self.data_Y_val[runNumber]


		# Nb features depends on data shape
		if data_X_train.ndim > 2:
			nb_features = data_X_train.shape[2]
		else:
			nb_features = 1


		if 'Ensemble' in self.experimentParameters["model_type"]:
			
			X_train_input = [data_X_train, data_X_train]
			X_test_input = [data_X_test, data_X_test]
			
			if self.data_X_val != None:
				X_val_input = [data_X_val, data_X_val]


		elif 'Vanilla' in self.experimentParameters["model_type"] or 'Deep1DCNN' in self.experimentParameters["model_type"]:

			if data_X_train.ndim < 3:

				if self.DEBUG:
					print("Dimension < 3")
				
				X_train_input = data_X_train.reshape(data_X_train.shape[0],data_X_train.shape[1],1)
				X_test_input = data_X_test.reshape(data_X_test.shape[0],data_X_test.shape[1],1)

				if self.data_X_val != None:
					X_val_input = data_X_val.reshape(data_X_val.shape[0],data_X_val.shape[1],1)

			else:
				
				if self.DEBUG:
					print("Dimension > 2")

				X_train_input = data_X_train
				X_test_input = data_X_test

				if self.data_X_val != None:
					X_val_input = data_X_val

		elif 'Double' in self.experimentParameters["model_type"]:
			if self.DEBUG:
				print(len(data_X_train[:,:,0]))
				print(len(data_X_train[:,:,1]))
				print(data_X_train.shape)

			X_train_input = [data_X_train[:,:,0], data_X_train[:,:,1]]
			X_test_input = [data_X_test[:,:,0], data_X_test[:,:,1]]

			if self.data_X_val != None:
				X_val_input = [data_X_val[:,:,0], data_X_val[:,:,1]]

		else:
			
			X_train_input = data_X_train
			X_test_input = data_X_test

			if self.data_X_val != None:
				X_val_input = data_X_val


		Y_train_input = data_Y_train
		Y_test_input = data_Y_test

		if self.data_X_val != None:
			Y_val_input = data_Y_val


		if self.DEBUG:
			print("Train {}:".format(np.array(X_train_input).shape))
			print("Test : {}".format(np.array(X_test_input).shape))

			if self.data_X_val != None:
				print("Val : {}".format(np.array(X_val_input).shape))
			else:
				print("Val : None")

			input("Press Enter to continue...")

		return X_train_input,Y_train_input , X_val_input, Y_val_input, X_test_input, Y_test_input, nb_features

	def buildModel(self):

		vocab_size = self.modelParameters['vocab_size']
		output_dim = self.modelParameters['output_dim']
		nb_timesteps = self.modelParameters['nb_timesteps']
		nb_features = self.modelParameters['nb_features']
		nb_classes = self.modelParameters['nb_classes']
		nb_head = self.modelParameters['nb_head']
		ff_dim = self.modelParameters['ff_dim']


		with strategy.scope():
			if self.modelType == 'VanillaLSTM':
				self.model = vanillaLSTM(output_dim, nb_timesteps, nb_features, nb_classes)

			elif self.modelType == 'VanillaLSTMBatchNorm':
				self.modelmodel = vanillaLSTMBatchNorm(output_dim, nb_timesteps, nb_features, nb_classes)

			elif self.modelType == 'VanillaLSTMBatchNorm2':
				self.modelmodel = vanillaLSTMBatchNorm2(nb_timesteps, nb_features, nb_classes)

			elif self.modelType == 'VanillaFCN':
				self.modelmodel = FCN(nb_timesteps, nb_features, nb_classes)

			elif self.modelType == 'VanillaFCNBatchNorm':
				self.modelmodel = FCNBatchNorm(nb_timesteps, nb_features, nb_classes)

			elif self.modelType == 'FourEmbeddededConcatLSTM':
				self.modelmodel = fourEmbeddededConcatLSTM(output_dim, nb_timesteps, nb_classes)

			elif self.modelType == 'FourEmbeddededAddLSTM':
				self.modelmodel = fourEmbeddedAddLSTM(output_dim, nb_timesteps, nb_classes)

			elif self.modelType == 'EmbLSTM':
				self.modelmodel = embeddededLSTM(vocab_size, output_dim, nb_timesteps, nb_classes)

			elif self.modelType == 'biLSTM':
				self.modelmodel = biLSTM(vocab_size, output_dim, nb_timesteps, nb_classes)

			elif self.modelType == 'Ensemble2LSTM':
				self.modelmodel = ensemble2LSTM(vocab_size, output_dim, nb_timesteps, nb_classes)

			elif self.modelType == 'CascadeEnsembleLSTM':
				self.modelmodel = cascadeEnsembleLSTM(vocab_size, output_dim, nb_timesteps, nb_classes)

			elif self.modelType == 'CascadeLSTM':
				self.modelmodel = cascadeLSTM(vocab_size, output_dim, nb_timesteps, nb_classes)

			elif self.modelType == 'AttResBiLSTM6':
				self.modelmodel = attentionResidualBiLSTM6(vocab_size, output_dim, nb_timesteps, nb_classes)

			elif self.modelType == 'EmbFCN':
				self.modelmodel = embeddedFCN(vocab_size, nb_timesteps, nb_classes, output_dim)

			elif self.modelType == 'EmbFCNbiLSTM':
				self.modelmodel = embeddedFCNbiLSTM(vocab_size, nb_timesteps, nb_classes, output_dim)

			elif self.modelType == 'EmbFCNBatchInp':
				self.modelmodel = embeddedFCNBatchNormInput(vocab_size, nb_timesteps, nb_classes, output_dim)

			elif self.modelType == 'EmbFCNBatchNormEmb':
				self.modelmodel = embeddedFCNBatchNormEmbedding(vocab_size, nb_timesteps, nb_classes, output_dim)

			elif self.modelType == 'EmbFCNDeep':
				self.modelmodel = embeddedFCNDeep(vocab_size, nb_timesteps, nb_classes, output_dim)

			elif self.modelType == 'Deep1DCNN':
				self.modelmodel = deep1DCNN(nb_timesteps, nb_features, nb_classes)

			elif self.modelType == 'EmbDeep1DCNN':
				self.modelmodel = embeddedDeep1DCNN(vocab_size, nb_timesteps, nb_classes, output_dim)

			elif self.modelType == 'TransformerFCN':
				#model = transformer1Layer(vocab_size, max_lenght, nb_classes, emb_dim=64, num_heads = 8, ff_dim = 64)
				self.modelmodel = transformerFCN(vocab_size, nb_timesteps, nb_classes, output_dim, num_heads, ff_dim)

			elif self.modelType == 'TransformerFCN_2':
				#model = transformer1Layer(vocab_size, max_lenght, nb_classes, emb_dim=64, num_heads = 8, ff_dim = 64)
				self.modelmodel = transformerFCN_2(vocab_size, nb_timesteps, nb_classes, output_dim, num_heads, ff_dim)

			elif self.modelType == 'FCNTransformer':
				#model = transformer1Layer(vocab_size, max_lenght, nb_classes, emb_dim=64, num_heads = 8, ff_dim = 64)
				self.modelmodel = FCNTransformer(vocab_size, nb_timesteps, nb_classes, output_dim, num_heads, ff_dim)

			elif self.modelType == 'Transformer3Layers':
				self.modelmodel = transformer3Layers(vocab_size, nb_timesteps, nb_classes, output_dim, num_heads, ff_dim)

			elif self.modelType == 'DoubleEmbFCNAdd':
				self.modelmodel = embeddedFCNx2Add(vocab_size, nb_timesteps, nb_classes, output_dim)

			elif self.modelType == 'FiveEmbFCNAdd':
				self.modelmodel = fiveEmbeddedFCNadd(vocab_size, nb_timesteps, nb_classes, output_dim)

			elif self.modelType == 'FiveEmbFCNconcat':
				self.modelmodel = fiveEmbeddedFCNconcat(vocab_size, nb_timesteps, nb_classes, output_dim)

			elif self.modelType == 'FiveEmbLSTMAdd':
				self.modelmodel = fiveEmbeddedAddLSTM(vocab_size, output_dim, nb_timesteps, nb_classes)

			elif self.modelType == 'FiveEmbLSTMconcat':
				self.modelmodel = fiveEmbeddedConcatLSTM(vocab_size, output_dim, nb_timesteps, nb_classes)

			else:
				raise('Please get the model name (VanillaLSTM | EmbLSTM | biLSTM | Ensemble2LSTM | CascadeEnsembleLSTM | CascadeLSTM | EmbFCN | Deep1DCNN | EmbDeep1DCNN)')

	def start(self):

		#Star time of the experiment
		self.currenttime = time.strftime("%Y_%m_%d_%H_%M_%S")

		self.experimentPath = os.path.join(self.experimentParameters["name"], self.experimentParameters["model_type"], "run_"+ self.experimentTag + "_" + str(self.currenttime))

		# create a folder with the model name
		# if the folder doesn't exist
		if not os.path.exists(self.experimentPath):
			os.makedirs(self.experimentPath)

		self.prepareDataset()

		#Split the dataset into train, val and test examples
		self.splitDataset()

		nb_runs = len(self.data_X_train)

		if self.DEBUG:
			print("NB RUN: {}".format(nb_runs))
		
		for runNumber in range(nb_runs):

			#prepare input according to the model type
			X_train_input, Y_train_input , X_val_input, Y_val_input, X_test_input, Y_test_input, nb_features = self.checkInputModel(runNumber)
			
			self.buildModel(nb_features)

			#compile the model
			self.compileModel()

			self.trainModel(X_train_input, Y_train_input, X_val_input, Y_val_input, runNumber)

			self.evaluateModel(X_test_input, Y_test_input, runNumber)


	def prepareDataset(self):
		raise NotImplementedError

	def splitDataset(self):
		raise NotImplementedError

	def compileModel(self):
		raise NotImplementedError

	def trainModel(self, X_train_input, Y_train_input, X_val_input, Y_val_input, runNumber=0):
		raise NotImplementedError

	def evaluateModel(self, X_test_input, Y_test_input, runNumber=0):
		raise NotImplementedError


	def saveConfig(self):

		experiment_parameters_name = "experiment_parameters.pickle"
		experiment_parameters_path = os.path.join(self.experimentPath, experiment_parameters_name)

		with open(experiment_parameters_path,"wb") as pickle_out:
			pickle.dump(self.experimentParameters, pickle_out)


		model_parameters_name = "model_parameters.pickle"
		model_parameters_path = os.path.join(self.experimentPath, model_parameters_name)

		with open(model_parameters_path,"wb") as pickle_out:
			pickle.dump(self.modelParameters, pickle_out)