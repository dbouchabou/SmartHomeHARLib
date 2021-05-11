#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import re

from .smartHomeDataset import SmartHomeDataset

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

SEED = 7

np.random.seed(SEED)
tf.random.set_seed(SEED)


class DatasetEncoder(object):

	def __init__(self, dataset):
		
		if not isinstance(dataset, SmartHomeDataset) :
			raise ValueError('Argument should be SmartHomeDataset type')

		self.dataset = dataset
		self.df = dataset.df.copy()
		self.encodingType = None
		self.eventDict = None
		self.actDict = None
		self.X = []
		self.Y = []

		self.__generateActDict()


	def __generateActDict(self):

		activityList = self.dataset.activitiesList
		activityList.sort()


		self.actDict = {}
		for i, activity in enumerate(activityList):
			self.actDict[activity] = i


	def _encodeColumn(self, val, valDict = None ):

		if valDict == None:
			valList = val.unique()
			valList.sort()

			valDict={}
			for i, v in enumerate(valList):
				valDict[v] = i+1	#add one to reserve 0 value to padding

		encodedVal = val.map(valDict)

		return valDict, encodedVal


	def __indexLabels(self):
		raise NotImplementedError
