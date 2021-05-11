#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import re

from .datasetEncoder import DatasetEncoder

class DatasetSegmentator(object):

	def __init__(self, datasetEncoder):
		
		if not isinstance(datasetEncoder, DatasetEncoder) :
			raise ValueError('Argument should be DatasetEncoder type')


		self.segmentationType = None
		self.encoder = datasetEncoder
		self.df = datasetEncoder.df
		self.encodingType = datasetEncoder.encodingType
		self.eventDict = datasetEncoder.eventDict
		self.actDict = datasetEncoder.actDict
		self.X = datasetEncoder.X
		self.Y = datasetEncoder.Y


	def dayWindows(self):

		#Reinitialize in case
		if self.segmentationType != None:
			self.__init__(self.encoder)


		self.segmentationType = "Day"


		dofw = []

		chunksX = []
		chunksY = []


		transitionIndex = self.df.datetime.dt.date.ne(self.df.datetime.dt.date.shift())

		ii = np.where(transitionIndex == True)[0]

		for i,end in enumerate(ii):
			if i > 0 :

				start = ii[i-1]
				#print("START: {}".format(start))
				#print("END: {}".format(end))
				#input("Press Enter to continue...")

				day = self.df[start:end]
				dofw.append(day.datetime.dt.dayofweek.values[-1])
				
				chunksX.append(self.X[start:end])
				chunksY.append(self.Y[start:end])



		lastDay = self.df[end:]
		dofw.append(lastDay.datetime.dt.dayofweek.values[-1])
		
		chunksX.append(self.X[end:])
		chunksY.append(self.Y[end:])


		self.X = chunksX
		self.Y = chunksY


		
		return dofw


	def explicitWindow(self):
		raise NotImplementedError


	def slidingSensorEventWindows(self,winSize,step=1):
		raise NotImplementedError


	def sensorEventWindows(self,winSize):
		raise NotImplementedError


	def slidingTimeWindow(self, interval,step=1):
		raise NotImplementedError


	def timeWindow(self, interval):
		raise NotImplementedError