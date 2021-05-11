#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
import re

sys.path.insert(0, "/home/d19bouch/workspace")
from SmartHomeHARLib.tools.DatasetEncoder import DatasetEncoder
#from .DatasetEncoder import DatasetEncoder

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

SEED = 7

np.random.seed(SEED)
tf.random.set_seed(SEED)



class CASASDatasetEncoder(DatasetEncoder):

	def __init__(self, dataset):

		if dataset.datasetType != "CASAS" :
			raise ValueError('Argument should be CASASDataset type')

		super().__init__(dataset)

		self.__indexLabels()

		self.Y = self.df.encodedActivity.values.astype(int)


	def __indexLabels(self):

		self.df["encodedActivity"] = self.df.activity.astype(str)

		_, self.df["encodedActivity"] = self._encodeColumn(self.df["encodedActivity"], self.actDict)


	def basic(self):

		#set the encoding type
		self.encodingType = "BASIC"



		#create event tokens
		self.df['merge'] =  self.df['sensor'] + self.df['value'].astype(str)

		#encode
		wordDict, self.df['encodedMerge'] = self._encodeColumn(self.df["merge"])

		self.X = self.df['encodedMerge'].values.astype(int)


		#save de word dictionary
		self.eventDict = wordDict


	def nlpBasic(self):

		#set the encoding type
		self.encodingType = "NLPBASIC"

		
		self.df['nlp'] =  self.df['sensor'] + self.df['value'].astype(str)

		sentence = " ".join(self.df['nlp'].values)

		tokenizer = Tokenizer(filters='')
		tokenizer.fit_on_texts([sentence])

		wordDict = tokenizer.word_index

		#encode
		self.df['encodedNLP'] = self.df['nlp'].str.lower()
		#self.df['encodedNLP'] = self.df['encodedNLP'].map(wordDict).astype(str)
		_, self.df['encodedNLP'] = self._encodeColumn(self.df["encodedNLP"],wordDict)


		self.X = self.df['encodedNLP'].values.astype(int)


		#save de word dictionary
		self.eventDict = wordDict


	def nlpMidnight(self):

		#set the encoding type
		self.encodingType = "NLPMIDNIGHT"


		self.df["sinceMidnight"] = self.df['datetime'].dt.hour.astype(int)*3600+self.df['datetime'].dt.minute.astype(int)*60+self.df['datetime'].dt.second.astype(int)
		self.df["sinceMidnight"] = self.df["sinceMidnight"].astype(str)
		self.df["merge"] = self.df['sensor'] + self.df['value'].astype(str)
		self.df['nlp'] =  self.df["merge"] + " " + self.df["sinceMidnight"]

		sentence = " ".join(self.df['nlp'].values)

		tokenizer = Tokenizer(filters='')
		tokenizer.fit_on_texts([sentence])

		wordDict = tokenizer.word_index

		self.df['encodedMerge']=self.df['merge'].str.lower()
		#self.df['encodedMerge'] = self.df['encodedMerge'].map(wordDict)
		_, self.df['encodedMerge'] = self._encodeColumn(self.df["encodedMerge"],wordDict)

		self.df['encodedSinceMidnight'] = self.df['sinceMidnight']
		#self.df['encodedSinceMidnight'] = self.df['encodedSinceMidnight'].map(wordDict)
		_, self.df['encodedSinceMidnight'] = self._encodeColumn(self.df["encodedSinceMidnight"],wordDict)

		#self.df['encodedNLP'] =  self.df["encodedMerge"].astype(str) + " " + self.df["encodedSinceMidnight"].astype(str)


		#self.X = " ".join(self.df['encodedNLP'].values)
		#self.X = self.X.split()
		#self.X = list(map(int, self.X))


		feature_1 = self.df.encodedMerge.values.astype(int)
		feature_2 = self.df.encodedSinceMidnight.values.astype(int)


		self.X = np.array([feature_1,feature_2])
		self.X = self.X.transpose()


		#save de word dictionary
		self.eventDict = wordDict


	def fourHandCrafted(self):
		
		#set the encoding type
		self.encodingType = "4HCFEATURE"


		self.df["dayOfWeek"] = self.df['datetime'].dt.dayofweek.astype(int)+1
		self.df["hour"] = self.df['datetime'].dt.hour.astype(int)+1


		#encoding sensors ID
		sensorsDict, self.df['encodedSensor'] = self._encodeColumn(self.df['sensor'])

		#encoding value sensors
		valuesDict, self.df['encodedValue'] = self._encodeColumn(self.df['value'])


		feature_1 = self.df.dayOfWeek.values.astype(int)
		feature_2 = self.df.hour.values.astype(int)
		feature_3 = self.df.encodedSensor.values.astype(int)
		feature_4 = self.df.encodedValue.values.astype(int)

		self.X = np.array([feature_1,feature_2,feature_3,feature_4])
		self.X = self.X.transpose()


		#save de word dictionary
		self.eventDict = [sensorsDict,valuesDict]
		

	def __directPreviousActivity(self):

		#previous act
		NaN = np.nan
		self.df["prevAct"] = NaN

		pa = self.df.prevAct.values.astype(object)


		ponentialIndex = self.df.activity.ne(self.df.activity.shift())

		ii = np.where(ponentialIndex == True)[0]


		for i,transiInd in enumerate(ii):
			if i > 0 :
				prevA = self.df[transiInd-1:transiInd].activity.values[-1]
				currentA = self.df[transiInd:transiInd+1].activity.values[-1]

				#print("previous act {}".format(prevA))
				#print("current act {}".format(currentA))

				pa[transiInd] = prevA

		prevA = self.df[ii[-1]-1:ii[-1]].activity.values[-1]
		currentA = self.df[ii[-1]:ii[-1]+1].activity.values[-1]

		#print("previous act {}".format(prevA))
		#print("current act {}".format(currentA))

		pa[ii[-1]] = prevA

		self.df["prevAct"] = pa

		self.df.prevAct = self.df.prevAct.fillna(method='ffill')

		self.df.prevAct = self.df.prevAct.fillna(method='bfill')


	def __previousActivityNoOther(self):

		#previous act
		act =  self.df.activity.values

		an_array = np.empty(len(act), dtype=object)
		prevAct = []


		ponentialIndex = self.df.activity.ne(self.df.activity.shift())

		ii = np.where(ponentialIndex == True)[0]

		for i,end in enumerate(ii):
			if i > 0 :
				#print(self.df[ii[i-1]:ii[i]].activity.values[-1])
				prevAct.append(self.df[ii[i-1]:ii[i]].activity.values[-1])
		#print(self.df[ii[i]:].activity.values[-1])        
		prevAct.append(self.df[ii[i]:].activity.values[-1])


		noOther = list(filter(lambda a: a != "Other", prevAct))

		noOther.insert(0, 'Other')


		lastReslAct = 0
		for i in ii:
			#print(i)
			#print(lastReslAct)
			an_array[i] =  noOther[lastReslAct]
			if act[i] != "Other":
				lastReslAct+=1

		self.df["prevAct"] = an_array

		self.df.prevAct = self.df.prevAct.fillna(method='ffill')


	def __previousActivityAsCodePaper(self):

		#previous act
		self.df["prevAct"] = self.df.activity


	def __directPreviousEventActivity(self):

		#previous act
		self.df["prevAct"] =  self.df.activity.shift()

		self.df.prevAct = self.df.prevAct.fillna(method='bfill')


	def fiveHandCrafted(self):
		#set the encoding type
		self.encodingType = "5HCFEATURE"


		self.df["dayOfWeek"] = self.df['datetime'].dt.dayofweek.astype(int)+1
		self.df["hour"] = self.df['datetime'].dt.hour.astype(int)+1


		#generate previous activity
		self.__previousAct()


		#encoding sensors ID
		sensorsDict, self.df['encodedSensor'] = self._encodeColumn(self.df['sensor'])

		#encoding value sensors
		valuesDict, self.df['encodedValue'] = self._encodeColumn(self.df['value'])

		#encoding previous avtivity
		activityDict, self.df['encodedPrevAct'] = self._encodeColumn(self.df['prevAct'])


		feature_1 = self.df.dayOfWeek.values.astype(int)
		feature_2 = self.df.hour.values.astype(int)
		feature_3 = self.df.encodedSensor.values.astype(int)
		feature_4 = self.df.encodedValue.values.astype(int)
		feature_5 = self.df.encodedPrevAct.values.astype(int)

		self.X = np.array([feature_1,feature_2,feature_3,feature_4,feature_5])
		self.X = self.X.transpose()


		#save de word dictionary
		self.eventDict = [sensorsDict,valuesDict, activityDict]


	def fiveHandCrafted2(self):
		#set the encoding type
		self.encodingType = "5HCFEATURE"


		self.df["dayOfWeek"] = self.df['datetime'].dt.dayofweek.astype(int)
		self.df["hour"] = self.df['datetime'].dt.hour.astype(int)
		self.df["minute"] = self.df['datetime'].dt.minute.astype(int)


		#generate previous activity
		self.__previousAct()


		#encoding sensors ID
		sensorsDict, self.df['encodedSensor'] = self._encodeColumn(self.df['sensor'])

		#encoding value sensors
		valuesDict, self.df['encodedValue'] = self._encodeColumn(self.df['value'])
		#self.df['encodedValue'] = self.df['value'].str.replace('ON','1.0')
		#self.df['encodedValue'] = self.df['value'].str.replace('OFF','0.0')


		#encoding previous avtivity
		activityDict, self.df['encodedPrevAct'] = self._encodeColumn(self.df['prevAct'])


		feature_1 = self.df.dayOfWeek.values.astype(int)
		feature_2 = self.df.hour.values.astype(float)+np.round(self.df.minute.values.astype(float)/60,3)
		feature_3 = self.df.encodedSensor.values.astype(int)
		feature_4 = self.df.encodedValue.values.astype(int)
		feature_5 = self.df.encodedPrevAct.values.astype(int)

		self.X = np.array([feature_1,feature_2,feature_3,feature_4,feature_5])
		self.X = self.X.transpose()


		#save de word dictionary
		self.eventDict = [sensorsDict,valuesDict, activityDict]

	def SurongFiveFeaturesDirectPreviousActivity(self):
		#set the encoding type
		self.encodingType = "SURONG_FIVE_FEATURES_DIRECT_PREVIOUS_ACT"


		self.df["dayOfWeek"] = self.df['datetime'].dt.dayofweek.astype(int)
		self.df["hour"] = self.df['datetime'].dt.hour.astype(int)

		#generate previous activity
		self.__directPreviousActivity()


		#encoding sensors ID
		sensorsDict, self.df['encodedSensor'] = self._encodeColumn(self.df['sensor'])

		#encoding value sensors
		self.df['encodedValue'] = self.df['value'].astype(str)
		self.df['encodedValue'] = self.df['encodedValue'].str.replace('ON','1.0')
		self.df['encodedValue'] = self.df['encodedValue'].str.replace('OFF','-1.0')

		val = self.df['encodedValue']
		valList = val.unique()
		valList.sort()

		valuesDict={}
		for i, v in enumerate(valList):
			valuesDict[v] = i+1	#add one to reserve 0 value to padding


		#encoding previous avtivity
		activityDict, self.df['encodedPrevAct'] = self._encodeColumn(self.df['prevAct'])


		feature_1 = self.df.encodedSensor.values.astype(float)
		feature_2 = self.df.encodedValue.values.astype(float)
		feature_3 = self.df.dayOfWeek.values.astype(float)
		feature_4 = self.df.hour.values.astype(float)
		feature_5 = self.df.encodedPrevAct.values.astype(float)

		self.X = np.array([feature_1,feature_2,feature_3,feature_4,feature_5])
		self.X = self.X.transpose()


		#save de word dictionary
		self.eventDict = [sensorsDict,valuesDict, activityDict]


	def SurongFiveFeaturesPreviousActivityNoOther(self):
		#set the encoding type
		self.encodingType = "SURONG_FIVE_FEATURES_PREVIOUS_ACT_NO_OTHER"


		self.df["dayOfWeek"] = self.df['datetime'].dt.dayofweek.astype(int)
		self.df["hour"] = self.df['datetime'].dt.hour.astype(int)

		#generate previous activity
		self.__previousActivityNoOther()


		#encoding sensors ID
		sensorsDict, self.df['encodedSensor'] = self._encodeColumn(self.df['sensor'])

		#encoding value sensors
		self.df['encodedValue'] = self.df['value'].astype(str)
		self.df['encodedValue'] = self.df['encodedValue'].str.replace('ON','1.0')
		self.df['encodedValue'] = self.df['encodedValue'].str.replace('OFF','-1.0')

		val = self.df['encodedValue']
		valList = val.unique()
		valList.sort()

		valuesDict={}
		for i, v in enumerate(valList):
			valuesDict[v] = i+1	#add one to reserve 0 value to padding


		#encoding previous avtivity
		activityDict, self.df['encodedPrevAct'] = self._encodeColumn(self.df['prevAct'])


		feature_1 = self.df.encodedSensor.values.astype(float)
		feature_2 = self.df.encodedValue.values.astype(float)
		feature_3 = self.df.dayOfWeek.values.astype(float)
		feature_4 = self.df.hour.values.astype(float)
		feature_5 = self.df.encodedPrevAct.values.astype(float)

		self.X = np.array([feature_1,feature_2,feature_3,feature_4,feature_5])
		self.X = self.X.transpose()


		#save de word dictionary
		self.eventDict = [sensorsDict,valuesDict, activityDict]


	def SurongFiveFeaturesPreviousActivityCodePaper(self):
		#set the encoding type
		self.encodingType = "SURONG_FIVE_FEATURES_PREVIOUS_ACT_CODE_PAPER"


		self.df["dayOfWeek"] = self.df['datetime'].dt.dayofweek.astype(int)
		self.df["hour"] = self.df['datetime'].dt.hour.astype(int)

		#generate previous activity
		self.__previousActivityAsCodePaper()


		#encoding sensors ID
		sensorsDict, self.df['encodedSensor'] = self._encodeColumn(self.df['sensor'])

		#encoding value sensors
		self.df['encodedValue'] = self.df['value'].astype(str)
		self.df['encodedValue'] = self.df['encodedValue'].str.replace('ON','1.0')
		self.df['encodedValue'] = self.df['encodedValue'].str.replace('OFF','-1.0')

		val = self.df['encodedValue']
		valList = val.unique()
		valList.sort()

		valuesDict={}
		for i, v in enumerate(valList):
			valuesDict[v] = i+1	#add one to reserve 0 value to padding


		#encoding previous avtivity
		activityDict, self.df['encodedPrevAct'] = self._encodeColumn(self.df['prevAct'])


		feature_1 = self.df.encodedSensor.values.astype(float)
		feature_2 = self.df.encodedValue.values.astype(float)
		feature_3 = self.df.dayOfWeek.values.astype(float)
		feature_4 = self.df.hour.values.astype(float)
		feature_5 = self.df.encodedPrevAct.values.astype(float)

		self.X = np.array([feature_1,feature_2,feature_3,feature_4,feature_5])
		self.X = self.X.transpose()


		#save de word dictionary
		self.eventDict = [sensorsDict,valuesDict, activityDict]


	def SurongFiveFeaturesPreviousEventActivityCodePaper(self):
		#set the encoding type
		self.encodingType = "SURONG_FIVE_FEATURES_PREVIOUS_EVENT_ACT_CODE_PAPER"


		self.df["dayOfWeek"] = self.df['datetime'].dt.dayofweek.astype(int)
		self.df["hour"] = self.df['datetime'].dt.hour.astype(int)

		#generate previous activity
		self.__directPreviousEventActivity()


		#encoding sensors ID
		sensorsDict, self.df['encodedSensor'] = self._encodeColumn(self.df['sensor'])

		#encoding value sensors
		self.df['encodedValue'] = self.df['value'].astype(str)
		self.df['encodedValue'] = self.df['encodedValue'].str.replace('ON','1.0')
		self.df['encodedValue'] = self.df['encodedValue'].str.replace('OFF','-1.0')

		val = self.df['encodedValue']
		valList = val.unique()
		valList.sort()

		valuesDict={}
		for i, v in enumerate(valList):
			valuesDict[v] = i+1	#add one to reserve 0 value to padding


		#encoding previous avtivity
		activityDict, self.df['encodedPrevAct'] = self._encodeColumn(self.df['prevAct'])


		feature_1 = self.df.encodedSensor.values.astype(float)
		feature_2 = self.df.encodedValue.values.astype(float)
		feature_3 = self.df.dayOfWeek.values.astype(float)
		feature_4 = self.df.hour.values.astype(float)
		feature_5 = self.df.encodedPrevAct.values.astype(float)

		self.X = np.array([feature_1,feature_2,feature_3,feature_4,feature_5])
		self.X = self.X.transpose()


		#save de word dictionary
		self.eventDict = [sensorsDict,valuesDict, activityDict]


	def fiveHandCrafted4(self):
		#set the encoding type
		self.encodingType = "5HCFEATURE"


		self.df["dayOfWeek"] = self.df['datetime'].dt.dayofweek.astype(int)
		self.df["hour"] = self.df['datetime'].dt.hour.astype(int)
		self.df["minute"] = self.df['datetime'].dt.minute.astype(int)


		#generate previous activity
		self.__previousAct()


		#encoding sensors ID
		sensorsDict, self.df['encodedSensor'] = self._encodeColumn(self.df['sensor'])

		#encoding value sensors
		#valuesDict, self.df['encodedValue'] = self._encodeColumn(self.df['value'])
		self.df['encodedValue'] = self.df['value'].astype(str)
		self.df['encodedValue'] = self.df['encodedValue'].str.replace('ON','1.0')
		self.df['encodedValue'] = self.df['encodedValue'].str.replace('OFF','-1.0')

		val = self.df['encodedValue']
		valList = val.unique()
		valList.sort()

		valuesDict={}
		for i, v in enumerate(valList):
			valuesDict[v] = i+1	#add one to reserve 0 value to padding


		#encoding previous avtivity
		activityDict, self.df['encodedPrevAct'] = self._encodeColumn(self.df['prevAct'])


		feature_1 = self.df.encodedSensor.values.astype(float)
		feature_2 = self.df.encodedValue.values.astype(float)
		feature_3 = self.df.dayOfWeek.values.astype(float)
		feature_4 = self.df.hour.values.astype(float)+np.round(self.df.minute.values.astype(float)/60,3)
		feature_5 = self.df.encodedPrevAct.values.astype(float)

		self.X = np.array([feature_1,feature_2,feature_3,feature_4,feature_5])
		self.X = self.X.transpose()


		#save de word dictionary
		self.eventDict = [sensorsDict,valuesDict, activityDict]




	def allSensorsOneRaw(self):

		#set the encoding type
		self.encodingType = "ALL_SENSORS_ONE_RAW"



		df2=self.df.copy()
		df2["merge"] = df2["sensor"] + df2["value"]

		merges = df2["merge"].unique()
		merges.sort()


		sensors = df2["sensor"].unique()
		sensors.sort()


		valDict={}
		for i, v in enumerate(merges):
			valDict[v] = i+1	#add one to reserve 0 value to padding

		df2 = df2.pivot(index='datetime', columns='sensor', values='merge')
		df2 = df2.fillna(method='ffill')
		df2 = df2.fillna(method='bfill')

		for s in sensors:
			df2[s] = df2[s].map(valDict)


		self.X = df2.values


		#save de word dictionary
		self.eventDict = valDict


	def allSensorsOneRaw2(self):

		#set the encoding type
		self.encodingType = "ALL_SENSORS_ONE_RAW"



		df2=self.df.copy()
		df2["merge"] = df2["sensor"] + df2["value"]

		merges = df2["merge"].unique()
		merges.sort()


		valDict={}
		for i, v in enumerate(merges):
			valDict[v] = i+1	#add one to reserve 0 value to padding

		df2 = df2.pivot(index='datetime', columns='sensor', values='merge')
		df2 = df2.fillna(method='ffill')
		df2 = df2.fillna(method='bfill')


		self.X = df2.values


		#save de word dictionary
		self.eventDict = valDict
