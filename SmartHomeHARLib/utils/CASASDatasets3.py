#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import re

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

SEED = 7

np.random.seed(SEED)

#from tensorflow import set_random_seed
#set_random_seed(SEED)
tf.random.set_seed(SEED)

class CASASDatasets:

	def __init__(self, name, filename):
		
		self.name = name
		self.filename = filename
		self.filepath = os.path.dirname(filename)
		self.df = None
		self.encodingType = None
		self.eventDict = {}
		self.actDict = {}
		self.segments = None


	def __removeDuplicates(self):

		#drop duplicate rows
		df = pd.read_csv(self.filepath+"/cleanedData",sep="\t",header=None,names=["datetime","sensor","value","activity","log"])

		df['datetime'] = pd.to_datetime(df["datetime"])
		df = df.sort_values(by=['datetime'])


		df = df.drop_duplicates(keep='first')
		df=df.reset_index()
		df=df.drop(['index'], axis=1)

		indexToDel = []

		ponentialIndex = df.datetime.eq(df.datetime.shift())

		ii = np.where(ponentialIndex == True)[0]

		for counter, value in enumerate(ii):

			#print(counter)

			first = value-1
			second = value

			#print(first)
			#print(second)
			#print(pd.isnull(df2.activity.iloc[first]))

			if pd.isnull(df.activity.iloc[first]):
				indexToDel.append(first)
				#print("suprr : {}".format(first))
			elif pd.isnull(df.activity.iloc[second]):
				indexToDel.append(second)
				#print("suprr : {}".format(second))

		df = df.drop(index=indexToDel)

		df=df.reset_index()
		df=df.drop(['index'], axis=1)

		df.to_csv(self.filepath+"/cleanedData", sep='\t', encoding='utf-8', header=False, index = False)


	def __cleanRawfile(self):

		#possible activity states in CASAS datasets
		#list not exhaustive for the moment
		prohibitedWords = ['_begin', '_end', '="begin"','="end"']
		big_regex = re.compile('|'.join(map(re.escape, prohibitedWords)))

		with open(self.filename, "r") as rawFile, open(self.filepath+"/cleanedData", "w") as cleanedFile:

			Lines = rawFile.readlines()

			for line in Lines:

				lineSplit = line.split()
				#print(lineSplit)
				date = lineSplit[0]
				time = lineSplit[1]
				sensor = lineSplit[2]
				value = lineSplit[3]
				activity = np.NaN
				log = np.NaN

				if len(lineSplit) > 4:
					#activity exist

					activity = lineSplit[4]


					if len(lineSplit) > 5:
						#log exist
						log = lineSplit[5]

					if "begin" in activity:
						log="begin"

					if "end" in activity:
						log="end"

					activity = big_regex.sub("", activity)

					
				newLine = "{} {}\t{}\t{}\t{}\t{}\n".format(date,time,sensor,value,activity,log)

				cleanedFile.write(newLine)


	def __annotate(self):

		print("annotate_1")

		df = pd.read_csv(self.filepath+"/cleanedData",sep="\t",header=None,names=["datetime","sensor","value","activity","log"])

		##proccess to annotate each event with the correct activity label
		df.activity = df.activity.fillna("Other")

		df.log = df.log.fillna("NA")

		activities = df.activity.values

		logs = df.log.values

		for i in range(len(activities)):
			
			if logs[i] == "begin":

				act = activities[i]
				j = i+1

				while activities[j] == "Other" and logs[j] == "NA":
					activities[j] = act
					j+=1

		for i in range(len(activities))[::-1]:
			
			if logs[i] == "end":
				
				act = activities[i]
				j = i-1
				
				while activities[j] == "Other" and logs[j] == "NA":
					activities[j] = act
					j-=1

		df.activity = activities

		df.to_csv(self.filepath+"/cleanedData", sep='\t', encoding='utf-8', header=False, index = False)


	def __annotate2(self):

		print("annotate_2")

		df = pd.read_csv(self.filepath+"/cleanedData",sep="\t",header=None,names=["datetime","sensor","value","activity","log"])

		#rempli les valeurs NaN de la colonne log avec la valeur précédente
		df.log = df.log.fillna(method='ffill')

		#rempli les valeur NaN de la colonne activity avec lavaleur de la colonne log
		df['activity'] = df['activity'].fillna(df['log'])

		df['activity'] = df['activity'].replace("end", "Other")

		df['activity'] = df['activity'].fillna("Other")

		df['activity'] = df['activity'].replace("begin", None)

		df.to_csv(self.filepath+"/cleanedData", sep='\t', encoding='utf-8', header=False, index = False)


	def __annotate3(self):
		print("annotate_3")

		df = pd.read_csv(self.filepath+"/cleanedData",sep="\t",header=None,names=["datetime","sensor","value","activity","log"])

		df.activity = df.activity.fillna("")
		df.log = df.log.fillna("")

		activities = df.activity.values
		logs = df.log.values

		annotation = []

		activityStack = []
		activityStack.append("Other")


		for i,activity in enumerate(activities):
			#print(activity)
			log = logs[i]

			if not activityStack :
				#stack is empty
				activityStack.append("Other")

			if activity != "":

				#activity exist
				if log != "":
					#log exist
					if log=="begin":
						#print(activity)
						if activityStack[-1] != activity:
							activityStack.append(activity)
						#activityStack.append(activity)
						act = activity
						#print("TETE = {}".format(activityStack[-1]))
					elif log=="end":
						
						activityStack = list(filter(lambda a: a != activity, activityStack))

						act = activity
						#print("")
						#print("{} end".format(activity))
						#print("TETE = {}".format(activityStack[-1]))
						#print(activityStack)
				else:
					act = activity
			else:
				act = activityStack[-1]

			#print(act)
			annotation.append(act)
		
		
		df.activity = annotation

		df.to_csv(self.filepath+"/cleanedData", sep='\t', encoding='utf-8', header=False, index = False)


	def __annotate4(self):
		print("annotate_4")

		df = pd.read_csv(self.filepath+"/cleanedData",sep="\t",header=None,names=["datetime","sensor","value","activity","log"])

		df.activity = df.activity.fillna("")
		df.log = df.log.fillna("")

		activities = df.activity.values
		logs = df.log.values

		annotation = []

		currentActivity = "Other"

		for i,activity in enumerate(activities):
			#print(activity)
			log = logs[i]

			if activity != "":
				#activity exist
				if log != "":
					#log exist
					if log=="begin":
						#print(activity)
						annotation.append(activity)
						currentActivity = activity
					elif log=="end":
						annotation.append(activity)
						currentActivity = "Other"
			else:
				annotation.append(currentActivity)
			
		
		df.activity = annotation

		df.to_csv(self.filepath+"/cleanedData", sep='\t', encoding='utf-8', header=False, index = False)



	def __cleanDataset(self):

		print("Start Cleanning")

		print("1 - Clean raw flie")
		self.__cleanRawfile()

		print("2 - Remove duplicate data")
		self.__removeDuplicates()

		print("3 - Annotate each events")
		#self.__annotate()
		#self.__annotate2()
		self.__annotate3()
		#self.__annotate4()


	def __generateActDict(self):

		activityList = self.df.activity.unique()
		activityList.sort()


		self.actDict = {}
		for i, activity in enumerate(activityList):
			self.actDict[activity] = i


	def loadDataset(self):

		#if the cleaned path doesn't exist
		if not os.path.isfile(self.filepath+"/cleanedData"):
			#raise Exception("cleaned data file doesn't exist")
			print("cleanedData file doesn't exist")
			self.__cleanDataset()

		print("Load dataset")
		# load the the raw file to a Pandas dataframe
		self.df = pd.read_csv(self.filepath+"/cleanedData",sep="\t",header=None,names=["datetime","sensor","value","activity","log"])
		self.df['datetime'] = pd.to_datetime(self.df["datetime"])


		self.__generateActDict()


	def getName(self):

		return self.name


	def renameAcivities(self, dictAct):

		for newActivityName in dictAct:
			#print(sensor)
			self.df.activity = self.df.activity.replace(newActivityName, dictAct[newActivityName])

		self.__generateActDict()


	def getActivityList(self):

		*activitiesList, = self.actDict

		return activitiesList

	def getActivityDict(self):

		return self.actDict


	def getSensorList(self):

		sensorList = self.df.sensor.unique()
		sensorList.sort()

		return sensorList


	def getNumActivities(self):
		
		activityList = self.df.activity.unique()
		
		return len(activityList)


	def getNumSensors(self):

		sensorList = self.df.sensor.unique()

		return len(sensorList)


	def getNumDays(self):

		dateList = self.df.datetime.dt.date.unique()

		return len(dateList)


	def getRawData(self):

		return self.df


	def getEncodingType(self):
		return self.encodingType


	#SEGMENTATION Of DATASET
	def explicitWindow(self):
		activitiesSeq = []

		ponentialIndex = self.df.activity.ne(self.df.activity.shift())

		ii = np.where(ponentialIndex == True)[0]

		for i,end in enumerate(ii):
			if i > 0 :

				activitiesSeq.append(self.df[ii[i-1]:end])

		self.segments = activitiesSeq
		

	def sensorEventWindows(self,winSize,step=1):
		sew = []

		numOfChunks = int(((self.df.shape[0]-winSize)/step)+1)

		for i in range(0,numOfChunks*step,step):
			sew.append(self.df[i:i+winSize])

		self.segments = sew


	#ENCODING
	def __pad(self, sequences, paddingLen, padtype):

		if paddingLen == 0 :
			print("maxlen = 0")
			padded_x = pad_sequences(sequences, padding=padtype)
		else:
			print("maxlen = {}".format(paddingLen))
			padded_x = pad_sequences(sequences, maxlen=paddingLen, padding=padtype)

		return padded_x

	def __indexLabels(self, labels):


		Y = labels
		for i,y in enumerate(labels):
			Y[i]=self.actDict[y]

		return np.array(Y)

	def basicEncoding(self):

		self.encodingType = "BASIC"

		
		eventsList = []

		
		self.df['merge'] = self.df['sensor'] + self.df['value'].astype(str)

		
		eventsList = self.df['merge'].unique()
		eventsList.sort()

		#create a event dict
		self.eventDict={}			
		for i, event in enumerate(eventsList):
			self.eventDict[event] = i+1


		for event in self.eventDict:
			#print(sensor)
			self.df['merge'] = self.df['merge'].replace(event, self.eventDict[event])

		for event in self.eventDict:
			#print(sensor)
			self.df['merge'] = self.df['merge'].replace(event, self.eventDict[event])
		

	def nlpBasicEncoding(self):

		self.encodingType = "NLPBASIC"

		
		self.df['merge'] = self.df['sensor'] + self.df['value'].astype(str)



	def nlpMidnightEncoding(self):

		self.encodingType = "NLPMIDNIGHT"


		
		self.df = self.df.sort_values(by=['datetime'])
			

		self.df["since_midnight"] = self.df['datetime'].dt.hour.astype(int)*3600+self.df['datetime'].dt.minute.astype(int)*60+self.df['datetime'].dt.second.astype(int)


		self.df['nlp'] = self.df['sensor'] + self.df['value'].astype(str) + " " + self.df["since_midnight"].astype(str)



	def __basicencoding(self):

		sequences = []
		labels = []

		for seg in self.segments:

			#already indexed
			sequences.append(seg['merge'].values)
			labels.append(seg.activity.values[-1])

		return sequences, labels


	def __nlpBasicEncoding(self):

		sentences = []
		labels = []

		for seg in self.segments:

			sentence = " ".join(seg['merge'].values)

			sentences.append(sentence)
			labels.append(seg.activity.values[-1])


		tokenizer = Tokenizer(filters='')
		tokenizer.fit_on_texts(sentences)

		wordDict = tokenizer.word_index

		indexed_sentences = tokenizer.texts_to_sequences(sentences)

		return indexed_sentences, labels, wordDict


	def __nlpMidnightEncoding(self):

		sentences = []
		labels = []

		for seg in self.segments:

			sentence = " ".join(seg['nlp'].values)

			sentences.append(sentence)
			labels.append(seg.activity.values[-1])


		tokenizer = Tokenizer(filters='')
		tokenizer.fit_on_texts(sentences)

		wordDict = tokenizer.word_index

		indexed_sentences = tokenizer.texts_to_sequences(sentences)

		return indexed_sentences, labels, wordDict


	def getData(self, paddingLen=0, padtype = 'pre'):

		if self.segments == None:
			raise("You should segment dataset before")

		

		if self.encodingType == "BASIC":

			sequences, labels = self.__basicencoding()
			eventDict = self.eventDict

		if self.encodingType == "NLPBASIC":

			sequences, labels, eventDict = self.__nlpBasicEncoding()

		if self.encodingType == "NLPMIDNIGHT":
			sequences, labels, eventDict = self.__nlpMidnightEncoding()
				

		padded_x = self.__pad(sequences, paddingLen, padtype)

		Y =  self.__indexLabels(labels)

		return eventDict, self.actDict, padded_x, Y

	def getSegments(self):
		return self.segments
			
