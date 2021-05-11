# coding: utf-8
#!/usr/bin/env python3

import os
import pandas as pd


class SmartHomeDataset(object):

	def __init__(self, name, filename, datasetType = None):
		
		self.name = name
		self.filename = filename
		self.filepath = os.path.dirname(filename)
		self.datasetType = datasetType
		self.df = None
		self.sensorsList = []
		self.activitiesList = []


		#if the cleaned path doesn't exist
		if not os.path.isfile(self.filepath+"/cleanedData"):
			#raise Exception("cleaned data file doesn't exist")
			print("cleanedData file doesn't exist")
			self._cleanDataset()
			self._annotate()


		self._loadDataset()

		self._generateSensorList()

		self._generateActivityList()


	def _generateSensorList(self):

		self.sensorsList = self.df.sensor.unique().astype(str)
		self.sensorsList.sort()


	def _annotate(self):
		raise NotImplementedError


	def _cleanDataset(self):
		raise NotImplementedError


	def _loadDataset(self):
		raise NotImplementedError


	def renameSensors(self, df, dictActivities):
		self.df.sensors = self.df.activity.map(dictSensors).astype(str)

		self._generateSensorList()


	def renameAcivities(self, dictActivities):
		raise NotImplementedError


	@property
	def getNumActivities(self):
		
		return len(self.activitiesList)


	@property
	def getNumSensors(self):

		return len(self.sensorsList)


	@property
	def getNumDays(self):

		return len(self.df.datetime.dt.date.unique())


	def statistics(self):

		statistics = {
			"dataset_type": "{}".format(self.datasetType),
			"dataset_name": "{}".format(self.name),
			"nb_sensors": "{}".format(self.getNumSensors),
			"nb_activities": "{}".format(self. getNumActivities),
			"nb_days": "{}".format(self.getNumDays)
		}

		return statistics