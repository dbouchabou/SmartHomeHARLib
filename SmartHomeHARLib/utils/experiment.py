# coding: utf-8
# !/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import csv
import json
import time


class Experiment(object):

	def __init__(self, dataset, experiment_parameters=None):
		
		self.DEBUG = False

		self.dataset = dataset

		self.experiment_parameters = experiment_parameters

		self.experiment_result_path = None


	def start(self):

		raise NotImplementedError


	def save_config(self):

		experiment_parameters_name = "experiment_parameters.json"
		experiment_parameters_path = os.path.join(self.experiment_result_path, experiment_parameters_name)

		with open(experiment_parameters_path,"w") as save_config_file:
			json.dump(self.experiment_parameters, save_config_file, indent = 4)