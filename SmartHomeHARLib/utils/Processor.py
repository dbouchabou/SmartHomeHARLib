# coding: utf-8
#!/usr/bin/env python3

import os
import pandas as pd


class Processor(object):

	def __init__(self, dataset):
		self.dataset = dataset
		self.data_X_train = None
		self.data_Y_train = None
		self.data_X_val = None
		self.data_Y_val = None
		self.data_X_test = None
		self.data_Y_test = None
		self.wordDict = None
		self.activityDict = None

	def process(self):
		raise NotImplementedError



	