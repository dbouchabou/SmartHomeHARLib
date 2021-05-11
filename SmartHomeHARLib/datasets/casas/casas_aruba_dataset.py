# coding: utf-8
# !/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
import re

from SmartHomeHARLib.datasets.casas import Dataset
from SmartHomeHARLib.utils.data_segmentation import *

class Aruba(Dataset):

    def __init__(self, path_to_dataset= None, clean_mode = "raw"):

        # If no specific path to dataset is given, use the default one
        if path_to_dataset == None:
            current_file_directory = os.path.dirname(__file__)
            path_to_dataset = current_file_directory+"/../original_datasets/CASAS/aruba/data"
            path_to_dataset = os.path.normpath(path_to_dataset)

        super().__init__( "Aruba", path_to_dataset)

        self.clean_mode = clean_mode

        if self.clean_mode == "no_temperatures":
            self.remove_temperature_sensors_values()

            self.keep_informative_days()


    def remove_temperature_sensors_values(self):

        self.df = self.df[-self.df["sensor"].str.startswith('T')]
        self.df = self.df.reset_index(drop = True)
        

    def keep_informative_days(self):
        """
        remove days that contains only the "Other" activity label from the df dataframe, because these days are short and informativeless

        """

        dofw, days = dataframe_day_window(self.df)

        interesting_days = []

        for i, day in enumerate(days):
            
            if len(day.activity.unique()) > 1 or day.activity.unique()[0] != "Other":
                
                interesting_days.append(day)

        self.df = vertical_stack = pd.concat(interesting_days, axis=0)
        self.df = self.df.reset_index(drop = True)
            

