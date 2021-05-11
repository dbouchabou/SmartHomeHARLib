# coding: utf-8
# !/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
import re

from SmartHomeHARLib.datasets.casas import Dataset
from SmartHomeHARLib.utils.data_segmentation import *


class Milan(Dataset):

    def __init__(self, name, filename, clean_mode = "raw"):
        super().__init__(name, filename, "MILAN")

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
            

