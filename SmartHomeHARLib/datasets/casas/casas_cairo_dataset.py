# coding: utf-8
# !/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
import re

from SmartHomeHARLib.datasets.casas import Dataset
from SmartHomeHARLib.utils.data_segmentation import *

class Cairo(Dataset):

    def __init__(self, path_to_dataset=None, clean_mode="raw"):

        # If no specific path to dataset is given, use the default one
        if path_to_dataset == None:
            current_file_directory = os.path.dirname(__file__)
            path_to_dataset = current_file_directory+"/../original_datasets/CASAS/cariro/data"
            path_to_dataset = os.path.normpath(path_to_dataset)

        super().__init__( "Aruba", path_to_dataset)

        self.clean_mode = clean_mode

        if self.clean_mode == "no_temperatures":
            self.remove_temperature_sensors_values()

            self.keep_informative_days()
            

