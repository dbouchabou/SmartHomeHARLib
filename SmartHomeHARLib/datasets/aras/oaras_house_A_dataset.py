# coding: utf-8
# !/usr/bin/env python3

import os


from SmartHomeHARLib.datasets.ordonez import Ordonez
from SmartHomeHARLib.utils.data_segmentation import *


class HouseA(Ordonez):

    def __init__(self, path_to_dataset=None):

        # If no specific path to dataset is given, use the default one
        if path_to_dataset == None:
            current_file_directory = os.path.dirname(__file__)
            path_to_dataset = current_file_directory+"/../original_datasets/ORDONEZ/house_A/OrdonezA"
            path_to_dataset = os.path.normpath(path_to_dataset)

        super().__init__("HOUSE_A", path_to_dataset)

            

