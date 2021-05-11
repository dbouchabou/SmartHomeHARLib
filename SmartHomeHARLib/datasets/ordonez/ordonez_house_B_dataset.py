# coding: utf-8
# !/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
import re

from SmartHomeHARLib.datasets.ordonez import Ordonez
from SmartHomeHARLib.utils.data_segmentation import *


class HouseB(Ordonez):

    def __init__(self, dataset_folder):
        filename = dataset_folder+"/ORDONEZ/house_B/OrdonezB"
        super().__init__("HOUSE_B", filename)

            

