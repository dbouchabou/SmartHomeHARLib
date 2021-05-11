# coding: utf-8
# !/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
import re
from datetime import datetime

from SmartHomeHARLib.datasets import SmartHomeDataset


class Dataset_2(SmartHomeDataset):

    def __init__(self, name, filename):
        super().__init__(name, filename, "CASAS")

    # self.__filename = filename
    # self.__filepath = os.path.dirname(filename)

    def _generateActivityList(self):

        self.activitiesList = self.df.activity.unique().astype(str)
        self.activitiesList.sort()

    def _loadDataset(self):

        print("Load dataset")
        # load the the raw file to a Pandas dataframe
        self.df = pd.read_csv(self.filepath + "/cleanedData", sep="\t", header=None,
                              names=["datetime", "sensor", "value", "activity"])
        self.df['datetime'] = pd.to_datetime(self.df["datetime"])
    

    def _annotate(self):
        
        # dateset fields
        timestamps = []
        sensors = []
        values = []
        activities = []

        activity = 'Other'  # empty

        with open(self.filename, 'rb') as features:
            database = features.readlines()
            for i, line in enumerate(database):  # each line
                f_info = line.decode().split()  # find fields
                try:
                    if 'M' == f_info[2][0] or 'D' == f_info[2][0] or 'T' == f_info[2][0]:
                        # choose only M D T sensors, avoiding unexpected errors
                        if not ('.' in str(np.array(f_info[0])) + str(np.array(f_info[1]))):
                            f_info[1] = f_info[1] + '.000000'
                        timestamps.append(datetime.strptime(str(np.array(f_info[0])) + str(np.array(f_info[1])),
                                                            "%Y-%m-%d%H:%M:%S.%f"))
                        sensors.append(str(np.array(f_info[2])))
                        values.append(str(np.array(f_info[3])))

                        if len(f_info) == 4:  # if activity does not exist
                            activities.append(activity)
                        else:  # if activity exists
                            des = str(' '.join(np.array(f_info[4:])))
                            if 'begin' in des:
                                activity = re.sub('begin', '', des)
                                if activity[-1] == ' ':  # if white space at the end
                                    activity = activity[:-1]  # delete white space
                                activities.append(activity)
                            if 'end' in des:
                                activities.append(activity)
                                activity = 'Other'
                except IndexError:
                    print(i, line)
        features.close()
        
        df = pd.DataFrame(list(zip(timestamps, sensors, values, activities)), columns = ["datetime", "sensor", "value", "activity"])

        df.to_csv(self.filepath + "/cleanedData", sep='\t', encoding='utf-8', header=False, index=False)

    def _cleanDataset(self):
        pass

    def renameAcivities(self, dictActivities):

        self.df.activity = self.df.activity.map(dictActivities).astype(str)

        self._generateActivityList()
