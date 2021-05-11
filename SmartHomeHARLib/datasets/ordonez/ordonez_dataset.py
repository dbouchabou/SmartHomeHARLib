# coding: utf-8
# !/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
import re

from SmartHomeHARLib.datasets import SmartHomeDataset


class Ordonez(SmartHomeDataset):

    def __init__(self, name, filename):
        super().__init__(name, filename, "ORDONEZ")

    # self.__filename = filename
    # self.__filepath = os.path.dirname(filename)

    def _generateActivityList(self):

        self.activitiesList = self.df.activity.unique().astype(str)
        self.activitiesList.sort()


    def _loadDataset(self):

        print("Load dataset")
        # load the the raw file to a Pandas dataframe
        self.df = pd.read_csv(self.filepath + "/cleanedData", 
                                sep="\t", header=None,
                                names=["datetime", "sensor", "place", "value", "activity"])

        self.df['datetime'] = pd.to_datetime(self.df["datetime"])


    def _annotate(self):

        filename = self.filename+"_ADLs.txt"

        df = pd.read_csv(self.filepath + "/cleanedData", 
                            sep="\t", 
                            header=None,
                            names=["datetime", "sensor", "place", "value"]
        )

        df_labels = pd.read_csv(filename, 
                 sep='\s+', 
                 header=None,  
                 skiprows = [0,1], 
                 engine='python',
                 names=["sd","st","ed","et", "activity"]
        )

        df_labels["start time"] = df_labels["sd"] + " " + df_labels["st"]
        df_labels["end time"] = df_labels["ed"] + " " + df_labels["et"]


        st = df_labels["start time"].values
        et = df_labels["end time"].values
        label = df_labels["activity"].values


        # Annotate sensor events with the label according to datetime
        # Add an empty activity column
        df["activity"] = ""

        for i in range(len(st)):
            mask = (df["datetime"] >= st[i]) & (df["datetime"] <= et[i])
            df.loc[mask, "activity"] = label[i]


        # Anotate no annotate sensor events with "Other"
        df = df.replace("", "Other")        


        df.to_csv(self.filepath + "/cleanedData", sep='\t', encoding='utf-8', header=False, index=False)


    def _cleanDataset(self):

        filename = self.filename+"_Sensors.txt"

        df = pd.read_csv(filename, 
                 sep='\s+', 
                 header=None,  
                 skiprows = [0,1], 
                 engine='python',
                 names=["sd","st","ed","et", "location", "type", "place"]
        )

        df["sensor"] = df["location"]+"_"+df["type"]+"_"+df["place"]
        df["start time"] = df["sd"] + " " + df["st"]
        df["end time"] = df["ed"] + " " + df["et"]

        df = df[["start time","end time","sensor","place"]]

        # Extract all sensors "ON" activations
        df_start = df[["start time","sensor","place"]]
        df_start["value"] = "ON"
        df_start = df_start.rename(columns={"start time": "datetime"})

        # Extract all sensors "OFF" activations
        df_end = df[["end time","sensor","place"]]
        df_end["value"] = "OFF"
        df_end = df_end.rename(columns={"end time": "datetime"})

        # Merge sensors activations
        df_final = pd.concat([df_start,df_end], axis=0)

        # Sort by datetime
        df_final['datetime'] = pd.to_datetime(df_final['datetime'])
        df_final = df_final.sort_values(by=['datetime'])
        df_final = df_final.reset_index(drop = True)
        

        # Store the final dataframe
        df_final.to_csv(self.filepath + "/cleanedData", sep='\t', encoding='utf-8', header=False, index=False)


    def renameAcivities(self, dictActivities):

        self.df.activity = self.df.activity.map(dictActivities).astype(str)

        self._generateActivityList()
