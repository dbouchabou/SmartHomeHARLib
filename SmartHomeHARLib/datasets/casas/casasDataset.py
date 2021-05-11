# coding: utf-8
# !/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
import re

from SmartHomeHARLib.datasets import SmartHomeDataset


class Dataset(SmartHomeDataset):

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
                              names=["datetime", "sensor", "value", "activity", "activityState"])
        self.df['datetime'] = pd.to_datetime(self.df["datetime"])

    def _annotate(self):

        df = pd.read_csv(self.filepath + "/cleanedData", sep="\t", header=None,
                         names=["datetime", "sensor", "value", "activity", "activityState"])

        df.activity = df.activity.fillna("")
        df.activityState = df.activityState.fillna("")

        activities = df.activity.values
        activitiesStates = df.activityState.values

        annotation = []

        activityStack = []
        activityStack.append("Other")

        for i, activity in enumerate(activities):
            # print(activity)
            activityState = activitiesStates[i]

            if not activityStack:
                # stack is empty
                activityStack.append("Other")

            if activity != "":

                # activity exist
                if activityState != "":
                    # log exist
                    if activityState == "begin":
                        # print(activity)
                        if activityStack[-1] != activity:
                            activityStack.append(activity)
                        # activityStack.append(activity)
                        act = activity

                    # if len(activityStack) > 2:
                    #
                    #	print("")
                    #	print(i)
                    #	print("{} begin".format(activity))
                    #	print("TETE = {}".format(activityStack[-1]))
                    #	print(activityStack)
                    elif activityState == "end":

                        activityStack = list(filter(lambda a: a != activity, activityStack))

                        act = activity

                    # if len(activityStack) > 2:
                    #
                    #	print("")
                    #	print(i)
                    #	print("{} end".format(activity))
                    #	print("TETE = {}".format(activityStack[-1]))
                    #	print(activityStack)
                else:
                    act = activity
            else:
                act = activityStack[-1]

            # print(act)
            annotation.append(act)

        df.activity = annotation

        df.to_csv(self.filepath + "/cleanedData", sep='\t', encoding='utf-8', header=False, index=False)

    def _cleanDataset(self):
        df = pd.read_csv(self.filename, sep="\s+", header=None,
                         names=["date", "time", "sensor", "value", "activity", "activityState"])

        df['datetime'] = pd.to_datetime(df["date"] + " " + df["time"])
        df = df.drop(columns=['date', 'time'])

        column_names = ["datetime", "sensor", "value", "activity", "activityState"]
        df = df.reindex(columns=column_names)

        df = df.sort_values(by=['datetime'])

        # Delete exact duplicate rows
        df = self.__deleteExactDuplicateRows(df)

        # Delete duplicate rows which doesnt have label
        df = self.__deleteSpecialDuplicateRows(df)

        # Fill log column
        df = self.__fillActivityStates(df)

        df.to_csv(self.filepath + "/cleanedData", sep='\t', encoding='utf-8', header=False, index=False)

    def __deleteExactDuplicateRows(self, df):

        # delete duplicate rows
        has_duplicate = df.duplicated()

        if len(has_duplicate[has_duplicate == True]) > 0:
            print("remove duplicate rows")
            df = df.drop_duplicates(keep='first')
            df = df.reset_index()
            df = df.drop(['index'], axis=1)

        return df

    def __deleteSpecialDuplicateRows(self, df):
        # Delete duplicate rows which dself,oesnt have label
        has_duplicate = df.duplicated(["datetime", "sensor", "value"])
        if len(has_duplicate[has_duplicate == True]) > 0:
            print("remove duplicate rows but keep one with an activity")

            indexToDel = []

            ponentialIndex = df.datetime.eq(df.datetime.shift())

            ii = np.where(ponentialIndex == True)[0]

            for counter, value in enumerate(ii):

                # print(counter)

                first = value - 1
                second = value

                # print(first)
                # print(df.iloc[first])
                # print(second)
                # print(df.iloc[second])
                # print(pd.isnull(df.activity.iloc[first]))

                if pd.isnull(df.activity.iloc[first]):
                    indexToDel.append(first)
                # print("suprr : {}".format(first))
                elif pd.isnull(df.activity.iloc[second]):
                    indexToDel.append(second)
                # print("suprr : {}".format(second))
            # print(" ")
            # print(" ")

            df = df.drop(index=indexToDel)

            df = df.reset_index()
            df = df.drop(['index'], axis=1)

        return df

    def __fillActivityStates(self, df):
        prohibitedWords = ['_begin', '_end', '="begin"', '="end"']
        big_regex = re.compile('|'.join(map(re.escape, prohibitedWords)))

        df.activity = df.activity.fillna("")

        activities = df.activity.values.astype(str)
        activitiesStates = df.activityState.values.astype(str)

        for i, activity in enumerate(activities):
            if "begin" in activity:
                activitiesStates[i] = "begin"

            if "end" in activity:
                activitiesStates[i] = "end"

            activities[i] = big_regex.sub("", activity)

        df.activity = activities
        df.activityState = activitiesStates

        return df

    def renameAcivities(self, dictActivities):

        self.df.activity = self.df.activity.map(dictActivities).astype(str)

        self._generateActivityList()
