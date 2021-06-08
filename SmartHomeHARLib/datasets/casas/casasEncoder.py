#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import re

from SmartHomeHARLib.datasets import DatasetEncoder

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Encoder(DatasetEncoder):

    def __init__(self, dataset):

        #if dataset.datasetType != "CASAS":
        #    raise ValueError('Argument should be CASASDataset type')

        super().__init__(dataset)

        self.__indexLabels()

        self.Y = self.df.encodedActivity.values.astype(int)

    def __indexLabels(self):

        self.df["encodedActivity"] = self.df.activity.astype(str)

        _, self.df["encodedActivity"] = self._encodeColumn(self.df["encodedActivity"], self.actDict)

    def basic(self):

        # set the encoding type
        self.encodingType = "BASIC"

        # create event tokens
        self.df['merge'] = self.df['sensor'] + self.df['value'].astype(str)

        # encode
        wordDict, self.df['encodedMerge'] = self._encodeColumn(self.df["merge"])

        self.X = self.df['encodedMerge'].values.astype(int)

        # save de word dictionary
        self.eventDict = wordDict

    def nlpBasic(self, custom_dict = None, lower = False):

        # set the encoding type
        self.encodingType = "NLPBASIC"

        self.df['nlp'] = self.df['sensor'] + self.df['value'].astype(str)

        if custom_dict == None:

            sentence = " ".join(self.df['nlp'].values)

            tokenizer = Tokenizer(filters='', lower = lower)
            tokenizer.fit_on_texts([sentence])

            wordDict = tokenizer.word_index
        else:
            wordDict = custom_dict

        print(wordDict)
        print(len(wordDict))
        input("Press Enter to continue...")

        # encode
        if lower:
            self.df['encodedNLP'] = self.df['nlp'].str.lower()
        else:
            self.df['encodedNLP'] = self.df['nlp'].astype(str)

        # self.df['encodedNLP'] = self.df['encodedNLP'].map(wordDict).astype(str)
        _, self.df['encodedNLP'] = self._encodeColumn(self.df["encodedNLP"], wordDict)

        self.X = self.df['encodedNLP'].values.astype(int)

        # save de word dictionary
        self.eventDict = wordDict

    def nlpMidnight(self):

        # set the encoding type
        self.encodingType = "NLPMIDNIGHT"

        self.df["sinceMidnight"] = self.df['datetime'].dt.hour.astype(int) * 3600 + self.df[
            'datetime'].dt.minute.astype(int) * 60 + self.df['datetime'].dt.second.astype(int)
        self.df["sinceMidnight"] = self.df["sinceMidnight"].astype(str)
        self.df["merge"] = self.df['sensor'] + self.df['value'].astype(str)
        self.df['nlp'] = self.df["merge"] + " " + self.df["sinceMidnight"]

        sentence = " ".join(self.df['nlp'].values)

        tokenizer = Tokenizer(filters='')
        tokenizer.fit_on_texts([sentence])

        wordDict = tokenizer.word_index

        self.df['encodedMerge'] = self.df['merge'].str.lower()
        # self.df['encodedMerge'] = self.df['encodedMerge'].map(wordDict)
        _, self.df['encodedMerge'] = self._encodeColumn(self.df["encodedMerge"], wordDict)

        self.df['encodedSinceMidnight'] = self.df['sinceMidnight']
        # self.df['encodedSinceMidnight'] = self.df['encodedSinceMidnight'].map(wordDict)
        _, self.df['encodedSinceMidnight'] = self._encodeColumn(self.df["encodedSinceMidnight"], wordDict)

        # self.df['encodedNLP'] =  self.df["encodedMerge"].astype(str) + " " + self.df["encodedSinceMidnight"].astype(str)

        # self.X = " ".join(self.df['encodedNLP'].values)
        # self.X = self.X.split()
        # self.X = list(map(int, self.X))

        feature_1 = self.df.encodedMerge.values.astype(int)
        feature_2 = self.df.encodedSinceMidnight.values.astype(int)

        self.X = np.array([feature_1, feature_2])
        self.X = self.X.transpose()

        # save de word dictionary
        self.eventDict = wordDict

    def fourHandCrafted(self):

        # set the encoding type
        self.encodingType = "4HCFEATURE"

        self.df["dayOfWeek"] = self.df['datetime'].dt.dayofweek.astype(int) + 1
        self.df["hour"] = self.df['datetime'].dt.hour.astype(int) + 1

        # encoding sensors ID
        sensorsDict, self.df['encodedSensor'] = self._encodeColumn(self.df['sensor'])

        # encoding value sensors
        valuesDict, self.df['encodedValue'] = self._encodeColumn(self.df['value'])

        feature_1 = self.df.dayOfWeek.values.astype(int)
        feature_2 = self.df.hour.values.astype(int)
        feature_3 = self.df.encodedSensor.values.astype(int)
        feature_4 = self.df.encodedValue.values.astype(int)

        self.X = np.array([feature_1, feature_2, feature_3, feature_4])
        self.X = self.X.transpose()

        # save de word dictionary
        self.eventDict = [sensorsDict, valuesDict]

    def __directPreviousActivity(self):

        # previous act
        NaN = np.nan
        self.df["prevAct"] = NaN

        pa = self.df.prevAct.values.astype(object)

        ponentialIndex = self.df.activity.ne(self.df.activity.shift())

        ii = np.where(ponentialIndex == True)[0]

        for i, transiInd in enumerate(ii):
            if i > 0:
                prevA = self.df[transiInd - 1:transiInd].activity.values[-1]
                currentA = self.df[transiInd:transiInd + 1].activity.values[-1]

                # print("previous act {}".format(prevA))
                # print("current act {}".format(currentA))

                pa[transiInd] = prevA

        prevA = self.df[ii[-1] - 1:ii[-1]].activity.values[-1]
        currentA = self.df[ii[-1]:ii[-1] + 1].activity.values[-1]

        # print("previous act {}".format(prevA))
        # print("current act {}".format(currentA))

        pa[ii[-1]] = prevA

        self.df["prevAct"] = pa

        self.df.prevAct = self.df.prevAct.fillna(method='ffill')

        self.df.prevAct = self.df.prevAct.fillna(method='bfill')

    def __previousActivityNoOther(self):

        # previous act
        act = self.df.activity.values

        an_array = np.empty(len(act), dtype=object)
        prevAct = []

        ponentialIndex = self.df.activity.ne(self.df.activity.shift())

        ii = np.where(ponentialIndex == True)[0]

        for i, end in enumerate(ii):
            if i > 0:
                # print(self.df[ii[i-1]:ii[i]].activity.values[-1])
                prevAct.append(self.df[ii[i - 1]:ii[i]].activity.values[-1])
        # print(self.df[ii[i]:].activity.values[-1])
        prevAct.append(self.df[ii[i]:].activity.values[-1])

        noOther = list(filter(lambda a: a != "Other", prevAct))

        noOther.insert(0, 'Other')

        lastReslAct = 0
        for i in ii:
            # print(i)
            # print(lastReslAct)
            an_array[i] = noOther[lastReslAct]
            if act[i] != "Other":
                lastReslAct += 1

        self.df["prevAct"] = an_array

        self.df.prevAct = self.df.prevAct.fillna(method='ffill')

    def __previousActivityAsCodePaper(self):

        # previous act
        self.df["prevAct"] = self.df.activity

    def __directPreviousEventActivity(self):

        # previous act
        self.df["prevAct"] = self.df.activity.shift()

        self.df.prevAct = self.df.prevAct.fillna(method='bfill')

    def fiveHandCrafted(self):
        # set the encoding type
        self.encodingType = "5HCFEATURE"

        self.df["dayOfWeek"] = self.df['datetime'].dt.dayofweek.astype(int) + 1
        self.df["hour"] = self.df['datetime'].dt.hour.astype(int) + 1

        # generate previous activity
        self.__previousAct()

        # encoding sensors ID
        sensorsDict, self.df['encodedSensor'] = self._encodeColumn(self.df['sensor'])

        # encoding value sensors
        valuesDict, self.df['encodedValue'] = self._encodeColumn(self.df['value'])

        # encoding previous avtivity
        activityDict, self.df['encodedPrevAct'] = self._encodeColumn(self.df['prevAct'])

        feature_1 = self.df.dayOfWeek.values.astype(int)
        feature_2 = self.df.hour.values.astype(int)
        feature_3 = self.df.encodedSensor.values.astype(int)
        feature_4 = self.df.encodedValue.values.astype(int)
        feature_5 = self.df.encodedPrevAct.values.astype(int)

        self.X = np.array([feature_1, feature_2, feature_3, feature_4, feature_5])
        self.X = self.X.transpose()

        # save de word dictionary
        self.eventDict = [sensorsDict, valuesDict, activityDict]

    def fiveHandCrafted2(self):
        # set the encoding type
        self.encodingType = "5HCFEATURE"

        self.df["dayOfWeek"] = self.df['datetime'].dt.dayofweek.astype(int)
        self.df["hour"] = self.df['datetime'].dt.hour.astype(int)
        self.df["minute"] = self.df['datetime'].dt.minute.astype(int)

        # generate previous activity
        self.__previousAct()

        # encoding sensors ID
        sensorsDict, self.df['encodedSensor'] = self._encodeColumn(self.df['sensor'])

        # encoding value sensors
        valuesDict, self.df['encodedValue'] = self._encodeColumn(self.df['value'])
        # self.df['encodedValue'] = self.df['value'].str.replace('ON','1.0')
        # self.df['encodedValue'] = self.df['value'].str.replace('OFF','0.0')

        # encoding previous avtivity
        activityDict, self.df['encodedPrevAct'] = self._encodeColumn(self.df['prevAct'])

        feature_1 = self.df.dayOfWeek.values.astype(int)
        feature_2 = self.df.hour.values.astype(float) + np.round(self.df.minute.values.astype(float) / 60, 3)
        feature_3 = self.df.encodedSensor.values.astype(int)
        feature_4 = self.df.encodedValue.values.astype(int)
        feature_5 = self.df.encodedPrevAct.values.astype(int)

        self.X = np.array([feature_1, feature_2, feature_3, feature_4, feature_5])
        self.X = self.X.transpose()

        # save de word dictionary
        self.eventDict = [sensorsDict, valuesDict, activityDict]

    def SurongFiveFeaturesDirectPreviousActivity(self):
        # set the encoding type
        self.encodingType = "SURONG_FIVE_FEATURES_DIRECT_PREVIOUS_ACT"

        self.df["dayOfWeek"] = self.df['datetime'].dt.dayofweek.astype(int)
        self.df["hour"] = self.df['datetime'].dt.hour.astype(int)

        # generate previous activity
        self.__directPreviousActivity()

        # encoding sensors ID
        sensorsDict, self.df['encodedSensor'] = self._encodeColumn(self.df['sensor'])

        # encoding value sensors
        self.df['encodedValue'] = self.df['value'].astype(str)
        self.df['encodedValue'] = self.df['encodedValue'].str.replace('ON', '1.0')
        self.df['encodedValue'] = self.df['encodedValue'].str.replace('OFF', '-1.0')

        val = self.df['encodedValue']
        valList = val.unique()
        valList.sort()

        valuesDict = {}
        for i, v in enumerate(valList):
            valuesDict[v] = i + 1  # add one to reserve 0 value to padding

        # encoding previous avtivity
        activityDict, self.df['encodedPrevAct'] = self._encodeColumn(self.df['prevAct'])

        feature_1 = self.df.encodedSensor.values.astype(float)
        feature_2 = self.df.encodedValue.values.astype(float)
        feature_3 = self.df.dayOfWeek.values.astype(float)
        feature_4 = self.df.hour.values.astype(float)
        feature_5 = self.df.encodedPrevAct.values.astype(float)

        self.X = np.array([feature_1, feature_2, feature_3, feature_4, feature_5])
        self.X = self.X.transpose()

        # save de word dictionary
        self.eventDict = [sensorsDict, valuesDict, activityDict]

    def SurongFiveFeaturesPreviousActivityNoOther(self):
        # set the encoding type
        self.encodingType = "SURONG_FIVE_FEATURES_PREVIOUS_ACT_NO_OTHER"

        self.df["dayOfWeek"] = self.df['datetime'].dt.dayofweek.astype(int)
        self.df["hour"] = self.df['datetime'].dt.hour.astype(int)

        # generate previous activity
        self.__previousActivityNoOther()

        # encoding sensors ID
        sensorsDict, self.df['encodedSensor'] = self._encodeColumn(self.df['sensor'])

        # encoding value sensors
        self.df['encodedValue'] = self.df['value'].astype(str)
        self.df['encodedValue'] = self.df['encodedValue'].str.replace('ON', '1.0')
        self.df['encodedValue'] = self.df['encodedValue'].str.replace('OFF', '-1.0')

        val = self.df['encodedValue']
        valList = val.unique()
        valList.sort()

        valuesDict = {}
        for i, v in enumerate(valList):
            valuesDict[v] = i + 1  # add one to reserve 0 value to padding

        # encoding previous avtivity
        activityDict, self.df['encodedPrevAct'] = self._encodeColumn(self.df['prevAct'])

        feature_1 = self.df.encodedSensor.values.astype(float)
        feature_2 = self.df.encodedValue.values.astype(float)
        feature_3 = self.df.dayOfWeek.values.astype(float)
        feature_4 = self.df.hour.values.astype(float)
        feature_5 = self.df.encodedPrevAct.values.astype(float)

        self.X = np.array([feature_1, feature_2, feature_3, feature_4, feature_5])
        self.X = self.X.transpose()

        # save de word dictionary
        self.eventDict = [sensorsDict, valuesDict, activityDict]

    def SurongFiveFeaturesPreviousActivityCodePaper(self):
        # set the encoding type
        self.encodingType = "SURONG_FIVE_FEATURES_PREVIOUS_ACT_CODE_PAPER"

        self.df["dayOfWeek"] = self.df['datetime'].dt.dayofweek.astype(int)
        self.df["hour"] = self.df['datetime'].dt.hour.astype(int)

        # generate previous activity
        self.__previousActivityAsCodePaper()

        # encoding sensors ID
        sensorsDict, self.df['encodedSensor'] = self._encodeColumn(self.df['sensor'])

        # encoding value sensors
        self.df['encodedValue'] = self.df['value'].astype(str)
        self.df['encodedValue'] = self.df['encodedValue'].str.replace('ON', '1.0')
        self.df['encodedValue'] = self.df['encodedValue'].str.replace('OFF', '-1.0')

        val = self.df['encodedValue']
        valList = val.unique()
        valList.sort()

        valuesDict = {}
        for i, v in enumerate(valList):
            valuesDict[v] = i + 1  # add one to reserve 0 value to padding

        # encoding previous avtivity
        activityDict, self.df['encodedPrevAct'] = self._encodeColumn(self.df['prevAct'])

        feature_1 = self.df.encodedSensor.values.astype(float)
        feature_2 = self.df.encodedValue.values.astype(float)
        feature_3 = self.df.dayOfWeek.values.astype(float)
        feature_4 = self.df.hour.values.astype(float)
        feature_5 = self.df.encodedPrevAct.values.astype(float)

        self.X = np.array([feature_1, feature_2, feature_3, feature_4, feature_5])
        self.X = self.X.transpose()

        # save de word dictionary
        self.eventDict = [sensorsDict, valuesDict, activityDict]

    def SurongFiveFeaturesPreviousEventActivityCodePaper(self):
        # set the encoding type
        self.encodingType = "SURONG_FIVE_FEATURES_PREVIOUS_EVENT_ACT_CODE_PAPER"

        self.df["dayOfWeek"] = self.df['datetime'].dt.dayofweek.astype(int)
        self.df["hour"] = self.df['datetime'].dt.hour.astype(int)

        # generate previous activity
        self.__directPreviousEventActivity()

        # encoding sensors ID
        sensorsDict, self.df['encodedSensor'] = self._encodeColumn(self.df['sensor'])

        # encoding value sensors
        self.df['encodedValue'] = self.df['value'].astype(str)
        self.df['encodedValue'] = self.df['encodedValue'].str.replace('ON', '1.0')
        self.df['encodedValue'] = self.df['encodedValue'].str.replace('OFF', '-1.0')

        val = self.df['encodedValue']
        valList = val.unique()
        valList.sort()

        valuesDict = {}
        for i, v in enumerate(valList):
            valuesDict[v] = i + 1  # add one to reserve 0 value to padding

        # encoding previous avtivity
        activityDict, self.df['encodedPrevAct'] = self._encodeColumn(self.df['prevAct'])

        feature_1 = self.df.encodedSensor.values.astype(float)
        feature_2 = self.df.encodedValue.values.astype(float)
        feature_3 = self.df.dayOfWeek.values.astype(float)
        feature_4 = self.df.hour.values.astype(float)
        feature_5 = self.df.encodedPrevAct.values.astype(float)

        self.X = np.array([feature_1, feature_2, feature_3, feature_4, feature_5])
        self.X = self.X.transpose()

        # save de word dictionary
        self.eventDict = [sensorsDict, valuesDict, activityDict]

    def fiveHandCrafted4(self):
        # set the encoding type
        self.encodingType = "5HCFEATURE"

        self.df["dayOfWeek"] = self.df['datetime'].dt.dayofweek.astype(int)
        self.df["hour"] = self.df['datetime'].dt.hour.astype(int)
        self.df["minute"] = self.df['datetime'].dt.minute.astype(int)

        # generate previous activity
        self.__previousAct()

        # encoding sensors ID
        sensorsDict, self.df['encodedSensor'] = self._encodeColumn(self.df['sensor'])

        # encoding value sensors
        # valuesDict, self.df['encodedValue'] = self._encodeColumn(self.df['value'])
        self.df['encodedValue'] = self.df['value'].astype(str)
        self.df['encodedValue'] = self.df['encodedValue'].str.replace('ON', '1.0')
        self.df['encodedValue'] = self.df['encodedValue'].str.replace('OFF', '-1.0')

        val = self.df['encodedValue']
        valList = val.unique()
        valList.sort()

        valuesDict = {}
        for i, v in enumerate(valList):
            valuesDict[v] = i + 1  # add one to reserve 0 value to padding

        # encoding previous avtivity
        activityDict, self.df['encodedPrevAct'] = self._encodeColumn(self.df['prevAct'])

        feature_1 = self.df.encodedSensor.values.astype(float)
        feature_2 = self.df.encodedValue.values.astype(float)
        feature_3 = self.df.dayOfWeek.values.astype(float)
        feature_4 = self.df.hour.values.astype(float) + np.round(self.df.minute.values.astype(float) / 60, 3)
        feature_5 = self.df.encodedPrevAct.values.astype(float)

        self.X = np.array([feature_1, feature_2, feature_3, feature_4, feature_5])
        self.X = self.X.transpose()

        # save de word dictionary
        self.eventDict = [sensorsDict, valuesDict, activityDict]

    def allSensorsOneRaw(self):

        # set the encoding type
        self.encodingType = "ALL_SENSORS_ONE_RAW"

        df2 = self.df.copy()
        df2["merge"] = df2["sensor"] + df2["value"]

        merges = df2["merge"].unique()
        merges.sort()

        sensors = df2["sensor"].unique()
        sensors.sort()

        merges = np.append(['<PAD>'],merges)

        valDict = {}
        for i, v in enumerate(merges):
            valDict[v] = i + 1  # add one to reserve 0 value to padding

        df2 = df2.pivot(index='datetime', columns='sensor', values='merge')
        df2 = df2.fillna(method='ffill')
        df2 = df2.fillna("<PAD>")

        for s in sensors:
            df2[s] = df2[s].map(valDict)

        self.X = df2.values
        #self.X = np.expand_dims(self.X, axis=2)
        #self.X = np.transpose(self.X, (0, 2, 1))

        # save de word dictionary
        self.eventDict = valDict

    def allSensorsOneRaw2(self):

        # set the encoding type
        self.encodingType = "ALL_SENSORS_ONE_RAW"

        df2 = self.df.copy()
        df2["merge"] = df2["sensor"] + df2["value"]

        merges = df2["merge"].unique()
        merges.sort()

        valDict = {}
        for i, v in enumerate(merges):
            valDict[v] = i + 1  # add one to reserve 0 value to padding

        df2 = df2.pivot(index='datetime', columns='sensor', values='merge')
        df2 = df2.fillna(method='ffill')
        df2 = df2.fillna(method='bfill')

        self.X = df2.values

        # save de word dictionary
        self.eventDict = valDict

    def allSensorsOneRaw3(self):

        # set the encoding type
        self.encodingType = "ALL_SENSORS_ONE_RAW3"

        df2 = self.df.copy()

        df2['value'] = df2['value'].astype(str)
        df2['value'] = df2['value'].str.replace('ON', '1.0')
        df2['value'] = df2['value'].str.replace('OFF', '0.0')
        df2['value'] = df2['value'].str.replace('OPEN', '1.0')
        df2['value'] = df2['value'].str.replace('CLOSE', '0.0')
        df2['value'] = df2['value'].astype(float)

        df2 = df2.pivot(index='datetime', columns='sensor', values='value')
        df2 = df2.fillna(method='ffill')
        df2 = df2.fillna(0)
        #df2 = df2.fillna(method='bfill')

        self.X = df2.values

        # save de word dictionary
        self.eventDict = None

    def allSensorsOneRaw4(self):

        seconds_in_day = 86400
        days_in_week = 7

        # set the encoding type
        self.encodingType = "ALL_SENSORS_ONE_RAW_TIME"

        df2 = self.df.copy()

        df2['value'] = df2['value'].astype(str)
        df2['value'] = df2['value'].str.replace('ON', '1.0')
        df2['value'] = df2['value'].str.replace('OFF', '-1.0')
        df2['value'] = df2['value'].str.replace('OPEN', '1.0')
        df2['value'] = df2['value'].str.replace('CLOSE', '-1.0')
        df2['value'] = df2['value'].astype(float)

        df2 = df2.pivot(index='datetime', columns='sensor', values='value')
        df2 = df2.fillna(method='ffill')
        #df2 = df2.fillna(method='bfill')
        df2 = df2.fillna(-1.0)


        self.df["sinceMidnight"] = self.df['datetime'].dt.hour.astype(int) * 3600 + self.df['datetime'].dt.minute.astype(int) * 60 + self.df['datetime'].dt.second.astype(int)

        self.df["day_of_the_week"] = self.df['datetime'].dt.dayofweek.astype(int)


        self.df['sin_seconds'] = np.sin(2*np.pi*self.df["sinceMidnight"]/seconds_in_day)
        self.df['cos_seconds'] = np.cos(2*np.pi*self.df["sinceMidnight"]/seconds_in_day)

        self.df['sin_day_of_the_week'] = np.sin(2*np.pi*self.df["day_of_the_week"]/days_in_week)
        self.df['cos_day_of_the_week'] = np.cos(2*np.pi*self.df["day_of_the_week"]/days_in_week)


        feature_2 = np.expand_dims(np.array(self.df.sin_seconds.values.astype(float)), axis=1)
        feature_3 = np.expand_dims(np.array(self.df.cos_seconds.values.astype(float)), axis=1)
        feature_4 = np.expand_dims(np.array(self.df.sin_day_of_the_week.values.astype(float)), axis=1)
        feature_5 = np.expand_dims(np.array(self.df.cos_day_of_the_week.values.astype(float)), axis=1)

        self.X = df2.values

        self.X = np.append(self.X, feature_2, axis=1)
        self.X = np.append(self.X, feature_3, axis=1)
        self.X = np.append(self.X, feature_4, axis=1)
        self.X = np.append(self.X, feature_5, axis=1)

        # save de word dictionary
        self.eventDict = None


    def allSensorsOneRaw5(self):

        seconds_in_day = 24
        days_in_week = 7

        # set the encoding type
        self.encodingType = "ALL_SENSORS_ONE_RAW_TIME"

        df2 = self.df.copy()

        df2['value'] = df2['value'].astype(str)
        df2['value'] = df2['value'].str.replace('ON', '1.0')
        df2['value'] = df2['value'].str.replace('OFF', '-1.0')
        df2['value'] = df2['value'].str.replace('OPEN', '1.0')
        df2['value'] = df2['value'].str.replace('CLOSE', '-1.0')
        df2['value'] = df2['value'].astype(float)

        df2 = df2.pivot(index='datetime', columns='sensor', values='value')
        df2 = df2.fillna(method='ffill')
        #df2 = df2.fillna(method='bfill')
        df2 = df2.fillna(-1.0)


        self.df["hour"] = self.df['datetime'].dt.hour.astype(int)

        self.df["day_of_the_week"] = self.df['datetime'].dt.dayofweek.astype(int)


        self.df['sin_seconds'] = np.sin(2*np.pi*self.df["hour"]/seconds_in_day)
        self.df['cos_seconds'] = np.cos(2*np.pi*self.df["hour"]/seconds_in_day)

        self.df['sin_day_of_the_week'] = np.sin(2*np.pi*self.df["day_of_the_week"]/days_in_week)
        self.df['cos_day_of_the_week'] = np.cos(2*np.pi*self.df["day_of_the_week"]/days_in_week)


        feature_2 = np.expand_dims(np.array(self.df.sin_seconds.values.astype(float)), axis=1)
        feature_3 = np.expand_dims(np.array(self.df.cos_seconds.values.astype(float)), axis=1)
        feature_4 = np.expand_dims(np.array(self.df.sin_day_of_the_week.values.astype(float)), axis=1)
        feature_5 = np.expand_dims(np.array(self.df.cos_day_of_the_week.values.astype(float)), axis=1)

        self.X = df2.values

        self.X = np.append(self.X, feature_2, axis=1)
        self.X = np.append(self.X, feature_3, axis=1)
        self.X = np.append(self.X, feature_4, axis=1)
        self.X = np.append(self.X, feature_5, axis=1)

        # save de word dictionary
        self.eventDict = None


    def rawSequenceFeatures(self):
            # set the encoding type
            self.encodingType = "NLPMIDNIGHT"

            self.df["sinceMidnight"] = self.df['datetime'].dt.hour.astype(int) * 3600 + self.df[
                'datetime'].dt.minute.astype(int) * 60 + self.df['datetime'].dt.second.astype(int)
            self.df["sinceMidnight"] = self.df["sinceMidnight"].astype(str)
            self.df["merge"] = self.df['sensor'] + self.df['value'].astype(str)
            self.df['nlp'] = self.df["merge"] + " " + self.df["sinceMidnight"]

            sentence = " ".join(self.df['nlp'].values)

            tokenizer = Tokenizer(filters='')
            tokenizer.fit_on_texts([sentence])

            wordDict = tokenizer.word_index

            self.df['encodedMerge'] = self.df['merge'].str.lower()
            # self.df['encodedMerge'] = self.df['encodedMerge'].map(wordDict)
            _, self.df['encodedMerge'] = self._encodeColumn(self.df["encodedMerge"], wordDict)

            self.df['encodedSinceMidnight'] = self.df['sinceMidnight']
            # self.df['encodedSinceMidnight'] = self.df['encodedSinceMidnight'].map(wordDict)
            _, self.df['encodedSinceMidnight'] = self._encodeColumn(self.df["encodedSinceMidnight"], wordDict)

            # self.df['encodedNLP'] =  self.df["encodedMerge"].astype(str) + " " + self.df["encodedSinceMidnight"].astype(str)

            # self.X = " ".join(self.df['encodedNLP'].values)
            # self.X = self.X.split()
            # self.X = list(map(int, self.X))

            feature_1 = self.df.encodedMerge.values.astype(int)
            feature_2 = self.df.encodedSinceMidnight.values.astype(int)

            self.X = np.array([feature_1, feature_2])
            self.X = self.X.transpose()

            # save de word dictionary
            self.eventDict = wordDict

    def eventEmbeddingRaw(self):

        # set the encoding type
        self.encodingType = "EVENT_EMBEDDING"

        self.df["hour"] = self.df['datetime'].dt.hour.astype(str)
        self.df["minutes"] = self.df['datetime'].dt.minute.astype(str)
        self.df["seconds"] = self.df['datetime'].dt.second.astype(str)
        self.df["dayOfWeek"] = self.df['datetime'].dt.day_name()

        #Add unit values
        listvalues = self.df.value.values

        units = []
        for v in listvalues:
            # if temperature then add degrees:
            if not v in ["ON", "OFF", "OPEN", "CLOSE"]:
                units.append("degrees")
            else:
                units.append("")

        self.df['units'] = units
        self.df['units'] = self.df['units'].astype(str)


        self.df["hours_str"] = self.df["hour"] + "heure"
        self.df["minute_str"] = self.df["minutes"] + "minute"
        self.df["second_str"] = self.df["seconds"] + "second"
        self.df["value_str"] = self.df["value"] + self.df['units']

        feature_1 = self.df.dayOfWeek.values.astype(str)
        feature_2 = self.df.hours_str.values.astype(str)
        feature_3 = self.df.minute_str.values.astype(str)
        feature_4 = self.df.second_str.values.astype(str)
        feature_5 = self.df.sensor.values.astype(str)
        feature_6 = self.df.value_str.values.astype(str)

        self.X = np.array([feature_1, feature_2, feature_3, feature_4, feature_5, feature_6])
        self.X = self.X.transpose()


    def eventEmbeddingRaw2(self):

        # set the encoding type
        self.encodingType = "EVENT_EMBEDDING"

        self.df["hour"] = self.df['datetime'].dt.hour.astype(str)
        self.df["dayOfWeek"] = self.df['datetime'].dt.day_name()

        #Add unit values
        listvalues = self.df.value.values

        units = []
        for v in listvalues:
            # if temperature then add degrees:
            if not v in ["ON", "OFF", "OPEN", "CLOSE"]:
                units.append("degrees")
            else:
                units.append("")

        self.df['units'] = units
        self.df['units'] = self.df['units'].astype(str)


        self.df["hours_str"] = self.df["hour"] + "heure"
        self.df["value_str"] = self.df["value"] + self.df['units']

        feature_1 = self.df.dayOfWeek.values.astype(str)
        feature_2 = self.df.hours_str.values.astype(str)
        feature_3 = self.df.sensor.values.astype(str)
        feature_4 = self.df.value_str.values.astype(str)

        self.X = np.array([feature_1, feature_2, feature_3, feature_4])
        self.X = self.X.transpose()

    def eventEmbeddingRaw3(self):

        # set the encoding type
        self.encodingType = "EVENT_EMBEDDING"

        self.df["hour"] = self.df['datetime'].dt.hour.astype(str)

        #Add unit values
        listvalues = self.df.value.values

        units = []
        for v in listvalues:
            # if temperature then add degrees:
            if not v in ["ON", "OFF", "OPEN", "CLOSE"]:
                units.append("degrees")
            else:
                units.append("")

        self.df['units'] = units
        self.df['units'] = self.df['units'].astype(str)


        self.df["hours_str"] = self.df["hour"] + "heure"
        self.df["value_str"] = self.df["value"] + self.df['units']

        feature_1 = self.df.hours_str.values.astype(str)
        feature_2 = self.df.sensor.values.astype(str)
        feature_3 = self.df.value_str.values.astype(str)

        self.X = np.array([feature_1, feature_2, feature_3])
        self.X = self.X.transpose()

    def eventEmbeddingRaw4(self):

        # set the encoding type
        self.encodingType = "EVENT_EMBEDDING"

        self.df["sinceMidnight"] = self.df['datetime'].dt.hour.astype(int) * 60 + self.df[
            'datetime'].dt.minute.astype(int)

        #Add unit values
        listvalues = self.df.value.values

        units = []
        for v in listvalues:
            # if temperature then add degrees:
            if not v in ["ON", "OFF", "OPEN", "CLOSE"]:
                units.append("degrees")
            else:
                units.append("")

        self.df['units'] = units
        self.df['units'] = self.df['units'].astype(str)

        self.df["value_str"] = self.df["value"] + self.df['units']

        feature_1 = self.df.sinceMidnight.values.astype(str)
        feature_2 = self.df.sensor.values.astype(str)
        feature_3 = self.df.value_str.values.astype(str)

        self.X = np.array([feature_1, feature_2, feature_3])
        self.X = self.X.transpose()


    def eventEmbeddingRaw5(self):

        # set the encoding type
        self.encodingType = "EVENT_EMBEDDING"

        self.df["sinceMidnight"] = self.df['datetime'].dt.hour.astype(int) * 3600 + self.df[
            'datetime'].dt.minute.astype(int) * 60 + self.df['datetime'].dt.second.astype(int)

        #Add unit values
        listvalues = self.df.value.values

        units = []
        for v in listvalues:
            # if temperature then add degrees:
            if not v in ["ON", "OFF", "OPEN", "CLOSE"]:
                units.append("degrees")
            else:
                units.append("")

        self.df['units'] = units
        self.df['units'] = self.df['units'].astype(str)

        self.df["value_str"] = self.df["value"] + self.df['units']

        feature_1 = self.df.sinceMidnight.values.astype(str)
        feature_2 = self.df.sensor.values.astype(str)
        feature_3 = self.df.value_str.values.astype(str)

        self.X = np.array([feature_1, feature_2, feature_3])
        self.X = self.X.transpose()


    def eventEmbeddingRaw6(self):

        # set the encoding type
        self.encodingType = "EVENT_EMBEDDING"

        self.df["sinceMidnight"] = self.df['datetime'].dt.hour.astype(int) * 3600 + self.df[
            'datetime'].dt.minute.astype(int) * 60 + self.df['datetime'].dt.second.astype(int)

        self.df['sensor_value'] = self.df['sensor'] + self.df['value'].astype(str)

        feature_1 = self.df.sinceMidnight.values.astype(str)
        feature_2 = self.df.sensor_value.values.astype(str)

        self.X = np.array([feature_1, feature_2])
        self.X = self.X.transpose()


    def eventEmbeddingSixFeature(self):

        # set the encoding type
        self.encodingType = "EVENT_EMBEDDING"

        self.df["hour"] = self.df['datetime'].dt.hour.astype(str)
        self.df["minutes"] = self.df['datetime'].dt.minute.astype(str)
        self.df["seconds"] = self.df['datetime'].dt.second.astype(str)
        self.df["dayOfWeek"] = self.df['datetime'].dt.day_name()

        #Add unit values
        listvalues = self.df.value.values

        units = []
        for v in listvalues:
            # if temperature then add degrees:
            if not v in ["ON", "OFF", "OPEN", "CLOSE"]:
                units.append("degrees")
            else:
                units.append("")

        self.df['units'] = units
        self.df['units'] = self.df['units'].astype(str)

        self.df["hours_str"] = self.df["hour"] + "heure"
        self.df["minute_str"] = self.df["minutes"] + "minute"
        self.df["second_str"] = self.df["seconds"] + "second"
        self.df["value_str"] = self.df["value"] + self.df['units']

        self.df['event_str'] = self.df["dayOfWeek"] + " " + self.df["hours_str"] + " " + self.df["minute_str"] + " " + self.df["second_str"] + " " + self.df["sensor"] + " " + self.df["value_str"]

        sentence = " ".join(self.df['event_str'].values)

        tokenizer = Tokenizer(filters='')
        tokenizer.fit_on_texts([sentence])

        wordDict = tokenizer.word_index

        _, self.df['dayOfWeek'] = self._encodeColumn(self.df["dayOfWeek"], wordDict)
        _, self.df['hours_str'] = self._encodeColumn(self.df["hours_str"], wordDict)
        _, self.df['minute_str'] = self._encodeColumn(self.df["minute_str"], wordDict)
        _, self.df['second_str'] = self._encodeColumn(self.df["second_str"], wordDict)
        _, self.df['sensor'] = self._encodeColumn(self.df["sensor"], wordDict)
        _, self.df['value_str'] = self._encodeColumn(self.df["value_str"], wordDict)

        feature_1 = self.df.dayOfWeek.values.astype(int)
        feature_2 = self.df.hours_str.values.astype(int)
        feature_3 = self.df.minute_str.values.astype(int)
        feature_4 = self.df.second_str.values.astype(int)
        feature_5 = self.df.sensor.values.astype(int)
        feature_6 = self.df.value_str.values.astype(int)

        self.X = np.array([feature_1, feature_2, feature_3, feature_4, feature_5, feature_6])
        self.X = self.X.transpose()

        # save de word dictionary
        self.eventDict = wordDict


    def eventEmbedding5(self):

        # set the encoding type
        self.encodingType = "EVENT_EMBEDDING_5"

        self.df["sinceMidnight"] = self.df['datetime'].dt.hour.astype(int) * 3600 + self.df[
            'datetime'].dt.minute.astype(int) * 60 + self.df['datetime'].dt.second.astype(int)

        #Add unit values
        listvalues = self.df.value.values

        units = []
        for v in listvalues:
            # if temperature then add degrees:
            if not v in ["ON", "OFF", "OPEN", "CLOSE"]:
                units.append("degrees")
            else:
                units.append("")

        self.df['units'] = units
        self.df['units'] = self.df['units'].astype(str)

        self.df["value_str"] = self.df["value"] + self.df['units']


        # Create word dictionary
        self.df["event_str"] = self.df.sinceMidnight.astype(str) + " " + self.df.sensor.astype(str) + " " + self.df.value_str.astype(str)

        sentence = " ".join(self.df['event_str'].values)

        tokenizer = Tokenizer(filters='', lower=False)
        tokenizer.fit_on_texts([sentence])

        wordDict = tokenizer.word_index

        # Encode
        self.df['sinceMidnight_encoded'] = self.df['sinceMidnight'].astype(str)
        self.df['sensor_encoded'] = self.df['sensor'].astype(str)
        self.df['value_encoded'] = self.df['value_str'].astype(str)

        _, self.df['sinceMidnight_encoded'] = self._encodeColumn(self.df["sinceMidnight_encoded"], wordDict)
        _, self.df['sensor_encoded'] = self._encodeColumn(self.df["sensor_encoded"], wordDict)
        _, self.df['value_encoded'] = self._encodeColumn(self.df["value_encoded"], wordDict)

        # Get features
        feature_1 = self.df.sinceMidnight_encoded.values.astype(int)
        feature_2 = self.df.sensor_encoded.values.astype(int)
        feature_3 = self.df.value_encoded.values.astype(int)

        self.X = np.array([feature_1, feature_2, feature_3])
        self.X = self.X.transpose()

        # save de word dictionary
        self.eventDict = wordDict


    def basic_raw(self):

        # set the encoding type
        self.encodingType = "BASIC_RAW"

        # create event tokens
        self.df['merge'] = self.df['sensor'].astype(str) + self.df['value'].astype(str)

        feature_1 = self.df['merge'].values.astype(str)

        self.X = np.array([feature_1])
        self.X = self.X.transpose()


    def basic_raw_encoded(self, custom_dict = None):

        # set the encoding type
        self.encodingType = "BASIC_RAW_ENCODED"

        # create event tokens
        self.df['merge'] = self.df['sensor'].astype(str) + self.df['value'].astype(str)


        if custom_dict == None:

            sentence = " ".join(self.df['merge'].values)

            tokenizer = Tokenizer(filters='', lower=False)
            tokenizer.fit_on_texts([sentence])

            wordDict = tokenizer.word_index
        else:
            wordDict = custom_dict

        self.df['merge_encoded'] = self.df['merge'].values.astype(str)

        _, self.df['merge_encoded'] = self._encodeColumn(self.df["merge_encoded"], wordDict)

        feature_1 = self.df.merge_encoded.values.astype(int)

        self.X = np.array([feature_1])
        self.X = self.X.transpose()

        # save de word dictionary
        self.eventDict = wordDict


    def basic_raw_time(self):

        # set the encoding type
        self.encodingType = "BASIC_RAW_TIME"

        # create event tokens
        self.df["sinceMidnight"] = self.df['datetime'].dt.hour.astype(int) * 3600 + self.df['datetime'].dt.minute.astype(int) * 60 + self.df['datetime'].dt.second.astype(int)
        self.df['merge'] = self.df['sensor'].astype(str) + self.df['value'].astype(str)
        
        feature_1 = self.df["sinceMidnight"].values.astype(str)
        feature_2 = self.df['merge'].values.astype(str)

        self.X = np.array([feature_1, feature_2])
        self.X = self.X.transpose()



    def basic_raw_time_encoded(self, custom_dict = None, lower = False):

        # set the encoding type
        self.encodingType = "BASIC_RAW_TIME_ENCODED"

        seconds_in_day = 86400
        days_in_week = 7

        # create event tokens
        self.df['merge'] = self.df['sensor'].astype(str) + self.df['value'].astype(str)

        self.df["sinceMidnight"] = self.df['datetime'].dt.hour.astype(int) * 3600 + self.df['datetime'].dt.minute.astype(int) * 60 + self.df['datetime'].dt.second.astype(int)

        self.df["day_of_the_week"] = self.df['datetime'].dt.dayofweek.astype(int)


        sentence = " ".join(self.df['merge'].values)

        tokenizer = Tokenizer(filters='', lower=lower)
        tokenizer.fit_on_texts([sentence])

        wordDict = tokenizer.word_index

        self.df['merge_encoded'] = self.df['merge'].values.astype(str)

        _, self.df['merge_encoded'] = self._encodeColumn(self.df["merge_encoded"], wordDict)

        self.df['sin_seconds'] = np.sin(2*np.pi*self.df["sinceMidnight"]/seconds_in_day)
        self.df['cos_seconds'] = np.cos(2*np.pi*self.df["sinceMidnight"]/seconds_in_day)

        self.df['sin_day_of_the_week'] = np.sin(2*np.pi*self.df["day_of_the_week"]/days_in_week)
        self.df['cos_day_of_the_week'] = np.cos(2*np.pi*self.df["day_of_the_week"]/days_in_week)


        feature_1 = self.df.merge_encoded.values.astype(int)
        feature_2 = self.df.sin_seconds.values.astype(float)
        feature_3 = self.df.cos_seconds.values.astype(float)
        feature_4 = self.df.sin_day_of_the_week.values.astype(float)
        feature_5 = self.df.cos_day_of_the_week.values.astype(float)

        self.X = np.array([feature_1,feature_2,feature_3,feature_4,feature_5], dtype="object")
        self.X = self.X.transpose()

        # save de word dictionary
        self.eventDict = wordDict


    def basic_raw_hour_encoded(self, custom_dict = None, lower = False):

        # set the encoding type
        self.encodingType = "BASIC_RAW_HOUR_ENCODED"

        # create event tokens
        self.df['merge'] = self.df['datetime'].dt.hour.astype(str) + self.df['sensor'].astype(str) + self.df['value'].astype(str)

        if custom_dict == None:

            sentence = " ".join(self.df['merge'].values)

            tokenizer = Tokenizer(filters='', lower=False)
            tokenizer.fit_on_texts([sentence])

            wordDict = tokenizer.word_index
        else:
            wordDict = custom_dict

        self.df['merge_encoded'] = self.df['merge'].values.astype(str)

        _, self.df['merge_encoded'] = self._encodeColumn(self.df["merge_encoded"], wordDict)

        feature_1 = self.df.merge_encoded.values.astype(int)

        self.X = np.array([feature_1], dtype="object")
        self.X = self.X.transpose()

        # save de word dictionary
        self.eventDict = wordDict

    def basic_raw_hour(self, custom_dict = None, lower = False):

        # set the encoding type
        self.encodingType = "BASIC_RAW_HOUR"

        # create event tokens
        self.df['merge'] = self.df['datetime'].dt.hour.astype(str) + self.df['sensor'].astype(str) + self.df['value'].astype(str)

        feature_1 = self.df['merge'].values.astype(str)

        self.X = np.array([feature_1], dtype="object")
        self.X = self.X.transpose()

    def basic_raw_minutes_encoded(self, custom_dict = None, lower = False):

        # set the encoding type
        self.encodingType = "BASIC_RAW_HOUR_ENCODED"

        # create event tokens
        self.df['merge'] = self.df['datetime'].dt.minute.astype(str) + self.df['sensor'].astype(str) + self.df['value'].astype(str)

        if custom_dict == None:

            sentence = " ".join(self.df['merge'].values)

            tokenizer = Tokenizer(filters='', lower=False)
            tokenizer.fit_on_texts([sentence])

            wordDict = tokenizer.word_index
        else:
            wordDict = custom_dict

        self.df['merge_encoded'] = self.df['merge'].values.astype(str)

        _, self.df['merge_encoded'] = self._encodeColumn(self.df["merge_encoded"], wordDict)

        feature_1 = self.df.merge_encoded.values.astype(int)

        self.X = np.array([feature_1])
        self.X = self.X.transpose()

        # save de word dictionary
        self.eventDict = wordDict

    def basic_raw_minutes(self, lower = False):

        # set the encoding type
        self.encodingType = "BASIC_RAW_MINUTES"

        # create event tokens
        self.df['merge'] = self.df['datetime'].dt.minute.astype(str) + self.df['sensor'].astype(str) + self.df['value'].astype(str)

        feature_1 = self.df['merge'].values.astype(str)

        self.X = np.array([feature_1])
        self.X = self.X.transpose()

    
    def basic_raw_time_2(self):

        # set the encoding type
        self.encodingType = "BASIC_RAW_TIME_2"

        # create event tokens
        self.df["sinceMidnight"] = self.df['datetime'].dt.hour.astype(int) * 3600 + self.df['datetime'].dt.minute.astype(int) * 60 + self.df['datetime'].dt.second.astype(int)
        self.df['merge'] = self.df['sensor'].astype(str) + self.df['value'].astype(str)

        feature_1 = self.df["sinceMidnight"].values.astype(str)
        feature_2 = self.df['merge'].values.astype(str)

        x_tmp = []
        y_tmp = []

        for i in range(len(feature_1)):
            x_tmp.append(feature_1[i])
            x_tmp.append(feature_2[i])

            # double the Y because the two features had the same label as they are part of the same event
            y_tmp.append(self.Y[i])
            y_tmp.append(self.Y[i])

        self.X = np.array(x_tmp).astype(str)
        self.Y = np.array(y_tmp).astype(int)


    def basic_raw_time_encoded_2(self, custom_dict = None):

        # set the encoding type
        self.encodingType = "BASIC_RAW_TIME_ENCODED_2"

        self.basic_raw_time_2()

        # if no dict is provided, gerere a dict
        if custom_dict == None:

            sentence = " ".join(self.X)

            tokenizer = Tokenizer(filters='', lower=False)
            tokenizer.fit_on_texts([sentence])

            wordDict = tokenizer.word_index
        else:
            wordDict = custom_dict

        self.X = [wordDict[word] for word in self.X]

        self.X = np.array(self.X).astype(int)

        # save de word dictionary
        self.eventDict = wordDict