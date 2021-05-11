# coding: utf-8
# !/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import re

from SmartHomeHARLib.datasets import DatasetSegmentator


class Segmentator(DatasetSegmentator):

    def __init__(self, datasetEncoder):

        #if datasetEncoder.dataset.datasetType != "CASAS":
        #    raise ValueError('Argument should be CASASDatasetEncoder type')

        super().__init__(datasetEncoder)

    def explicitWindow(self):

        # Reinitialize in case
        if self.segmentationType != None:
            self.__init__(self.encoder)

        self.segmentationType = "EW"

        chunksX = []
        chunksY = []

        transitionIndex = self.df.activity.ne(self.df.activity.shift())

        ii = np.where(transitionIndex == True)[0]

        for i, end in enumerate(ii):
            if i > 0:
                start = ii[i - 1]

                # activitySeq = self.df[start:end]
                if(len(self.X[start:end])>1):
                    chunksX.append(self.X[start:end])
                    chunksY.append(self.Y[start:end][-1])

        # lastActivitySeq = self.df[end:]

        if(len(self.X[end:])>1):
            chunksX.append(self.X[end:])
            chunksY.append(self.Y[end:][-1])

        self.X = chunksX
        self.Y = chunksY

    def __slidingWindow(self, X, Y, winSize, step=1):

        chunksX = []
        chunksY = []

        numOfChunks = int(((X.shape[0] - winSize) / step) + 1)

        for i in range(0, numOfChunks * step, step):
            # sew = self.df[i:i+winSize]

            chunksX.append(X[i:i + winSize])
            chunksY.append(Y[i:i + winSize][-1])

        return chunksX, chunksY

    def __slidingWindow2(self, X, Y, winSize, step=1):

        chunksX = []
        chunksY = []

        numOfChunks = int(((X.shape[0] - winSize) / step) + 1)

        for i in range(0, numOfChunks * step, step):
            # sew = self.df[i:i+winSize]

            chunksX.append(X[i:i + winSize])
            chunksY.append(Y[i:i + winSize])

        return chunksX, chunksY

    def slidingSensorEventWindows(self, winSize, step=1):

        # Reinitialize in case
        if self.segmentationType != None:
            self.__init__(self.encoder)

        self.segmentationType = "SSEW"

        chunksX, chunksY = self.__slidingWindow(self.X, self.Y, winSize, step)

        self.X = chunksX
        self.Y = chunksY

    def sensorEventWindows(self, winSize):

        # Reinitialize in case
        if self.segmentationType != None:
            self.__init__(self.encoder)

        self.slidingSensorEventWindows(winSize, winSize)

        self.segmentationType = "SEW"

    def slidingTimeWindow(self, interval, step=1):

        # Reinitialize in case
        if self.segmentationType != None:
            self.__init__(self.encoder)

        pass

    def timeWindow(self, interval):

        # Reinitialize in case
        if self.segmentationType != None:
            self.__init__(self.encoder)

        self.slidingTimeWindow(interval, interval)

        self.segmentationType = "TW"

    def daySlidingSensorEventWindows(self, winSize, step=1):

        # Reinitialize in case
        if self.segmentationType != None:
            self.__init__(self.encoder)

        self.segmentationType = "DAYSSEW"

        chunksByDayX = []
        chunksByDayY = []

        dofw = self.dayWindows(dayofweek=True)

        for dayX, dayY in zip(self.X, self.Y):
            chunksX, chunksY = self.__slidingWindow(dayX, dayY, winSize, step)

            chunksByDayX.append(chunksX)
            chunksByDayY.append(chunksY)

        self.X = chunksByDayX
        self.Y = chunksByDayY

        return dofw


    def daySlidingSensorEventWindows2(self, winSize, step=1):

        # Reinitialize in case
        if self.segmentationType != None:
            self.__init__(self.encoder)

        self.segmentationType = "DAYSSEW"

        chunksByDayX = []
        chunksByDayY = []

        dofw = self.dayWindows(dayofweek=True)

        for dayX, dayY in zip(self.X, self.Y):
            chunksX, chunksY = self.__slidingWindow2(dayX, dayY, winSize, step)

            chunksByDayX.append(chunksX)
            chunksByDayY.append(chunksY)

        self.X = chunksByDayX
        self.Y = chunksByDayY

        return dofw
