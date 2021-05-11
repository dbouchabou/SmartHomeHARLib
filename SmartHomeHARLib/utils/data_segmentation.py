# coding: utf-8
# !/usr/bin/env python3

import os
import sys
import time
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.utils import *

def dataframe_day_window(df):

    dofw = []

    chunksX = []
    chunksY = []


    transitionIndex = df.datetime.dt.date.ne(df.datetime.dt.date.shift())

    ii = np.where(transitionIndex == True)[0]

    for i,end in enumerate(ii):
        if i > 0 :

            start = ii[i-1]
            #print("START: {}".format(start))
            #print("END: {}".format(end))
            #input("Press Enter to continue...")

            day = df[start:end]
            dofw.append(day.datetime.dt.dayofweek.values[-1])

            chunksX.append(day)



    lastDay = df[end:]
    dofw.append(lastDay.datetime.dt.dayofweek.values[-1])

    chunksX.append(lastDay)

    return dofw, chunksX

def sliding_window(x, y, window_size, step=1, last_label_value = False, next_label_value = False):

    chunks_x = []
    chunks_y = []

    if next_label_value:
        window_size = window_size + 1
        last_label_value = True

    num_of_chunks = int(((x.shape[0] - window_size) / step) + 1)

    for i in range(0, num_of_chunks * step, step):

        if next_label_value:
            chunks_x.append(x[i:i + window_size-1])
        else:
            chunks_x.append(x[i:i + window_size])

        if last_label_value:
            chunks_y.append(y[i:i + window_size][-1])
        else:
            chunks_y.append(y[i:i + window_size])

    return np.array(chunks_x), np.array(chunks_y)

def sliding_sensor_event_windows(x, y, window_size, step=1):

    chunks_x, chunks_y = sliding_window(x, y, window_size, step, last_label_value = True)

    return np.array(chunks_x), np.array(chunks_y)

def sensor_event_windows(x, y, window_size):

    chunks_x, chunks_y = sliding_window(x, y, window_size, window_size, last_label_value = True)

    return np.array(chunks_x), np.array(chunks_y)


def progressive_sliding_window(x, y, window_size, begin = 1, step=1, label_value_last = False, label_value_next = False):

    chunks_x = []
    chunks_y = []

    if label_value_next:
        limit = -1
    elif label_value_last:
        limit = 1
    else:
        limit = 0

    # Special case of sequence with one event
    if len(x) < 2:
        chunks_x.append(x)
        chunks_y.append(y)


    for i in range(begin,len(x)+limit, step): 

            if i-window_size <= 0:
                chunk = x[:i]
                prediction = y[:i]
            else:
                chunk = x[i-window_size:i]
                prediction = y[i-window_size:i]

            # Label type choise by default label are labels of each element of x
            if label_value_next:
                prediction = y[i]

            elif label_value_last:
                prediction = y[i-1]

            chunks_x.append(chunk)
            chunks_y.append(prediction)

    return np.array(chunks_x), np.array(chunks_y)