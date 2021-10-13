#!/usr/bin/env python3

import numpy as np

def into_days(dataset):

    df = dataset.df
    
    days = []

    transitionIndex = df.datetime.dt.date.ne(df.datetime.dt.date.shift())

    ii = np.where(transitionIndex == True)[0]

    for i,end in enumerate(ii):
        if i > 0 :

            start = ii[i-1]
            #print("START: {}".format(start))
            #print("END: {}".format(end))
            #input("Press Enter to continue...")

            days.append(df[start:end])



    lastDay = df[end:]
    days.append(lastDay)
    
    return days