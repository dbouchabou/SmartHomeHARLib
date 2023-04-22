# coding: utf-8
# !/usr/bin/env python3
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.utils import *

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix


class Evaluator(object):

    def __init__(self, testX, testY, model=None, model_path=None, custom_objects=None, model_type="keras"):

        self.X_test_input = testX

        self.Y_test_input = testY

        self.Y_pred = []

        self.ascore = 0.0
        self.report = None
        self.cm = None
        self.bscore = 0.0
        self.model_type = model_type

        if model != None:
            self.saved_model = model

        if model_path != None:
            self.saved_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

        if self.model_type == "keras":
            print(self.saved_model.summary())

    def evaluate(self):
        Y_hat = self.saved_model.predict(self.X_test_input)

        if self.model_type == "keras":
            self.Y_pred = np.argmax(Y_hat, axis=1)
            self.Y_pred = self.Y_pred.astype('int32')
        else:
            self.Y_pred = Y_hat

    def simpleEvaluation(self, batch_size, Y_test_input=None, verbose=False):

        if len(Y_test_input) < 1:
            Y = self.Y_test_input
        else:
            Y = Y_test_input

        # evaluate model
        _, self.ascore = self.saved_model.evaluate(self.X_test_input, Y, batch_size=batch_size, verbose=verbose)

    def classificationReport(self, listActivities, labels):

        self.report = classification_report(self.Y_test_input, self.Y_pred, target_names=listActivities, digits=4,
                                            labels=labels, output_dict=True)

    def confusionMatrix(self):

        self.cm = confusion_matrix(self.Y_test_input, self.Y_pred)

    def multi_label_confusion_matrix(self):

        self.cm = multilabel_confusion_matrix(self.Y_test_input, self.Y_pred)

    def accuracyCompute(self):

        self.ascore = accuracy_score(self.Y_test_input, self.Y_pred)

    def balanceAccuracyCompute(self):

        self.bscore = balanced_accuracy_score(self.Y_test_input, self.Y_pred)

    def saveClassificationReport(self, pathResults):
        df = pd.DataFrame(self.report).transpose()
        df.to_csv(pathResults, sep='\t', encoding='utf-8')

    def saveConfusionMatrix(self, pathResults):
        df = pd.DataFrame(self.cm)
        df.to_csv(pathResults, sep='\t', encoding='utf-8', header=False, index=False)
