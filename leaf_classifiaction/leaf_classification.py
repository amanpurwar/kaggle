""" Writing leaf_classification.
Author : NaHopayega
Date: 20th dec 2016

"""
import pandas as pd
import numpy as np
import random
import csv as csv
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model, decomposition, datasets
from collections import Counter
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from xgboost.sklearn import XGBClassifier
from imblearn.over_sampling import SMOTE

def prepTrainingData():
    global ids
    #loading data into dataframe
    train_df = pd.read_csv('train.csv', header=0)
