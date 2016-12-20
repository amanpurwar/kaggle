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

# global variables
Ports_dict = {}
ids=None

def prepTrainingData():
    global ids
    #loading data into dataframe
    train_df = pd.read_csv('train.csv', header=0)

    #drop the id column
    train_df=train_df.drop('id',axis=1)

    #replacing the species with a number/categorising them
    Ports = list(enumerate(np.unique(train_df['species'])))    # determine all values of species,
    Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
    train_df.species = train_df.species.map( lambda x: Ports_dict[x]).astype(int)     # Convert all species strings to int
    print(Ports_dict)

    train_data=train_df.values

    return train_data[:,1:],train_data[:,0]

def prepareTestData():
    global Ports_dict,ids
    #test data file load
    test_df=pd.read_csv('test.csv',header=0)

    ids=test_df['id'].values # collecting id of test data before dropping id field
    test_df=test_df.drop('id',axis=1)

    #converting to numpy arr
    test_data=test_df.values

    return test_data

def TestArea(algo_str,X_train,y_train,X_test):
    print 'testing started...'
    if algo_str is "rforest":
        forest = RandomForestClassifier(n_estimators=250,criterion='entropy',min_samples_split=10,max_features=4)
        forest = forest.fit( X_train, y_train )
        #for printing the importances
        '''
        importances = forest.feature_importances_
        indices = np.argsort(importances)[::-1]
        for f in range(0,5):
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
        '''
        
        clf=forest

    elif algo_str is "logistic":
        logreg = linear_model.LogisticRegression(tol=1e-5,C=1e5,solver='liblinear',max_iter=500)
        logreg = logreg.fit(X_train, y_train)
        clf = logreg

    elif algo_str is "svm":
        #for SVC in place of linearSVC
        '''
        clf = SVC()
        clf=clf.fit(train_data[0::,1::], train_data[0::,0])
        
        '''
        clf=LinearSVC(C=1e3,dual=False,random_state=42)
        clf=clf.fit(X_train, y_train)

    elif algo_str is "xgboost":
        smte=SMOTE(random_state=42,kind='borderline1')
        X_smote,y_smote=smte.fit_sample(X_train,y_train)
        clf=XGBClassifier(n_estimators=160)
        clf=clf.fit(X_smote,y_smote)
        

    print 'Predicting...'
    output = clf.predict(X_test).astype(int)


    predictions_file = open("leaf_classification.csv", "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["id","Species"])
    open_file_object.writerows(zip(ids, output))
    predictions_file.close()
    print 'Done.'
if __name__ == '__main__':
    X_train, y_train = prepTrainingData()
    X_test = prepareTestData()
    #TestArea('xgboost', X_train, y_train, X_test)


    
