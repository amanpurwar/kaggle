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
from  numpy.lib.recfunctions import append_fields
from sklearn.feature_selection import SelectKBest
import matplotlib.pyplot as plt

# global variables
Ports_dict = {}
ids=None
name_sorted=[]
clf_kbest=SelectKBest(k=176)

def prepTrainingData():
    global ids,clf_kbest
    #loading data into dataframe
    train_df = pd.read_csv('train.csv', header=0)

    #drop the id column
    train_df=train_df.drop('id',axis=1)

    #replacing the species with a number/categorising them
    Ports = list(enumerate(np.unique(train_df['species'])))    # determine all values of species,
    name_sorted.append('id')
    for i,name in Ports:
        name_sorted.append(name)

    name_sorted[1:]=sorted(name_sorted[1:])
    #print name_sorted

    Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index

 

    train_df.species = train_df.species.map( lambda x: Ports_dict[x]).astype(int)     # Convert all species strings to int
    
    train_data=train_df.values
    clf_kbest=clf_kbest.fit(train_data[:,1:],train_data[:,0])
    train_data_X=clf_kbest.transform(train_data[:,1:])

    return train_data_X,train_data[:,0]

def prepareTestData():
    global Ports_dict,ids,clf_kbest
    #test data file load
    test_df=pd.read_csv('test.csv',header=0)
    
    ids=test_df['id'].values # collecting id of test data before dropping id field
    test_df=test_df.drop('id',axis=1)
    

    #converting to numpy arr
    test_data=test_df.values
    test_data=clf_kbest.transform(test_data)

    return test_data

def TestArea(algo_str,X_train,y_train,X_test):
    global ids
    print 'testing started...'
    if algo_str is "rforest":
        
        forest = RandomForestClassifier(n_estimators=400,criterion='entropy',min_samples_split=7,max_features=0.5)
        forest = forest.fit( X_train, y_train )
        #for printing the importances
        '''
        importances = forest.feature_importances_
        indices = np.argsort(importances)[::-1]
        std = np.std([tree.feature_importances_ for tree in forest.estimators_],
            axis=0)
        for f in range(0,176):
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(0,176), importances[indices],
        color="r", yerr=indices, align="center")
        plt.show()
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
    #output = clf.predict(X_test).astype(int)
    output = clf.predict_proba(X_test)
    print output.shape
    print ids
    ids=np.reshape(ids,(594,1))
    print ids.shape
    predictions_file = open("output.csv", "wb")

    '''
    open_file_object = csv.writer(predictions_file)
    new_name=[]
    new_name.append('id')
    for i in name_sorted:
        new_name.append(i)
    #print new_name.shape()
    open_file_object.writerows(zip("id",name_sorted))
    #open_file_object.writerow(["id",[i for i in name_sorted]])
    #open_file_object.writerow("id")
    output=np.concatenate((ids, output), axis=1)
    print output.shape
    outpu=np.concatenate((new_name,output),axis=0)
    np.savetxt("foo.csv",output, delimiter=",")
    #open_file_object.writerows(zip(ids, output))
    open_file_object.writerows(zip(ids,[i for i in output])) 
    '''
    
    #for i in range(0,len(ids)):
    #    ids[i]=ids[i].astype('int32')
    #id = np.rec.array(output, dtype=[('id', np.int64)])
    #new_a = append_fields(id,'output', output, usemask=False, dtypes=[np.float64])
    
    output=np.concatenate((ids, output), axis=1)
    open_file_object = csv.writer(predictions_file,quoting = csv.QUOTE_MINIMAL,lineterminator='\n')
    open_file_object.writerow(name_sorted)
    open_file_object.writerows(output)
    
    #open_file_object.writerows(zip(ids, output))

    predictions_file.close()
    print 'Done.'

if __name__ == '__main__':
    X_train,y_train = prepTrainingData()
    X_test = prepareTestData()
    TestArea('rforest',X_train,y_train,X_test)


    
