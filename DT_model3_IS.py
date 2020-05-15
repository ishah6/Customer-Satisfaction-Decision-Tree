# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 15:40:11 2019

@author: ishani
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt


#Read training data file
trainfile = open(r"C:\Users\ishan\Documents\MS\ASU\508 Data Mining 1\Assignment 1\Santander Customer Satisfaction - TRAIN.csv",'r')
trainData = pd.read_csv(trainfile)

#Read test data file
testfile = open(r"C:\Users\ishan\Documents\MS\ASU\508 Data Mining 1\Assignment 1\Santander Customer Satisfaction - TEST-Without TARGET.csv",'r')
testData = pd.read_csv(testfile)

#print("=======")
testData.head()
trainData.head()

#Copy Train data excluding target AND ID
trainData_Copy = trainData.iloc[:,1:-1].copy()
#Copy Test data excluding ID
testData_Copy = testData.iloc[:,1:].copy()

trainData_Copy.head()
testData_Copy.head()



#Separate Train data and test data
X_Train = trainData_Copy
X_Test = testData_Copy

X_Train.head()
X_Test.head()

#Select just Target Column
Y_Train = trainData.iloc[:, -1]
#Y_Test = testData

#Create Decision Tree Classifier
clf=DecisionTreeClassifier(max_depth=20,criterion='gini',min_samples_leaf=10)

#Apply Classifier on Train and Target
clf.fit(X_Train,Y_Train)

#Get Class Prediction as a data frame with header as Prediction
pred=pd.DataFrame(clf.predict(X_Train),columns=["Prediction"])

pred.head()

#Get Class Prediction probabilities as a data frame 
#Get Prediction Probability for the predicted class as a dataframe
pred_Probability =pd.DataFrame(clf.predict_proba(X_Train))

pred_Probability.head()

#Write into a file with actual prediction and corresponding probability
pd.concat([pred,pred_Probability],axis=1).to_csv(r"C:\Users\ishan\Documents\MS\ASU\508 Data Mining 1\Assignment 1\TrainResult3.csv", index = None)

res=pd.read_csv(r'C:\Users\ishan\Documents\MS\ASU\508 Data Mining 1\Assignment 1\TrainResult3.csv')
res.head()

#Print Classification Report
print(classification_report(Y_Train,pred))

#Testset prediction======================================================================
#Get Class Prediction as a data frame with header as Prediction
pred=pd.DataFrame(clf.predict(X_Test),columns=["TARGET"])

pred.head()

#Get Class Prediction probabilities as a data frame 
#Get Prediction Probability for the predicted class as a dataframe
pred_Probability =pd.DataFrame(clf.predict_proba(X_Test))

pred_Probability.head()

#Write into a file with actual prediction and corresponding probability
#pd.concat([pred,pred_Probability],axis=1).to_csv(r"C:/Users/ishan/Documents/MS/ASU/508 Data Mining 1/Assignment 1/Result1.csv", index = None)

ID_Test = testData.iloc[:, 0]
ID_Test.head()

pd.concat([ID_Test,pred],axis=1).to_csv(r"C:/Users/ishan/Documents/MS/ASU/508 Data Mining 1/Assignment 1/TestResult3.csv", index = None)
res=pd.read_csv('C:/Users/ishan/Documents/MS/ASU/508 Data Mining 1/Assignment 1/TestResult3.csv')
res.head()

#Print Classification Report
#print(classification_report(Y_Test,pred))

#show datatypes and data shape
#print(trainData.shape)
#print(trainData.dtypes)

# Describe data - EDA
print(trainData.describe())


